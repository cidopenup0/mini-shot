import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional


class PredictionLogger:
    """
    Handles logging of predictions to SQLite database
    """
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def initialize(self) -> None:
        """Initialize the database and create tables"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    disease TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    metadata TEXT
                )
            """)
            
            # Create feedback table for user corrections
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER,
                    filename TEXT NOT NULL,
                    predicted_class TEXT NOT NULL,
                    correct_class TEXT NOT NULL,
                    is_correct BOOLEAN NOT NULL,
                    confidence REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    feedback_timestamp TEXT NOT NULL,
                    image_path TEXT,
                    FOREIGN KEY (prediction_id) REFERENCES predictions (id)
                )
            """)
            conn.commit()
    
    def log_prediction(
        self,
        filename: str,
        disease: str,
        confidence: float,
        timestamp: datetime,
        metadata: Optional[Dict] = None
    ) -> int:
        """
        Log a prediction to the database
        
        Args:
            filename: Name of the uploaded file
            disease: Predicted disease class
            confidence: Confidence score
            timestamp: Prediction timestamp
            metadata: Optional additional metadata
            
        Returns:
            prediction_id: The ID of the inserted prediction
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO predictions 
                    (filename, disease, confidence, timestamp, metadata)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    filename,
                    disease,
                    confidence,
                    timestamp.isoformat(),
                    json.dumps(metadata) if metadata else None
                ))
                conn.commit()
                return cursor.lastrowid
        except Exception as e:
            print(f"Failed to log prediction: {str(e)}")
            return None
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """
        Retrieve recent predictions from the database with feedback status
        
        Args:
            limit: Number of predictions to retrieve
        
        Returns:
            List of prediction dictionaries with feedback status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                # Join predictions with feedback to get feedback status
                cursor.execute("""
                    SELECT 
                        p.id, 
                        p.filename, 
                        p.disease, 
                        p.confidence, 
                        p.timestamp, 
                        p.metadata,
                        f.is_correct,
                        f.correct_class
                    FROM predictions p
                    LEFT JOIN feedback f ON p.id = f.prediction_id
                    ORDER BY p.id DESC
                    LIMIT ?
                """, (limit,))
                
                rows = cursor.fetchall()
                
                predictions = []
                for row in rows:
                    pred = {
                        "id": row[0],
                        "filename": row[1],
                        "disease": row[2],
                        "confidence": row[3],
                        "timestamp": row[4],
                        "metadata": json.loads(row[5]) if row[5] else None,
                        "feedback_status": None if row[6] is None else ("correct" if row[6] else "incorrect"),
                        "correct_class": row[7] if row[7] else None
                    }
                    predictions.append(pred)
                
                return predictions
        except Exception as e:
            print(f"Failed to retrieve predictions: {str(e)}")
            return []
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about predictions
        
        Returns:
            Dictionary with statistics
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Total predictions
                cursor.execute("SELECT COUNT(*) FROM predictions")
                total = cursor.fetchone()[0]
                
                # Most common diseases
                cursor.execute("""
                    SELECT disease, COUNT(*) as count
                    FROM predictions
                    GROUP BY disease
                    ORDER BY count DESC
                    LIMIT 5
                """)
                common_diseases = [
                    {"disease": row[0], "count": row[1]}
                    for row in cursor.fetchall()
                ]
                
                # Average confidence
                cursor.execute("SELECT AVG(confidence) FROM predictions")
                avg_confidence = cursor.fetchone()[0]
                
                return {
                    "total_predictions": total,
                    "common_diseases": common_diseases,
                    "average_confidence": avg_confidence
                }
        except Exception as e:
            print(f"Failed to get statistics: {str(e)}")
            return {}
    
    def log_feedback(
        self,
        prediction_id: Optional[int],
        filename: str,
        predicted_class: str,
        correct_class: str,
        is_correct: bool,
        confidence: float,
        original_timestamp: str,
        image_path: Optional[str] = None
    ) -> bool:
        """
        Log user feedback for a prediction
        
        Args:
            prediction_id: ID of the original prediction (if available)
            filename: Original filename
            predicted_class: What the model predicted
            correct_class: The correct class according to user
            is_correct: Whether the prediction was correct
            confidence: Confidence of the original prediction
            original_timestamp: Timestamp of original prediction
            image_path: Optional path to saved image for retraining
            
        Returns:
            bool: Success status
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO feedback 
                    (prediction_id, filename, predicted_class, correct_class, 
                     is_correct, confidence, timestamp, feedback_timestamp, image_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prediction_id,
                    filename,
                    predicted_class,
                    correct_class,
                    is_correct,
                    confidence,
                    original_timestamp,
                    datetime.now().isoformat(),
                    image_path
                ))
                conn.commit()
                return True
        except Exception as e:
            print(f"Failed to log feedback: {str(e)}")
            return False
    
    def get_feedback_for_retraining(self) -> List[Dict]:
        """
        Get all feedback data formatted for model retraining
        
        Returns:
            List of feedback entries with corrected labels
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, filename, predicted_class, correct_class, 
                           is_correct, confidence, timestamp, image_path
                    FROM feedback
                    ORDER BY feedback_timestamp DESC
                """)
                
                rows = cursor.fetchall()
                
                feedback_data = []
                for row in rows:
                    entry = {
                        "feedback_id": row[0],
                        "filename": row[1],
                        "predicted_class": row[2],
                        "correct_class": row[3],
                        "is_correct": bool(row[4]),
                        "confidence": row[5],
                        "timestamp": row[6],
                        "image_path": row[7]
                    }
                    feedback_data.append(entry)
                
                return feedback_data
        except Exception as e:
            print(f"Failed to retrieve feedback: {str(e)}")
            return []

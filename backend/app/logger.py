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
            conn.commit()
    
    def log_prediction(
        self,
        filename: str,
        disease: str,
        confidence: float,
        timestamp: datetime,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Log a prediction to the database
        
        Args:
            filename: Name of the uploaded file
            disease: Predicted disease class
            confidence: Confidence score
            timestamp: Prediction timestamp
            metadata: Optional additional metadata
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
        except Exception as e:
            print(f"Failed to log prediction: {str(e)}")
    
    def get_recent_predictions(self, limit: int = 50) -> List[Dict]:
        """
        Retrieve recent predictions from the database
        
        Args:
            limit: Number of predictions to retrieve
        
        Returns:
            List of prediction dictionaries
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, filename, disease, confidence, timestamp, metadata
                    FROM predictions
                    ORDER BY id DESC
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
                        "metadata": json.loads(row[5]) if row[5] else None
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

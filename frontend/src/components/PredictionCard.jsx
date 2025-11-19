import React, { useState } from 'react';
import { formatClassName, formatConfidence } from '../utils/helpers';
import { api } from '../services/api';

const PredictionCard = ({ prediction, imageUrl }) => {
  const [feedbackGiven, setFeedbackGiven] = useState(false);
  const [feedbackLoading, setFeedbackLoading] = useState(false);
  const [showCorrectClassInput, setShowCorrectClassInput] = useState(false);
  const [correctClass, setCorrectClass] = useState('');
  
  if (!prediction) return null;

  const confidencePercentage = prediction.confidence * 100;
  const isHighConfidence = confidencePercentage >= 80;
  const isMediumConfidence = confidencePercentage >= 50 && confidencePercentage < 80;

  const handleFeedback = async (isCorrect) => {
    if (feedbackLoading) return;
    
    if (!isCorrect && !showCorrectClassInput) {
      setShowCorrectClassInput(true);
      return;
    }

    setFeedbackLoading(true);
    
    const feedbackData = {
      prediction_id: prediction.prediction_id || null,
      filename: prediction.filename,
      predicted_class: prediction.class,
      is_correct: isCorrect,
      correct_class: isCorrect ? prediction.class : correctClass,
      confidence: prediction.confidence,
      timestamp: prediction.timestamp
    };

    const result = await api.submitFeedback(feedbackData);
    
    setFeedbackLoading(false);
    
    if (result.success) {
      setFeedbackGiven(true);
      setShowCorrectClassInput(false);
    } else {
      alert('Failed to submit feedback. Please try again.');
    }
  };

  return (
    <div className="card mt-6 bg-gradient-to-br from-primary-50 to-white border-2 border-primary-200">
      <div className="flex items-center mb-4">
        <div className="w-12 h-12 bg-primary-600 rounded-full flex items-center justify-center">
          <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </div>
        <h2 className="text-2xl font-bold text-gray-800 ml-4">Prediction Result</h2>
      </div>

      <div className="space-y-4">
        <div className="bg-white rounded-lg p-4 shadow-sm">
          <p className="text-sm font-medium text-gray-600 mb-2">Detected Disease:</p>
          <p className="text-2xl font-bold text-primary-700">{formatClassName(prediction.class)}</p>
        </div>

        <div className="bg-white rounded-lg p-4 shadow-sm">
          <div className="flex justify-between items-center mb-2">
            <p className="text-sm font-medium text-gray-600">Confidence:</p>
            <span className={`text-xl font-bold ${
              isHighConfidence ? 'text-green-600' : 
              isMediumConfidence ? 'text-yellow-600' : 
              'text-red-600'
            }`}>
              {formatConfidence(prediction.confidence)}
            </span>
          </div>
          
          <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${
                isHighConfidence ? 'bg-green-600' : 
                isMediumConfidence ? 'bg-yellow-600' : 
                'bg-red-600'
              }`}
              style={{ width: `${confidencePercentage}%` }}
            ></div>
          </div>
          
          <p className="text-xs text-gray-500 mt-2">
            {isHighConfidence && 'High confidence - Result is very reliable'}
            {isMediumConfidence && 'Medium confidence - Result may need verification'}
            {!isHighConfidence && !isMediumConfidence && 'Low confidence - Consider retaking the image'}
          </p>
        </div>

        {imageUrl && (
          <div className="bg-white rounded-lg p-4 shadow-sm">
            <p className="text-sm font-medium text-gray-600 mb-2">Analyzed Image:</p>
            <img
              src={imageUrl}
              alt="Analyzed"
              className="w-full max-w-md mx-auto rounded-lg border-2 border-gray-200"
            />
          </div>
        )}

        {/* Feedback Section */}
        <div className="bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg p-5 border-2 border-blue-200">
          {!feedbackGiven ? (
            <>
              <div className="flex items-center mb-3">
                <svg className="w-5 h-5 text-blue-600 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <h3 className="text-lg font-bold text-gray-800">Was this prediction correct?</h3>
              </div>
              <p className="text-sm text-gray-600 mb-4">Your feedback helps improve the model accuracy</p>
              
              {!showCorrectClassInput ? (
                <div className="flex gap-3">
                  <button
                    onClick={() => handleFeedback(true)}
                    disabled={feedbackLoading}
                    className="flex-1 px-6 py-3 bg-green-600 text-white font-semibold rounded-lg shadow-md hover:bg-green-700 hover:shadow-lg transition-all duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                    Yes, Correct
                  </button>
                  <button
                    onClick={() => handleFeedback(false)}
                    disabled={feedbackLoading}
                    className="flex-1 px-6 py-3 bg-red-600 text-white font-semibold rounded-lg shadow-md hover:bg-red-700 hover:shadow-lg transition-all duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center"
                  >
                    <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                    No, Incorrect
                  </button>
                </div>
              ) : (
                <div className="space-y-3">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      What is the correct disease class?
                    </label>
                    <input
                      type="text"
                      value={correctClass}
                      onChange={(e) => setCorrectClass(e.target.value)}
                      placeholder="e.g., Tomato Early Blight"
                      className="w-full px-4 py-2 border-2 border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none"
                    />
                  </div>
                  <div className="flex gap-3">
                    <button
                      onClick={() => handleFeedback(false)}
                      disabled={feedbackLoading || !correctClass.trim()}
                      className="flex-1 px-6 py-3 bg-blue-600 text-white font-semibold rounded-lg shadow-md hover:bg-blue-700 hover:shadow-lg transition-all duration-200 disabled:bg-gray-400 disabled:cursor-not-allowed"
                    >
                      {feedbackLoading ? 'Submitting...' : 'Submit Correction'}
                    </button>
                    <button
                      onClick={() => {
                        setShowCorrectClassInput(false);
                        setCorrectClass('');
                      }}
                      disabled={feedbackLoading}
                      className="px-6 py-3 bg-gray-300 text-gray-700 font-semibold rounded-lg hover:bg-gray-400 transition-all duration-200"
                    >
                      Cancel
                    </button>
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="flex items-center justify-center py-3">
              <svg className="w-8 h-8 text-green-600 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <p className="text-lg font-bold text-green-800">Thank you for your feedback!</p>
                <p className="text-sm text-green-600">Your input helps improve the model</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default PredictionCard;

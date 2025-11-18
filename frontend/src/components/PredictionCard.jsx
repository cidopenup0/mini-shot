import React from 'react';
import { formatClassName, formatConfidence } from '../utils/helpers';

const PredictionCard = ({ prediction, imageUrl }) => {
  if (!prediction) return null;

  const confidencePercentage = prediction.confidence * 100;
  const isHighConfidence = confidencePercentage >= 80;
  const isMediumConfidence = confidencePercentage >= 50 && confidencePercentage < 80;

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
      </div>
    </div>
  );
};

export default PredictionCard;

import React from 'react';
import { formatFileSize } from '../utils/helpers';

const PreviewCard = ({ file, imageUrl, onRemove }) => {
  if (!file || !imageUrl) return null;

  return (
    <div className="card mt-6">
      <div className="flex items-start justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800">Image Preview</h3>
        <button
          onClick={onRemove}
          className="text-red-600 hover:text-red-800 transition-colors"
          title="Remove image"
        >
          <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>
      
      <div className="flex flex-col md:flex-row gap-4">
        <div className="flex-shrink-0">
          <img
            src={imageUrl}
            alt="Preview"
            className="w-full md:w-64 h-64 object-cover rounded-lg border-2 border-gray-200"
          />
        </div>
        
        <div className="flex-1">
          <div className="bg-gray-50 rounded-lg p-4 space-y-2">
            <div className="flex justify-between">
              <span className="text-sm font-medium text-gray-600">Filename:</span>
              <span className="text-sm text-gray-800 truncate ml-2">{file.name}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm font-medium text-gray-600">Size:</span>
              <span className="text-sm text-gray-800">{formatFileSize(file.size)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-sm font-medium text-gray-600">Type:</span>
              <span className="text-sm text-gray-800">{file.type}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PreviewCard;

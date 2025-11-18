import React, { useState } from 'react';
import ImageUpload from '../components/ImageUpload';
import PreviewCard from '../components/PreviewCard';
import PredictionCard from '../components/PredictionCard';
import Loader from '../components/Loader';
import ErrorBox from '../components/ErrorBox';
import { api } from '../services/api';

const Home = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleFileSelect = (file) => {
    setSelectedFile(file);
    setPrediction(null);
    setError(null);
    
    // Create preview URL
    const url = URL.createObjectURL(file);
    setImageUrl(url);
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPrediction(null);
    setError(null);
    
    if (imageUrl) {
      URL.revokeObjectURL(imageUrl);
      setImageUrl(null);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      setError('Please select an image first');
      return;
    }

    setLoading(true);
    setError(null);
    setPrediction(null);

    const result = await api.predictDisease(selectedFile);

    setLoading(false);

    if (result.success) {
      setPrediction(result.data);
    } else {
      setError(result.error);
    }
  };

  const handleReset = () => {
    handleRemoveImage();
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="text-center mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-4">
          Plant Disease Detection System
        </h1>
        <p className="text-lg text-gray-600">
          Upload a leaf image to detect plant diseases using AI
        </p>
      </div>

      <div className="card">
        <ImageUpload 
          onFileSelect={handleFileSelect} 
          disabled={loading}
        />

        {selectedFile && imageUrl && (
          <PreviewCard
            file={selectedFile}
            imageUrl={imageUrl}
            onRemove={handleRemoveImage}
          />
        )}

        {selectedFile && !prediction && !loading && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={handlePredict}
              className="btn-primary"
              disabled={loading}
            >
              <span className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Predict Disease
              </span>
            </button>
          </div>
        )}

        {loading && <Loader text="Analyzing image..." />}

        {error && (
          <div className="mt-6">
            <ErrorBox message={error} onRetry={handlePredict} />
          </div>
        )}
      </div>

      {prediction && (
        <>
          <PredictionCard prediction={prediction} imageUrl={imageUrl} />
          
          <div className="mt-6 flex justify-center space-x-4">
            <button
              onClick={handleReset}
              className="btn-primary"
            >
              Analyze Another Image
            </button>
          </div>
        </>
      )}

      <div className="mt-8 card bg-blue-50 border border-blue-200">
        <div className="flex items-start">
          <svg className="w-6 h-6 text-blue-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800">Tips for best results</h3>
            <ul className="mt-2 text-sm text-blue-700 list-disc list-inside space-y-1">
              <li>Take clear, well-lit photos of the affected leaf</li>
              <li>Focus on the diseased area</li>
              <li>Avoid blurry or dark images</li>
              <li>Use JPEG or PNG format, max 10MB</li>
            </ul>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;

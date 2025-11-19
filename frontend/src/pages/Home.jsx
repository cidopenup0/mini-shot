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
    <div className="max-w-5xl mx-auto">
      {/* Hero Section */}
      <div className="text-center mb-10 animate-fadeIn">
        <div className="inline-block mb-4">
          <div className="w-20 h-20 bg-gradient-to-br from-primary-400 to-primary-600 rounded-2xl flex items-center justify-center shadow-xl mx-auto transform hover:scale-110 transition-transform duration-300">
            <svg className="w-12 h-12 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
        </div>
        <h1 className="text-5xl md:text-6xl font-extrabold bg-gradient-to-r from-primary-600 via-primary-700 to-primary-900 bg-clip-text text-transparent mb-4">
          Plant Disease Detection
        </h1>
        <p className="text-xl text-gray-600 max-w-2xl mx-auto">
          Upload a leaf image to detect plant diseases using advanced AI technology
        </p>
        <div className="flex items-center justify-center gap-4 mt-6">
          <span className="px-4 py-2 bg-primary-100 text-primary-700 rounded-full text-sm font-semibold">
            ✓ 38 Disease Classes
          </span>
          <span className="px-4 py-2 bg-primary-100 text-primary-700 rounded-full text-sm font-semibold">
            ✓ 95%+ Accuracy
          </span>
          <span className="px-4 py-2 bg-primary-100 text-primary-700 rounded-full text-sm font-semibold">
            ✓ Instant Results
          </span>
        </div>
      </div>

      {/* Main Upload/Preview Card */}
      {!selectedFile && !loading && !prediction && (
        <div className="card hover:shadow-2xl transition-shadow duration-300 animate-slideUp">
          <ImageUpload 
            onFileSelect={handleFileSelect} 
            disabled={loading}
          />
        </div>
      )}

      {selectedFile && imageUrl && !prediction && !loading && (
        <div className="card hover:shadow-2xl transition-shadow duration-300 animate-fadeIn">
          <PreviewCard
            file={selectedFile}
            imageUrl={imageUrl}
            onRemove={handleRemoveImage}
          />
          
          <div className="mt-6 flex justify-center">
            <button
              onClick={handlePredict}
              className="btn-primary px-10 py-4 text-lg font-bold shadow-xl hover:shadow-2xl transform hover:scale-105 active:scale-95"
            >
              <span className="flex items-center">
                <svg className="w-6 h-6 mr-3 animate-pulse" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
                Analyze & Predict Disease
              </span>
            </button>
          </div>
        </div>
      )}

      {loading && (
        <div className="card animate-fadeIn">
          <Loader text="Analyzing image with AI..." />
        </div>
      )}

      {error && !loading && (
        <div className="card animate-shake">
          <ErrorBox message={error} onRetry={handlePredict} />
        </div>
      )}

      {prediction && (
        <div className="animate-fadeIn">
          <div className="relative overflow-hidden rounded-2xl shadow-2xl">
            <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-primary-50 opacity-50"></div>
            <div className="relative">
              <PredictionCard prediction={prediction} imageUrl={imageUrl} />
            </div>
          </div>
          
          <div className="mt-8 flex justify-center space-x-4">
            <button
              onClick={handleReset}
              className="btn-primary px-8 py-3 text-lg shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95 flex items-center"
            >
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Analyze Another Image
            </button>
          </div>
        </div>
      )}

      {/* Tips Section */}
      <div className="mt-8 relative overflow-hidden rounded-2xl shadow-lg hover:shadow-xl transition-shadow duration-300">
        <div className="absolute inset-0 bg-gradient-to-br from-blue-50 via-white to-indigo-50"></div>
        <div className="relative card border-2 border-blue-100">
          <div className="flex items-start">
            <div className="flex-shrink-0 w-12 h-12 rounded-full bg-gradient-to-br from-blue-500 to-indigo-600 flex items-center justify-center shadow-lg">
              <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
            <div className="ml-4 flex-1">
              <h3 className="text-lg font-bold text-gray-900 mb-3">📸 Tips for Best Results</h3>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                <div className="flex items-start">
                  <span className="text-primary-600 font-bold mr-2">✓</span>
                  <p className="text-sm text-gray-700">Take clear, well-lit photos of the affected leaf</p>
                </div>
                <div className="flex items-start">
                  <span className="text-primary-600 font-bold mr-2">✓</span>
                  <p className="text-sm text-gray-700">Focus on the diseased area for better accuracy</p>
                </div>
                <div className="flex items-start">
                  <span className="text-primary-600 font-bold mr-2">✓</span>
                  <p className="text-sm text-gray-700">Avoid blurry, dark, or low-resolution images</p>
                </div>
                <div className="flex items-start">
                  <span className="text-primary-600 font-bold mr-2">✓</span>
                  <p className="text-sm text-gray-700">Use JPEG or PNG format (max 10MB size)</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;

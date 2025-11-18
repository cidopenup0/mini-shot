import React, { useState, useEffect } from 'react';
import Loader from '../components/Loader';
import ErrorBox from '../components/ErrorBox';
import { api } from '../services/api';

const Health = () => {
  const [healthData, setHealthData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHealth = async () => {
    setLoading(true);
    setError(null);

    const result = await api.checkHealth();

    setLoading(false);

    if (result.success) {
      setHealthData(result.data);
    } else {
      setError(result.error);
    }
  };

  useEffect(() => {
    fetchHealth();
    
    // Auto-refresh every 30 seconds
    const interval = setInterval(fetchHealth, 30000);
    
    return () => clearInterval(interval);
  }, []);

  const getStatusColor = (status) => {
    return status ? 'text-green-600' : 'text-red-600';
  };

  const getStatusBgColor = (status) => {
    return status ? 'bg-green-100' : 'bg-red-100';
  };

  const getStatusText = (status) => {
    return status ? 'Online' : 'Offline';
  };

  return (
    <div className="max-w-4xl mx-auto">
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          System Health Status
        </h1>
        <p className="text-lg text-gray-600">
          Monitor the backend and model status
        </p>
      </div>

      {loading && <Loader text="Checking system health..." />}

      {error && !loading && (
        <div className="space-y-4">
          <ErrorBox message={error} onRetry={fetchHealth} />
          
          <div className="card bg-red-50 border-2 border-red-200">
            <div className="flex items-center">
              <div className="w-16 h-16 bg-red-600 rounded-full flex items-center justify-center animate-pulse">
                <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <div className="ml-6">
                <h2 className="text-2xl font-bold text-red-800">Backend Offline</h2>
                <p className="text-red-700 mt-1">Unable to connect to the server</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {!loading && !error && healthData && (
        <div className="space-y-6">
          {/* Overall Status */}
          <div className={`card ${healthData.status === 'healthy' ? 'bg-green-50 border-2 border-green-200' : 'bg-yellow-50 border-2 border-yellow-200'}`}>
            <div className="flex items-center">
              <div className={`w-16 h-16 ${healthData.status === 'healthy' ? 'bg-green-600' : 'bg-yellow-600'} rounded-full flex items-center justify-center`}>
                <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  {healthData.status === 'healthy' ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  )}
                </svg>
              </div>
              <div className="ml-6">
                <h2 className="text-2xl font-bold text-gray-800">
                  {healthData.status === 'healthy' ? 'System Healthy' : 'System Warning'}
                </h2>
                <p className="text-gray-600 mt-1">Backend is operational</p>
              </div>
            </div>
          </div>

          {/* Detailed Status */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Backend Status */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">Backend Server</h3>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBgColor(true)} ${getStatusColor(true)}`}>
                  {getStatusText(true)}
                </span>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Status:</span>
                  <span className="font-medium text-gray-800">{healthData.status || 'N/A'}</span>
                </div>
                {healthData.timestamp && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Last Check:</span>
                    <span className="font-medium text-gray-800">
                      {new Date(healthData.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Model Status */}
            <div className="card">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-800">AI Model</h3>
                <span className={`px-3 py-1 rounded-full text-sm font-medium ${getStatusBgColor(healthData.model_loaded)} ${getStatusColor(healthData.model_loaded)}`}>
                  {healthData.model_loaded ? 'Loaded' : 'Not Loaded'}
                </span>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-600">Model Loaded:</span>
                  <span className="font-medium text-gray-800">
                    {healthData.model_loaded ? 'Yes' : 'No'}
                  </span>
                </div>
                {healthData.model_version && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Version:</span>
                    <span className="font-medium text-gray-800">{healthData.model_version}</span>
                  </div>
                )}
                {healthData.num_classes && (
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-600">Classes:</span>
                    <span className="font-medium text-gray-800">{healthData.num_classes}</span>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Additional Info */}
          {healthData.message && (
            <div className="card bg-blue-50 border border-blue-200">
              <div className="flex items-start">
                <svg className="w-6 h-6 text-blue-600 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <div className="ml-3">
                  <h3 className="text-sm font-medium text-blue-800">System Message</h3>
                  <p className="mt-1 text-sm text-blue-700">{healthData.message}</p>
                </div>
              </div>
            </div>
          )}

          {/* Refresh Button */}
          <div className="flex justify-center">
            <button
              onClick={fetchHealth}
              className="btn-primary"
            >
              <span className="flex items-center">
                <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh Status
              </span>
            </button>
          </div>
        </div>
      )}
    </div>
  );
};

export default Health;

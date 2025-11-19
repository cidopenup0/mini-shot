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
      {/* Header Section */}
      <div className="mb-8 animate-slideUp">
        <div className="flex items-center mb-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-blue-500 to-cyan-600 flex items-center justify-center shadow-lg mr-4">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-blue-600 via-cyan-600 to-blue-800 bg-clip-text text-transparent">
              System Health
            </h1>
            <p className="text-lg text-gray-600 mt-1">
              Monitor backend and AI model status in real-time
            </p>
          </div>
        </div>
      </div>

      {loading && (
        <div className="animate-fadeIn">
          <Loader text="Checking system health..." />
        </div>
      )}

      {error && !loading && (
        <div className="space-y-4 animate-shake">
          <ErrorBox message={error} onRetry={fetchHealth} />
          
          <div className="card bg-gradient-to-br from-red-50 to-red-100 border-2 border-red-300 shadow-xl">
            <div className="flex items-center">
              <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-red-700 rounded-full flex items-center justify-center animate-pulse shadow-lg">
                <svg className="w-8 h-8 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </div>
              <div className="ml-6">
                <h2 className="text-2xl font-bold text-red-900">Backend Offline</h2>
                <p className="text-red-700 mt-1">Unable to connect to the server</p>
                <p className="text-sm text-red-600 mt-2">Please ensure the backend is running on port 8000</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {!loading && !error && healthData && (
        <div className="space-y-6 animate-fadeIn">
          {/* Overall Status */}
          <div className={`card shadow-xl hover:shadow-2xl transition-shadow duration-300 ${healthData.status === 'healthy' ? 'bg-gradient-to-br from-green-50 to-emerald-50 border-2 border-green-300' : 'bg-gradient-to-br from-yellow-50 to-amber-50 border-2 border-yellow-300'}`}>
            <div className="flex items-center">
              <div className={`w-20 h-20 ${healthData.status === 'healthy' ? 'bg-gradient-to-br from-green-500 to-emerald-600' : 'bg-gradient-to-br from-yellow-500 to-amber-600'} rounded-full flex items-center justify-center shadow-lg ${healthData.status === 'healthy' ? 'animate-pulse' : ''}`}>
                <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  {healthData.status === 'healthy' ? (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  ) : (
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                  )}
                </svg>
              </div>
              <div className="ml-6">
                <h2 className="text-3xl font-bold text-gray-900">
                  {healthData.status === 'healthy' ? 'System Healthy' : 'System Warning'}
                </h2>
                <p className="text-gray-600 mt-1 text-lg">Backend is operational and ready</p>
              </div>
            </div>
          </div>

          {/* Detailed Status */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Backend Status */}
            <div className="card hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-primary-200">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-blue-100 to-blue-200 flex items-center justify-center mr-3">
                    <svg className="w-6 h-6 text-blue-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2m-2-4h.01M17 16h.01" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-gray-800">Backend Server</h3>
                </div>
                <span className={`px-4 py-2 rounded-full text-sm font-bold shadow-md ${getStatusBgColor(true)} ${getStatusColor(true)}`}>
                  {getStatusText(true)}
                </span>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Status</span>
                  <span className="font-bold text-gray-800 uppercase">{healthData.status || 'N/A'}</span>
                </div>
                {healthData.timestamp && (
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-600 font-medium">Last Check</span>
                    <span className="font-medium text-gray-800">
                      {new Date(healthData.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                )}
              </div>
            </div>

            {/* Model Status */}
            <div className="card hover:shadow-xl transition-all duration-300 border border-gray-100 hover:border-primary-200">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center">
                  <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-100 to-purple-200 flex items-center justify-center mr-3">
                    <svg className="w-6 h-6 text-purple-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
                    </svg>
                  </div>
                  <h3 className="text-xl font-bold text-gray-800">AI Model</h3>
                </div>
                <span className={`px-4 py-2 rounded-full text-sm font-bold shadow-md ${getStatusBgColor(healthData.model_loaded)} ${getStatusColor(healthData.model_loaded)}`}>
                  {healthData.model_loaded ? 'Loaded' : 'Not Loaded'}
                </span>
              </div>
              <div className="space-y-4">
                <div className="flex justify-between items-center py-2 border-b border-gray-100">
                  <span className="text-gray-600 font-medium">Model Loaded</span>
                  <span className="font-bold text-gray-800">
                    {healthData.model_loaded ? 'Yes' : 'No'}
                  </span>
                </div>
                {healthData.model_version && (
                  <div className="flex justify-between items-center py-2 border-b border-gray-100">
                    <span className="text-gray-600 font-medium">Version</span>
                    <span className="font-medium text-gray-800">{healthData.model_version}</span>
                  </div>
                )}
                {healthData.num_classes && (
                  <div className="flex justify-between items-center py-2">
                    <span className="text-gray-600 font-medium">Classes</span>
                    <span className="font-bold text-primary-600 text-lg">{healthData.num_classes}</span>
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
          <div className="flex justify-center pt-4">
            <button
              onClick={fetchHealth}
              className="btn-primary px-8 py-3 text-lg shadow-lg hover:shadow-xl transform hover:scale-105 active:scale-95"
            >
              <span className="flex items-center">
                <svg className="w-6 h-6 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                </svg>
                Refresh Status
              </span>
            </button>
          </div>
          
          {/* Auto-refresh indicator */}
          <p className="text-center text-sm text-gray-500 mt-4">
            ⏱ Auto-refreshes every 30 seconds
          </p>
        </div>
      )}
    </div>
  );
};

export default Health;

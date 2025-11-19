import React, { useState, useEffect } from 'react';
import HistoryTable from '../components/HistoryTable';
import Loader from '../components/Loader';
import ErrorBox from '../components/ErrorBox';
import { api } from '../services/api';

const History = () => {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const fetchHistory = async () => {
    setLoading(true);
    setError(null);

    const result = await api.getHistory();

    setLoading(false);

    if (result.success) {
      // Backend returns {success: true, count: X, predictions: [...]}
      const historyData = Array.isArray(result.data) 
        ? result.data 
        : result.data.predictions || result.data.history || [];
      setHistory(historyData);
    } else {
      setError(result.error);
    }
  };

  useEffect(() => {
    fetchHistory();
  }, []);

  return (
    <div className="max-w-6xl mx-auto">
      {/* Header Section */}
      <div className="mb-8 animate-slideUp">
        <div className="flex items-center mb-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center shadow-lg mr-4">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-purple-600 via-indigo-600 to-purple-800 bg-clip-text text-transparent">
              Prediction History
            </h1>
            <p className="text-lg text-gray-600 mt-1">
              View all your previous plant disease predictions
            </p>
          </div>
        </div>
      </div>

      {loading && (
        <div className="animate-fadeIn">
          <Loader text="Loading history..." />
        </div>
      )}

      {error && !loading && (
        <div className="animate-shake">
          <ErrorBox message={error} onRetry={fetchHistory} />
        </div>
      )}

      {!loading && !error && (
        <div className="animate-fadeIn">
          {/* Stats Bar */}
          <div className="mb-6 card bg-gradient-to-r from-purple-50 to-indigo-50 border border-purple-100 flex justify-between items-center">
            <div className="flex items-center">
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-purple-500 to-indigo-600 flex items-center justify-center mr-3">
                <span className="text-white font-bold text-lg">{history.length}</span>
              </div>
              <div>
                <p className="text-sm text-gray-500">Total Predictions</p>
                <p className="text-lg font-bold text-gray-800">{history.length} analyses</p>
              </div>
            </div>
            <button
              onClick={fetchHistory}
              className="px-4 py-2 bg-white rounded-lg shadow hover:shadow-md transition-all duration-200 flex items-center text-purple-600 font-medium"
            >
              <svg className="w-5 h-5 mr-2" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
          </div>

          <HistoryTable history={history} />
        </div>
      )}
    </div>
  );
};

export default History;

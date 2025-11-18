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
      // Ensure we have an array, handle both {history: [...]} and direct array responses
      const historyData = Array.isArray(result.data) ? result.data : result.data.history || [];
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
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          Prediction History
        </h1>
        <p className="text-lg text-gray-600">
          View all your previous plant disease predictions
        </p>
      </div>

      {loading && <Loader text="Loading history..." />}

      {error && !loading && (
        <ErrorBox message={error} onRetry={fetchHistory} />
      )}

      {!loading && !error && (
        <>
          <div className="mb-4 flex justify-between items-center">
            <p className="text-sm text-gray-600">
              Total predictions: <span className="font-semibold">{history.length}</span>
            </p>
            <button
              onClick={fetchHistory}
              className="text-sm text-primary-600 hover:text-primary-700 font-medium flex items-center"
            >
              <svg className="w-4 h-4 mr-1" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
              </svg>
              Refresh
            </button>
          </div>

          <HistoryTable history={history} />
        </>
      )}
    </div>
  );
};

export default History;

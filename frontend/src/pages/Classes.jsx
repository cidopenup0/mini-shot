import React, { useState, useEffect } from 'react';
import ClassesList from '../components/ClassesList';
import Loader from '../components/Loader';
import ErrorBox from '../components/ErrorBox';
import { api } from '../services/api';

const Classes = () => {
  const [classes, setClasses] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');

  const fetchClasses = async () => {
    setLoading(true);
    setError(null);

    const result = await api.getClasses();

    setLoading(false);

    if (result.success) {
      // Handle both {classes: [...]} and direct array responses
      const classesData = Array.isArray(result.data) ? result.data : result.data.classes || [];
      setClasses(classesData);
    } else {
      setError(result.error);
    }
  };

  useEffect(() => {
    fetchClasses();
  }, []);

  const filteredClasses = classes.filter(className =>
    className.toLowerCase().includes(searchTerm.toLowerCase())
  );

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header Section */}
      <div className="mb-8 animate-slideUp">
        <div className="flex items-center mb-3">
          <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-green-500 to-emerald-600 flex items-center justify-center shadow-lg mr-4">
            <svg className="w-6 h-6 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
            </svg>
          </div>
          <div>
            <h1 className="text-4xl md:text-5xl font-bold bg-gradient-to-r from-green-600 via-emerald-600 to-green-800 bg-clip-text text-transparent">
              Disease Classes
            </h1>
            <p className="text-lg text-gray-600 mt-1">
              Our model can detect <span className="font-bold text-primary-600">{classes.length}</span> different plant diseases
            </p>
          </div>
        </div>
      </div>

      {/* Search Bar */}
      {!loading && !error && (
        <div className="card mb-6 hover:shadow-xl transition-shadow duration-300 animate-fadeIn">
          <div className="flex items-center">
            <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-primary-100 to-primary-200 flex items-center justify-center mr-3">
              <svg className="w-5 h-5 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
              </svg>
            </div>
            <input
              type="text"
              placeholder="Search for diseases (e.g., Apple, Tomato, rust)..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1 outline-none text-gray-700 text-lg placeholder-gray-400"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="ml-2 w-8 h-8 rounded-full bg-gray-100 hover:bg-gray-200 flex items-center justify-center transition-colors"
              >
                <svg className="w-5 h-5 text-gray-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      )}

      {loading && (
        <div className="animate-fadeIn">
          <Loader text="Loading disease classes..." />
        </div>
      )}

      {error && !loading && (
        <div className="animate-shake">
          <ErrorBox message={error} onRetry={fetchClasses} />
        </div>
      )}

      {!loading && !error && (
        <div className="animate-fadeIn">
          {searchTerm && (
            <div className="mb-4 px-4 py-2 bg-primary-50 border border-primary-200 rounded-lg">
              <p className="text-sm text-primary-800">
                Showing <span className="font-bold">{filteredClasses.length}</span> of <span className="font-bold">{classes.length}</span> diseases
              </p>
            </div>
          )}
          
          {filteredClasses.length > 0 ? (
            <ClassesList classes={filteredClasses} />
          ) : (
            <div className="card text-center py-16 border-2 border-dashed border-gray-200">
              <div className="w-20 h-20 rounded-full bg-gray-100 flex items-center justify-center mx-auto mb-4">
                <svg className="w-10 h-10 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <p className="text-gray-500 text-xl font-medium mb-2">No diseases found</p>
              <p className="text-gray-400">No results matching "<span className="font-semibold">{searchTerm}</span>"</p>
              <button
                onClick={() => setSearchTerm('')}
                className="mt-4 text-primary-600 hover:text-primary-700 font-medium"
              >
                Clear search
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default Classes;

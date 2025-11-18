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
      <div className="mb-8">
        <h1 className="text-4xl font-bold text-gray-800 mb-2">
          Supported Disease Classes
        </h1>
        <p className="text-lg text-gray-600">
          Our model can detect {classes.length} different plant diseases
        </p>
      </div>

      {!loading && !error && (
        <div className="card mb-6">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-gray-400 mr-3" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z" />
            </svg>
            <input
              type="text"
              placeholder="Search for diseases..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              className="flex-1 outline-none text-gray-700"
            />
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="ml-2 text-gray-400 hover:text-gray-600"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>
      )}

      {loading && <Loader text="Loading disease classes..." />}

      {error && !loading && (
        <ErrorBox message={error} onRetry={fetchClasses} />
      )}

      {!loading && !error && (
        <>
          {searchTerm && (
            <p className="mb-4 text-sm text-gray-600">
              Showing {filteredClasses.length} of {classes.length} diseases
            </p>
          )}
          
          {filteredClasses.length > 0 ? (
            <ClassesList classes={filteredClasses} />
          ) : (
            <div className="card text-center py-12">
              <svg className="w-16 h-16 text-gray-300 mx-auto mb-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.172 16.172a4 4 0 015.656 0M9 10h.01M15 10h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-gray-500 text-lg">No diseases found matching "{searchTerm}"</p>
            </div>
          )}
        </>
      )}
    </div>
  );
};

export default Classes;

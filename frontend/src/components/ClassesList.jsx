import React from 'react';
import { formatClassName } from '../utils/helpers';

const ClassesList = ({ classes }) => {
  if (!classes || classes.length === 0) {
    return (
      <div className="card text-center py-12">
        <p className="text-gray-500">No disease classes available</p>
      </div>
    );
  }

  // Group classes by plant type
  const groupedClasses = classes.reduce((acc, className) => {
    const plantType = className.split('_')[0] || 'Other';
    if (!acc[plantType]) {
      acc[plantType] = [];
    }
    acc[plantType].push(className);
    return acc;
  }, {});

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
      {Object.entries(groupedClasses).map(([plantType, diseaseList]) => (
        <div key={plantType} className="card">
          <div className="flex items-center mb-4">
            <div className="w-10 h-10 bg-primary-100 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-primary-600" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 3v4M3 5h4M6 17v4m-2-2h4m5-16l2.286 6.857L21 12l-5.714 2.143L13 21l-2.286-6.857L5 12l5.714-2.143L13 3z" />
              </svg>
            </div>
            <h3 className="text-xl font-bold text-gray-800 ml-3">
              {formatClassName(plantType)}
            </h3>
          </div>
          
          <div className="space-y-2">
            {diseaseList.map((disease, index) => (
              <div
                key={index}
                className="bg-gray-50 rounded-lg p-3 hover:bg-primary-50 transition-colors border border-gray-200 hover:border-primary-300"
              >
                <div className="flex items-start">
                  <span className="flex-shrink-0 w-6 h-6 bg-primary-600 text-white rounded-full flex items-center justify-center text-xs font-bold">
                    {index + 1}
                  </span>
                  <span className="ml-3 text-sm text-gray-700">
                    {formatClassName(disease)}
                  </span>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
};

export default ClassesList;

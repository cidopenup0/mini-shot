import axios from 'axios';

// Base URL for the FastAPI backend
const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds timeout
  headers: {
    'Content-Type': 'application/json',
  },
});

// API functions
export const api = {
  // Health check
  checkHealth: async () => {
    try {
      const response = await apiClient.get('/health');
      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || error.message || 'Backend is offline' 
      };
    }
  },

  // Get all disease classes
  getClasses: async () => {
    try {
      const response = await apiClient.get('/classes');
      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || error.message || 'Failed to fetch disease classes' 
      };
    }
  },

  // Predict disease from image
  predictDisease: async (imageFile) => {
    try {
      const formData = new FormData();
      formData.append('file', imageFile);

      const response = await apiClient.post('/predict', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });

      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || error.message || 'Failed to predict disease' 
      };
    }
  },

  // Get prediction history
  getHistory: async () => {
    try {
      const response = await apiClient.get('/history');
      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || error.message || 'Failed to fetch prediction history' 
      };
    }
  },

  // Submit feedback for a prediction
  submitFeedback: async (feedbackData) => {
    try {
      const response = await apiClient.post('/feedback', feedbackData);
      return { success: true, data: response.data };
    } catch (error) {
      return { 
        success: false, 
        error: error.response?.data?.detail || error.message || 'Failed to submit feedback' 
      };
    }
  },
};

export default api;

// File validation utilities

export const ALLOWED_FILE_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
export const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB in bytes

export const validateFile = (file) => {
  const errors = [];

  // Check if file exists
  if (!file) {
    errors.push('Please select a file');
    return { isValid: false, errors };
  }

  // Check file type
  if (!ALLOWED_FILE_TYPES.includes(file.type)) {
    errors.push('Only JPEG and PNG images are allowed');
  }

  // Check file size
  if (file.size > MAX_FILE_SIZE) {
    errors.push('File size must be less than 10MB');
  }

  return {
    isValid: errors.length === 0,
    errors
  };
};

export const formatFileSize = (bytes) => {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];
  const i = Math.floor(Math.log(bytes) / Math.log(k));
  return Math.round(bytes / Math.pow(k, i) * 100) / 100 + ' ' + sizes[i];
};

export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(2)}%`;
};

export const formatClassName = (className) => {
  // Convert underscore-separated class names to readable format
  if (!className) return 'Unknown';
  return className
    .replace(/_/g, ' ')
    .split(' ')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export const formatTimestamp = (timestamp) => {
  const date = new Date(timestamp);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
};

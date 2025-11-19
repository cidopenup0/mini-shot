import React from 'react';
import { useNavigate } from 'react-router-dom';

const Login = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = React.useState({ email: '', password: '' });
  const [error, setError] = React.useState(null);
  const [loading, setLoading] = React.useState(false);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setFormData((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    setError(null);

    if (!formData.email || !formData.password) {
      setError('Email and password are required');
      return;
    }

    setLoading(true);

    setTimeout(() => {
      sessionStorage.setItem('plantdoc-user', JSON.stringify({ email: formData.email }));
      setLoading(false);
      navigate('/history');
    }, 800);
  };

  return (
    <div className="max-w-4xl mx-auto grid md:grid-cols-2 gap-10 items-center">
      <div className="space-y-6 animate-fadeIn">
        <p className="text-sm uppercase tracking-widest text-primary-600 font-semibold">
          Secure access
        </p>
        <h1 className="text-4xl font-extrabold text-gray-900">
          Login to view history & analytics
        </h1>
        <p className="text-lg text-gray-600">
          Your dashboard keeps every prediction organized. Sign in to revisit past diagnoses,
          download reports, and continue monitoring plant health across fields.
        </p>
        <div className="grid grid-cols-1 gap-4">
          <div className="p-4 rounded-xl border border-gray-200 bg-white shadow-sm">
            <p className="font-semibold text-gray-900">History</p>
            <p className="text-sm text-gray-600">Review previous uploads and outcomes.</p>
          </div>
          <div className="p-4 rounded-xl border border-gray-200 bg-white shadow-sm">
            <p className="font-semibold text-gray-900">Classes</p>
            <p className="text-sm text-gray-600">Explore supported diseases per crop.</p>
          </div>
          <div className="p-4 rounded-xl border border-gray-200 bg-white shadow-sm">
            <p className="font-semibold text-gray-900">Health tips</p>
            <p className="text-sm text-gray-600">Get remediation guidance instantly.</p>
          </div>
        </div>
      </div>

      <form onSubmit={handleSubmit} className="card space-y-5 animate-slideUp" noValidate>
        <div>
          <label htmlFor="email" className="block text-sm font-semibold text-gray-700 mb-2">
            Email
          </label>
          <input
            id="email"
            name="email"
            type="email"
            value={formData.email}
            onChange={handleChange}
            placeholder="farmer@example.com"
            className="input"
          />
        </div>

        <div>
          <label htmlFor="password" className="block text-sm font-semibold text-gray-700 mb-2">
            Password
          </label>
          <input
            id="password"
            name="password"
            type="password"
            value={formData.password}
            onChange={handleChange}
            placeholder="••••••••"
            className="input"
          />
        </div>

        <div className="flex items-center justify-between text-sm text-gray-500">
          <label className="flex items-center space-x-2">
            <input type="checkbox" className="rounded border-gray-300 text-primary-600 focus:ring-primary-500" />
            <span>Remember me</span>
          </label>
          <button type="button" className="text-primary-600 hover:text-primary-700">
            Forgot password?
          </button>
        </div>

        {error && (
          <div className="text-sm text-red-600 bg-red-50 border border-red-100 rounded-lg px-3 py-2">
            {error}
          </div>
        )}

        <button
          type="submit"
          className="btn-primary w-full py-3 font-semibold flex items-center justify-center space-x-2"
          disabled={loading}
        >
          {loading ? (
            <>
              <svg className="w-5 h-5 animate-spin" viewBox="0 0 24 24" fill="none">
                <circle
                  className="opacity-25"
                  cx="12"
                  cy="12"
                  r="10"
                  stroke="currentColor"
                  strokeWidth="4"
                ></circle>
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z"
                ></path>
              </svg>
              <span>Signing in...</span>
            </>
          ) : (
            <>
              <span>Login & view history</span>
              <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
              </svg>
            </>
          )}
        </button>

        <p className="text-xs text-gray-500 text-center">
          Demo login only: credentials are kept locally in this browser.
        </p>
      </form>
    </div>
  );
};

export default Login;


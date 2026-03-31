import React, { useState } from 'react';
import axios from 'axios';
import { User, Lock, UserPlus, LogIn } from 'lucide-react';

const Login = ({ onLogin }) => {
  const [isRegistering, setIsRegistering] = useState(false);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setMessage('');

    const endpoint = isRegistering ? 'register' : 'login';
    const url = `http://127.0.0.1:5000/${endpoint}`;

    try {
      const response = await axios.post(url, {
        username: username,
        password: password
      });

      if (isRegistering) {
        setMessage('Registration successful! Please log in.');
        setIsRegistering(false); // Switch back to login mode automatically
      } else {
        // It's a login, so pass the user up to the main App
        onLogin(response.data.username);
      }
    } catch (err) {
      if (err.response) {
        const backendMessage =
          err.response.data?.error ||
          err.response.data?.detail ||
          err.response.statusText ||
          'An error occurred';
        setError(backendMessage);
      } else {
        setError('Server not connecting. Is backend running?');
      }
    }
  };

  return (
    <div className="min-h-screen bg-blue-50 flex items-center justify-center">
      <div className="bg-white p-8 rounded-xl shadow-lg w-96">
        <div className="text-center mb-8">
          <h1 className="text-2xl font-bold text-blue-900">
            {isRegistering ? 'Create Account' : 'Serenity Login'}
          </h1>
          <p className="text-gray-500 text-sm mt-2">
            {isRegistering ? 'Sign up for a new account' : 'Welcome back, please sign in'}
          </p>
        </div>

        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4 text-sm">
            {error}
          </div>
        )}

        {message && (
          <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded mb-4 text-sm">
            {message}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <User className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          <div className="relative">
            <Lock className="absolute left-3 top-3 h-5 w-5 text-gray-400" />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 transition flex items-center justify-center gap-2"
          >
            {isRegistering ? <UserPlus size={18} /> : <LogIn size={18} />}
            {isRegistering ? 'Sign Up' : 'Sign In'}
          </button>
        </form>

        <div className="mt-6 text-center text-sm">
          <button
            onClick={() => {
              setIsRegistering(!isRegistering);
              setError('');
              setMessage('');
            }}
            className="text-blue-600 hover:underline font-medium"
          >
            {isRegistering
              ? 'Already have an account? Login'
              : "Don't have an account? Register"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default Login;
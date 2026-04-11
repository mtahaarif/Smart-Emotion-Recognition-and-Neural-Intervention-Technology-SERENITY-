import React, { useState } from 'react';
import axios from 'axios';
import { Activity, Lock, LogIn, User, UserPlus } from 'lucide-react';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

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
    const url = `${API_BASE_URL}/${endpoint}`;

    try {
      const response = await axios.post(url, {
        username: username.trim(),
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
    <div className="min-h-screen bg-slate-950 text-slate-100 flex items-center justify-center px-4">
      <div className="w-full max-w-md rounded-2xl border border-cyan-900/60 bg-slate-900/70 p-8 shadow-2xl shadow-cyan-950/20">
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14 rounded-xl bg-slate-950 border border-cyan-900/60 text-cyan-300 mb-4">
            <Activity size={24} />
          </div>
          <h1 className="text-2xl font-bold text-cyan-300">
            {isRegistering ? 'Create Account' : 'Serenity Login'}
          </h1>
          <p className="text-slate-400 text-sm mt-2">
            {isRegistering ? 'Sign up for a new account' : 'Welcome back, please sign in'}
          </p>
        </div>

        {error && (
          <div className="bg-rose-950/40 border border-rose-600/60 text-rose-200 px-4 py-3 rounded-lg mb-4 text-sm">
            {error}
          </div>
        )}

        {message && (
          <div className="bg-emerald-950/40 border border-emerald-600/60 text-emerald-200 px-4 py-3 rounded-lg mb-4 text-sm">
            {message}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div className="relative">
            <User className="absolute left-3 top-3 h-5 w-5 text-slate-500" />
            <input
              type="text"
              placeholder="Username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-cyan-900/60 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-600 text-slate-100"
              required
            />
          </div>

          <div className="relative">
            <Lock className="absolute left-3 top-3 h-5 w-5 text-slate-500" />
            <input
              type="password"
              placeholder="Password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full pl-10 pr-4 py-2 bg-slate-950 border border-cyan-900/60 rounded-lg focus:outline-none focus:ring-2 focus:ring-cyan-600 text-slate-100"
              required
            />
          </div>

          <button
            type="submit"
            className="w-full bg-cyan-700 text-white py-2 rounded-lg hover:bg-cyan-600 transition flex items-center justify-center gap-2 font-semibold"
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
            className="text-cyan-300 hover:underline font-medium"
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
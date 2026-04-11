import React from 'react';
import { LogOut, Activity } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Dashboard = ({ user, onLogout }) => {
  const navigate = useNavigate();

  const handleStartSession = () => {
    navigate('/emotion/live');
  };

  return (
    <div className="min-h-screen bg-gray-50 flex flex-col">
      {/* 1. Navbar */}
      <nav className="bg-white border-b px-8 py-4 flex justify-between items-center sticky top-0 z-50 shadow-sm">
        <h1 className="text-2xl font-bold text-blue-600 flex items-center gap-2">
          <Activity className="text-blue-600" /> Serenity AI
        </h1>
        <div className="flex items-center gap-4">
          <span className="text-gray-500">Patient: <b>{user}</b></span>
          <button onClick={onLogout} className="text-red-500 font-medium hover:text-red-700 flex items-center gap-1">
            <LogOut size={18} /> Logout
          </button>
        </div>
      </nav>

      {/* 2. Main Content (Dashboard Only) */}
      <div className="flex-1 p-6 overflow-y-auto">
        <div className="flex flex-col items-center justify-center h-[80vh] text-center space-y-8 animate-fade-in">
          <div className="bg-blue-100 p-6 rounded-full text-blue-600 mb-4">
            <Activity size={64} />
          </div>
          <h2 className="text-4xl font-bold text-gray-800">Hello, {user}.</h2>
          <p className="text-xl text-gray-500 max-w-lg">
            I am your AI Psychologist. I will analyze your facial expressions and voice tone in real-time.
          </p>
          <button
            onClick={handleStartSession}
            className="px-12 py-5 bg-blue-600 text-white text-xl font-bold rounded-full shadow-2xl hover:bg-blue-700 transition-transform hover:scale-105 active:scale-95"
          >
            Start Live Therapy Session
          </button>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
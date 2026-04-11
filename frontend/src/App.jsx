import React, { Suspense, lazy, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';

const Login = lazy(() => import('./components/Login'));
const Dashboard = lazy(() => import('./components/Dashboard'));
const UnifiedEmotionPage = lazy(() => import('./pages/UnifiedEmotionPage'));

function App() {
  const [user, setUser] = useState(() => localStorage.getItem('serenity_user'));

  const handleLogin = (username) => {
    localStorage.setItem('serenity_user', username);
    setUser(username);
  };

  const handleLogout = () => {
    localStorage.removeItem('serenity_user');
    setUser(null);
  };

  const isAuthenticated = Boolean(user);

  const requireAuth = (element) => {
    if (!isAuthenticated) {
      return <Navigate to="/login" replace />;
    }
    return element;
  };

  return (
    <BrowserRouter>
      <Suspense fallback={<div className="min-h-screen bg-slate-950 text-slate-200 flex items-center justify-center">Loading...</div>}>
        <Routes>
          <Route
            path="/login"
            element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login onLogin={handleLogin} />}
          />
          <Route
            path="/dashboard"
            element={requireAuth(<Dashboard user={user} onLogout={handleLogout} />)}
          />
          <Route
            path="/emotion/live"
            element={requireAuth(<UnifiedEmotionPage user={user} onLogout={handleLogout} />)}
          />
          <Route
            path="*"
            element={<Navigate to={isAuthenticated ? '/dashboard' : '/login'} replace />}
          />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}

export default App;
import React, { useMemo, useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Login from './components/Login';
import Dashboard from './components/Dashboard';
import UnifiedEmotionPage from './pages/UnifiedEmotionPage';

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

  const isAuthenticated = useMemo(() => Boolean(user), [user]);

  const requireAuth = (element) => {
    if (!isAuthenticated) {
      return <Navigate to="/login" replace />;
    }
    return element;
  };

  return (
    <BrowserRouter>
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
    </BrowserRouter>
  );
}

export default App;
import React, { useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import Login from './components/Login';
import MBCHubPage from './pages/MBCHubPage';
import AdminPage from './pages/AdminPage';
import HardwareDiagnosticsPage from './pages/HardwareDiagnosticsPage';
import SafetyPlanPage from './pages/SafetyPlanPage';
import QuestionnairesPage from './pages/QuestionnairesPage';
import UnifiedEmotionPage from './pages/UnifiedEmotionPage';
import { ClinicalProvider } from './context/ClinicalContext';

const AuthGuard = ({ isAuthenticated, children }) => {
  if (!isAuthenticated) {
    return <Navigate to="/login" replace />;
  }
  return children;
};

const ProtectedRoute = ({ isAuthenticated, children }) => (
  <AuthGuard isAuthenticated={isAuthenticated}>{children}</AuthGuard>
);

function App() {
  const [user, setUser] = useState(() => localStorage.getItem('serenity_user'));
  const isAuthenticated = Boolean(user);

  const handleLogin = (username) => {
    localStorage.setItem('serenity_user', username);
    setUser(username);
  };

  const handleLogout = () => {
    localStorage.removeItem('serenity_user');
    setUser(null);
  };

  return (
    <BrowserRouter>
      <ClinicalProvider>
        {/* CrisisRedirector completely removed. No more forced navigation locks. */}
        <Routes>
          <Route
            path="/login"
            element={isAuthenticated ? <Navigate to="/dashboard" replace /> : <Login onLogin={handleLogin} />}
          />

          <Route
            path="/"
            element={<Navigate to={isAuthenticated ? '/dashboard' : '/login'} replace />}
          />

          <Route
            path="/dashboard"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <Dashboard user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/questionnaires"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <QuestionnairesPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/emotion/live"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <UnifiedEmotionPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/session"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <UnifiedEmotionPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/mbc-hub"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <MBCHubPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/safety"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <SafetyPlanPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/admin"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <AdminPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route
            path="/diagnostics"
            element={(
              <ProtectedRoute isAuthenticated={isAuthenticated}>
                <HardwareDiagnosticsPage user={user} onLogout={handleLogout} />
              </ProtectedRoute>
            )}
          />

          <Route path="*" element={<Navigate to={isAuthenticated ? '/dashboard' : '/login'} replace />} />
        </Routes>
      </ClinicalProvider>
    </BrowserRouter>
  );
}

export default App;
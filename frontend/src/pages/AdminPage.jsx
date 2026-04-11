import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  Database,
  Loader2,
  LogOut,
  MessageSquare,
  RefreshCw,
  ShieldAlert,
  UserRound,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const formatDate = (value) => {
  if (!value) {
    return 'Unknown time';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }
  return parsed.toLocaleString();
};

const MetricIcon = ({ id }) => {
  if (id === 'users') return <UserRound size={16} />;
  if (id === 'turns') return <MessageSquare size={16} />;
  if (id === 'emotion_events') return <Brain size={16} />;
  if (id === 'flagged_users') return <ShieldAlert size={16} />;
  return <Database size={16} />;
};

const AdminPage = ({ user, onLogout }) => {
  const [overview, setOverview] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [limit, setLimit] = useState(300);
  const [chatVisibleCount, setChatVisibleCount] = useState(80);
  const [sessionVisibleCount, setSessionVisibleCount] = useState(40);
  const [questionnaireVisibleCount, setQuestionnaireVisibleCount] = useState(80);

  const metrics = useMemo(() => (Array.isArray(overview?.metrics) ? overview.metrics : []), [overview]);
  const chats = useMemo(() => (Array.isArray(overview?.chats) ? overview.chats : []), [overview]);
  const sessions = useMemo(() => (Array.isArray(overview?.sessions) ? overview.sessions : []), [overview]);
  const questionnaireResults = useMemo(
    () => (Array.isArray(overview?.questionnaire_results) ? overview.questionnaire_results : []),
    [overview]
  );
  const topEmotions = useMemo(() => (Array.isArray(overview?.top_emotions) ? overview.top_emotions : []), [overview]);
  const flaggedUsers = useMemo(() => (Array.isArray(overview?.flagged_users) ? overview.flagged_users : []), [overview]);
  const displayedChats = useMemo(() => chats.slice(0, chatVisibleCount), [chats, chatVisibleCount]);
  const displayedSessions = useMemo(() => sessions.slice(0, sessionVisibleCount), [sessions, sessionVisibleCount]);
  const displayedQuestionnaires = useMemo(
    () => questionnaireResults.slice(0, questionnaireVisibleCount),
    [questionnaireResults, questionnaireVisibleCount]
  );

  const loadOverview = async (targetLimit = limit) => {
    setErrorMessage('');
    setIsLoading(true);
    setChatVisibleCount(80);
    setSessionVisibleCount(40);
    setQuestionnaireVisibleCount(80);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/admin/overview`, {
        params: { limit: targetLimit },
      });
      setOverview(response.data || null);
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to load admin overview.';
      setErrorMessage(backendMessage);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadOverview(limit);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <nav className="border-b border-cyan-900/50 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/95 backdrop-blur">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="inline-flex items-center gap-2 text-slate-300 hover:text-white font-medium">
            <ArrowLeft size={18} /> Back
          </Link>
          <h1 className="text-2xl font-bold text-cyan-300 flex items-center gap-2">
            <Activity className="text-cyan-300" /> Serenity Admin Observatory
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-slate-400">Signed in: {user}</span>
          <button onClick={onLogout} className="text-rose-400 font-medium hover:text-rose-300 flex items-center gap-1">
            <LogOut size={18} /> Logout
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto space-y-4">
          <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-wider text-cyan-300">Mental Health Summary</p>
                <h2 className="text-xl font-semibold text-white">System-Wide Local Insights</h2>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={limit}
                  min={0}
                  max={5000}
                  onChange={(event) => setLimit(Number(event.target.value || 300))}
                  className="w-28 rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
                />
                <button
                  type="button"
                  onClick={() => loadOverview(Math.max(0, Math.min(5000, Number(limit) || 0)))}
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-700 hover:bg-cyan-600 px-4 py-2 font-semibold"
                >
                  <RefreshCw size={15} /> Refresh
                </button>
              </div>
            </div>
            <p className="text-xs text-slate-500 mt-2">Default limit is 300 for fast rendering. Set limit to 0 only when you need the full dataset.</p>

            {isLoading ? (
              <div className="mt-4 rounded-lg border border-cyan-800/60 bg-slate-900 p-3 text-slate-300 inline-flex items-center gap-2">
                <Loader2 className="animate-spin" size={16} /> Building admin summary from local database...
              </div>
            ) : (
              <>
                <p className="mt-4 text-slate-200 leading-relaxed">{overview?.summary || 'No summary available yet.'}</p>
                <p className="text-xs text-slate-500 mt-2">Generated: {formatDate(overview?.generated_at)}</p>
              </>
            )}

            {errorMessage && (
              <div className="mt-4 rounded-lg border border-rose-600/60 bg-rose-950/30 p-3 text-rose-200 text-sm inline-flex items-center gap-2">
                <AlertTriangle size={15} /> {errorMessage}
              </div>
            )}
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            <div className="lg:col-span-2 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
              <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Chats and Conversations</h3>
              <div className="space-y-3 max-h-[420px] overflow-auto pr-1">
                {chats.length === 0 && <p className="text-slate-500 text-sm">No conversation turns available.</p>}
                {displayedChats.map((chat) => (
                  <div key={chat.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 space-y-1">
                    <p className="text-xs text-slate-400">
                      {chat.username || 'unknown'} • {formatDate(chat.timestamp)} • emotion: {String(chat.dominant_emotion || 'neutral').toLowerCase()}
                    </p>
                    <p className="text-sm text-cyan-300">User: {chat.user_text}</p>
                    <p className="text-sm text-white">Serenity: {chat.assistant_text}</p>
                  </div>
                ))}
                {displayedChats.length < chats.length && (
                  <button
                    type="button"
                    onClick={() => setChatVisibleCount((prev) => prev + 80)}
                    className="w-full rounded-lg border border-cyan-800/60 bg-slate-900 px-3 py-2 text-sm text-cyan-300 hover:border-cyan-600"
                  >
                    Load 80 More Chats ({displayedChats.length}/{chats.length})
                  </button>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
              <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Emotion Distribution</h3>
              <div className="space-y-2 max-h-[420px] overflow-auto pr-1">
                {topEmotions.length === 0 && <p className="text-slate-500 text-sm">No emotion events available.</p>}
                {topEmotions.map((item) => (
                  <div key={item.emotion} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 flex items-center justify-between">
                    <span className="text-slate-200 capitalize">{item.emotion}</span>
                    <span className="text-cyan-300 font-semibold">{item.count}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
              <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Sessions and Emotion Trails</h3>
              <div className="space-y-3 max-h-[360px] overflow-auto pr-1">
                {sessions.length === 0 && <p className="text-slate-500 text-sm">No legacy sessions available.</p>}
                {displayedSessions.map((session) => (
                  <div key={session.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                    <p className="text-xs text-slate-400">
                      Session #{session.id} • {session.username || 'unknown'} • {formatDate(session.timestamp)}
                    </p>
                    <p className="text-sm text-slate-200 mt-1">{session.conversation || 'No conversation text stored for this session.'}</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {(session.emotions || []).slice(0, 8).map((emotionItem) => (
                        <span
                          key={`${session.id}-${emotionItem.id}`}
                          className="rounded-full border border-cyan-900/40 bg-slate-900 px-2 py-1 text-xs text-cyan-300"
                        >
                          {String(emotionItem.emotion || 'neutral').toLowerCase()} ({Number(emotionItem.confidence || 0).toFixed(2)})
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
                {displayedSessions.length < sessions.length && (
                  <button
                    type="button"
                    onClick={() => setSessionVisibleCount((prev) => prev + 40)}
                    className="w-full rounded-lg border border-cyan-800/60 bg-slate-900 px-3 py-2 text-sm text-cyan-300 hover:border-cyan-600"
                  >
                    Load 40 More Sessions ({displayedSessions.length}/{sessions.length})
                  </button>
                )}
              </div>
            </div>

            <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
              <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Questionnaires and Risk Signals</h3>
              <div className="space-y-3 max-h-[360px] overflow-auto pr-1">
                {questionnaireResults.length === 0 && <p className="text-slate-500 text-sm">No questionnaire entries found.</p>}
                {displayedQuestionnaires.map((row) => (
                  <div key={row.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                    <p className="text-xs text-slate-400">
                      {row.username || 'unknown'} • {row.questionnaire_type} • {formatDate(row.created_at)}
                    </p>
                    <p className="text-sm text-cyan-300">Score: {row.total_score}</p>
                    <p className="text-sm text-emerald-300">Severity: {row.severity}</p>
                  </div>
                ))}
                {displayedQuestionnaires.length < questionnaireResults.length && (
                  <button
                    type="button"
                    onClick={() => setQuestionnaireVisibleCount((prev) => prev + 80)}
                    className="w-full rounded-lg border border-cyan-800/60 bg-slate-900 px-3 py-2 text-sm text-cyan-300 hover:border-cyan-600"
                  >
                    Load 80 More Results ({displayedQuestionnaires.length}/{questionnaireResults.length})
                  </button>
                )}
              </div>

              {flaggedUsers.length > 0 && (
                <div className="mt-4 rounded-lg border border-amber-600/60 bg-amber-950/30 p-3">
                  <p className="text-amber-300 font-semibold mb-2 inline-flex items-center gap-2">
                    <ShieldAlert size={15} /> Elevated Screening Indicators
                  </p>
                  <div className="space-y-2 text-sm text-amber-100">
                    {flaggedUsers.map((entry) => (
                      <p key={entry.username}>
                        {entry.username}: {Object.entries(entry.flags || {})
                          .filter((pair) => Boolean(pair[1]))
                          .map((pair) => pair[0])
                          .join(', ') || 'elevated'}
                      </p>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </section>

          <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Metrics Dashboard</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-3">
              {metrics.map((metric) => (
                <article key={metric.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                  <p className="text-xs text-slate-400 inline-flex items-center gap-2">
                    <MetricIcon id={metric.id} /> {metric.label}
                  </p>
                  <p className="text-2xl font-bold text-cyan-300 mt-2">{metric.value}</p>
                  <p className="text-xs text-slate-500 mt-1">{metric.description}</p>
                </article>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default AdminPage;

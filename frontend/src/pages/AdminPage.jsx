import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  Clock3,
  Database,
  FileText,
  Loader2,
  LogOut,
  MessageSquare,
  Shield,
  RefreshCw,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const formatDate = (value) => {
  if (!value) return 'Unknown time';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
};

const riskTone = (riskLevel) => {
  const key = String(riskLevel || '').toLowerCase();
  if (key === 'elevated') return 'border-rose-500/60 bg-rose-950/30 text-rose-200';
  if (key === 'monitor') return 'border-amber-500/60 bg-amber-950/30 text-amber-200';
  return 'border-emerald-500/60 bg-emerald-950/30 text-emerald-200';
};

const MetricIcon = ({ id }) => {
  if (id === 'turns') return <MessageSquare size={15} />;
  if (id === 'sessions') return <Clock3 size={15} />;
  if (id === 'emotion_events') return <Brain size={15} />;
  if (id === 'risk_score') return <Shield size={15} />;
  if (id === 'distress_signals') return <AlertTriangle size={15} />;
  return <Database size={15} />;
};

const AdminPage = ({ user, onLogout }) => {
  const [overview, setOverview] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [limit, setLimit] = useState(300);
  const [chatVisibleCount, setChatVisibleCount] = useState(30);

  const metrics = useMemo(() => (Array.isArray(overview?.metrics) ? overview.metrics : []), [overview]);
  const chats = useMemo(() => (Array.isArray(overview?.chats) ? overview.chats : []), [overview]);
  const sessions = useMemo(() => (Array.isArray(overview?.sessions) ? overview.sessions : []), [overview]);
  const questionnaireResults = useMemo(
    () => (Array.isArray(overview?.questionnaire_results) ? overview.questionnaire_results : []),
    [overview]
  );
  const topEmotions = useMemo(() => (Array.isArray(overview?.top_emotions) ? overview.top_emotions : []), [overview]);
  const profile = useMemo(() => (overview?.profile && typeof overview.profile === 'object' ? overview.profile : null), [overview]);
  const clinical = useMemo(
    () => (overview?.clinical_parameters && typeof overview.clinical_parameters === 'object' ? overview.clinical_parameters : {}),
    [overview]
  );
  const displayedChats = useMemo(() => chats.slice(0, chatVisibleCount), [chats, chatVisibleCount]);

  const summaryLines = useMemo(() => {
    const text = String(overview?.summary || '').trim();
    if (!text) return [];
    const rows = text.includes('\n') ? text.split('\n') : text.split(/(?<=\.)\s+/);
    return rows.map((line) => line.replace(/^[-•]\s*/, '').trim()).filter(Boolean).slice(0, 6);
  }, [overview]);

  const scoreRows = useMemo(() => {
    if (!profile?.latest_scores) return [];
    return Object.entries(profile.latest_scores).map(([type, score]) => ({
      type,
      score,
      severity: profile?.latest_severity?.[type] || 'unknown',
      trend: profile?.screening_trends?.[type] || 'insufficient_data',
    }));
  }, [profile]);

  const loadOverview = async (targetLimit = limit) => {
    const finalLimit = clamp(Number(targetLimit) || 300, 20, 3000);
    setErrorMessage('');
    setIsLoading(true);
    setChatVisibleCount(30);
    try {
      if (!user) {
        throw new Error('Missing active user session.');
      }
      const response = await axios.get(`${API_BASE_URL}/api/admin/overview`, {
        params: { limit: finalLimit, username: user },
      });
      setOverview(response.data || null);
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to load admin overview.';
      setErrorMessage(String(backendMessage));
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadOverview(limit);
  }, []);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <nav className="border-b border-cyan-900/50 px-6 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/95 backdrop-blur">
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
          <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-wider text-cyan-300">Clinical Report</p>
                <h2 className="text-xl font-semibold text-white">SERENITY Professional Evaluation: {user}</h2>
                <p className="text-xs text-slate-500 mt-1">
                  Generated: {formatDate(overview?.generated_at)}
                  {overview?.summary_source ? ` • source: ${overview.summary_source}` : ''}
                </p>
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="number"
                  value={limit}
                  min={20}
                  max={3000}
                  onChange={(event) => setLimit(clamp(Number(event.target.value) || 300, 20, 3000))}
                  className="w-28 rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
                />
                <button
                  type="button"
                  onClick={() => loadOverview(limit)}
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-700 hover:bg-cyan-600 px-4 py-2 font-semibold"
                >
                  <RefreshCw size={15} /> Refresh
                </button>
              </div>
            </div>

            {isLoading ? (
              <div className="mt-4 rounded-lg border border-cyan-800/60 bg-slate-900 p-3 text-slate-300 inline-flex items-center gap-2">
                <Loader2 className="animate-spin" size={16} /> Building profile report...
              </div>
            ) : (
              <div className="mt-4 space-y-2">
                {summaryLines.length === 0 ? (
                  <p className="text-slate-300">No summary available yet.</p>
                ) : (
                  summaryLines.map((line, idx) => (
                    <p key={`summary-${idx}`} className="text-slate-200 leading-relaxed">
                      • {line}
                    </p>
                  ))
                )}
              </div>
            )}

            {errorMessage && (
              <div className="mt-4 rounded-lg border border-rose-600/60 bg-rose-950/30 p-3 text-rose-200 text-sm inline-flex items-center gap-2">
                <AlertTriangle size={15} /> {errorMessage}
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Metrics Dashboard</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-6 gap-3">
              {metrics.map((metric) => (
                <article key={metric.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                  <p className="text-xs text-slate-400 inline-flex items-center gap-2">
                    <MetricIcon id={metric.id} /> {metric.label || metric.id}
                  </p>
                  <p className="text-2xl font-bold text-cyan-300 mt-2">{metric.value}</p>
                  <p className="text-xs text-slate-500 mt-1">{metric.description || ''}</p>
                </article>
              ))}
            </div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            <div className="xl:col-span-2 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4 space-y-4">
              <div className="flex items-center justify-between">
                <h3 className="text-xs uppercase tracking-wider text-cyan-300">Risk Formulation</h3>
                <span className={`px-2 py-1 rounded-full border text-xs font-medium uppercase ${riskTone(profile?.risk_level)}`}>
                  {profile?.risk_level || 'stable'}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-sm">
                <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                  <p className="text-xs text-slate-400 uppercase">Core Indicators</p>
                  <p className="text-slate-200 mt-2">Risk score: <span className="text-cyan-300 font-semibold">{profile?.risk_score ?? 0}</span></p>
                  <p className="text-slate-200">Distress signals: <span className="text-cyan-300 font-semibold">{clinical?.distress_signal_count ?? 0}</span></p>
                  <p className="text-slate-200">Negative affect ratio: <span className="text-cyan-300 font-semibold">{clinical?.negative_emotion_ratio ?? 0}</span></p>
                  <p className="text-slate-500 text-xs mt-2">Last seen: {formatDate(profile?.last_seen)}</p>
                </article>

                <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                  <p className="text-xs text-slate-400 uppercase">Protective and Risk Factors</p>
                  <p className="text-slate-200 mt-2">Engagement level: <span className="text-cyan-300 font-semibold">{profile?.engagement_level || 'low'}</span></p>
                  <p className="text-slate-200">Engagement score: <span className="text-cyan-300 font-semibold">{profile?.engagement_score ?? 0}</span></p>
                  <p className="text-slate-200">Active clinical flags:</p>
                  <p className="text-xs text-amber-300 mt-1">
                    {Array.isArray(profile?.active_flags) && profile.active_flags.length > 0
                      ? profile.active_flags.join(', ')
                      : 'none'}
                  </p>
                </article>
              </div>

              {profile?.latest_assistant_note && (
                <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                  <p className="text-xs text-slate-400 uppercase">Recent Therapeutic Note</p>
                  <p className="text-sm text-slate-200 mt-2">{profile.latest_assistant_note}</p>
                </article>
              )}
            </div>

            <div className="space-y-4">
              <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Measurement-Based Evaluation</h3>
                <div className="space-y-2 max-h-[260px] overflow-auto pr-1">
                  {scoreRows.length === 0 && <p className="text-slate-500 text-sm">No screening data available.</p>}
                  {scoreRows.map((row) => (
                    <article key={row.type} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                      <p className="text-sm text-cyan-300 font-semibold">{row.type}</p>
                      <p className="text-xs text-slate-300 mt-1">Score: {row.score}</p>
                      <p className="text-xs text-slate-300">Severity: {row.severity}</p>
                      <p className="text-xs text-slate-300">Trend: {row.trend}</p>
                    </article>
                  ))}
                </div>
              </div>

              <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Emotion Distribution</h3>
                <div className="space-y-2 max-h-[220px] overflow-auto pr-1">
                  {topEmotions.length === 0 && <p className="text-slate-500 text-sm">No emotion data available.</p>}
                  {topEmotions.slice(0, 6).map((item) => (
                    <div key={item.emotion} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 flex items-center justify-between">
                      <span className="text-slate-200 capitalize">{item.emotion}</span>
                      <span className="text-cyan-300 font-semibold">{item.count}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Recent Conversation Notes</h3>
            <div className="space-y-3 max-h-[400px] overflow-auto pr-1">
              {chats.length === 0 && <p className="text-slate-500 text-sm">No conversation turns available.</p>}
              {displayedChats.map((chat) => (
                <article key={chat.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 space-y-1">
                  <p className="text-xs text-slate-400">
                    {chat.username || 'unknown'} • {formatDate(chat.timestamp)} • emotion: {String(chat.dominant_emotion || 'neutral').toLowerCase()}
                  </p>
                  <p className="text-sm text-cyan-300">User: {chat.user_text}</p>
                  <p className="text-sm text-white">Serenity: {chat.assistant_text}</p>
                </article>
              ))}

              {displayedChats.length < chats.length && (
                <button
                  type="button"
                  onClick={() => setChatVisibleCount((prev) => prev + 30)}
                  className="w-full rounded-lg border border-cyan-800/60 bg-slate-900 px-3 py-2 text-sm text-cyan-300 hover:border-cyan-600"
                >
                  Load 30 More Notes ({displayedChats.length}/{chats.length})
                </button>
              )}
            </div>
          </section>

          <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
            <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
              <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Recent Questionnaire Entries</h3>
              <div className="space-y-2 max-h-[260px] overflow-auto pr-1">
                {questionnaireResults.length === 0 && <p className="text-slate-500 text-sm">No questionnaire entries found.</p>}
                {questionnaireResults.slice(0, 12).map((row) => (
                  <article key={row.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                    <p className="text-xs text-slate-400">
                      {row.username || 'unknown'} • {row.questionnaire_type} • {formatDate(row.created_at)}
                    </p>
                    <p className="text-sm text-cyan-300">Score: {row.total_score}</p>
                    <p className="text-xs text-slate-300">Severity: {row.severity}</p>
                  </article>
                ))}
              </div>
            </div>

            <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
              <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Session Timeline</h3>
              <div className="space-y-2 max-h-[260px] overflow-auto pr-1">
                {sessions.length === 0 && <p className="text-slate-500 text-sm">No legacy sessions available.</p>}
                {sessions.slice(0, 8).map((session) => (
                  <article key={session.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                    <p className="text-xs text-slate-400">
                      Session #{session.id} • {session.username || 'unknown'} • {formatDate(session.timestamp)}
                    </p>
                    <p className="text-xs text-slate-300 mt-1 line-clamp-2">
                      {session.conversation || 'No conversation text stored for this session.'}
                    </p>
                  </article>
                ))}
              </div>
            </div>
          </section>

          <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
              <FileText size={14} /> Professional Technique Signals
            </h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
              <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                <p className="text-xs text-slate-400 uppercase">Measurement-Based Care</p>
                <p className="text-slate-200 mt-2">Uses PHQ-9, GAD-7, and PCL-5 latest scores with trend interpretation.</p>
              </article>
              <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                <p className="text-xs text-slate-400 uppercase">Structured Risk Formulation</p>
                <p className="text-slate-200 mt-2">Combines screening flags, distress language signals, and affect ratio into risk score.</p>
              </article>
              <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                <p className="text-xs text-slate-400 uppercase">Trauma-Informed Follow-up</p>
                <p className="text-slate-200 mt-2">Summaries prioritize safety planning, non-judgmental check-ins, and cadence targets.</p>
              </article>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default AdminPage;

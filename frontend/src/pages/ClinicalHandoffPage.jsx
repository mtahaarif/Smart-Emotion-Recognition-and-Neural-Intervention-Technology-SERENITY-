import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  AlertTriangle,
  ArrowLeft,
  ClipboardList,
  FileText,
  Loader2,
  LogOut,
  Shield,
  TrendingUp,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const bandTone = (value) => {
  const key = String(value || '').toLowerCase();
  if (key === 'critical') return 'border-rose-600/70 bg-rose-950/30 text-rose-200';
  if (key === 'high') return 'border-orange-500/70 bg-orange-950/30 text-orange-200';
  if (key === 'moderate') return 'border-amber-500/70 bg-amber-950/30 text-amber-200';
  return 'border-emerald-500/70 bg-emerald-950/30 text-emerald-200';
};

const formatDate = (value) => {
  if (!value) return 'Unknown time';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString();
};

const ClinicalHandoffPage = ({ user, onLogout }) => {
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [copyMessage, setCopyMessage] = useState('');
  const [payload, setPayload] = useState(null);

  const forecast = useMemo(() => {
    if (payload?.forecast && typeof payload.forecast === 'object') {
      return payload.forecast;
    }
    return {};
  }, [payload]);

  const handoff = useMemo(() => {
    if (payload?.handoff && typeof payload.handoff === 'object') {
      return payload.handoff;
    }
    return {};
  }, [payload]);

  const triage = useMemo(() => {
    if (handoff?.triage && typeof handoff.triage === 'object') {
      return handoff.triage;
    }
    return {};
  }, [handoff]);

  const screeningRows = useMemo(() => {
    return Array.isArray(handoff?.screening_snapshot) ? handoff.screening_snapshot : [];
  }, [handoff]);

  const cbtSnapshot = useMemo(() => {
    if (handoff?.cbt_snapshot && typeof handoff.cbt_snapshot === 'object') {
      return handoff.cbt_snapshot;
    }
    return {};
  }, [handoff]);

  const adherence = useMemo(() => {
    if (handoff?.adherence_snapshot && typeof handoff.adherence_snapshot === 'object') {
      return handoff.adherence_snapshot;
    }
    return {};
  }, [handoff]);

  const loadHandoff = async () => {
    setLoading(true);
    setErrorMessage('');
    setCopyMessage('');
    try {
      const response = await axios.get(`${API_BASE_URL}/api/clinical/handoff`, {
        params: { username: user },
      });
      setPayload(response.data || null);
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to load clinical handoff report.';
      setErrorMessage(String(backendMessage));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadHandoff();
  }, [user]);

  const copyMarkdown = async () => {
    const markdown = String(handoff?.report_markdown || '').trim();
    if (!markdown) {
      setCopyMessage('No handoff markdown available to copy yet.');
      return;
    }

    try {
      await navigator.clipboard.writeText(markdown);
      setCopyMessage('Handoff markdown copied to clipboard.');
    } catch (_error) {
      setCopyMessage('Clipboard copy failed in this browser context.');
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <nav className="border-b border-cyan-900/50 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/95 backdrop-blur">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="inline-flex items-center gap-2 text-slate-300 hover:text-white font-medium">
            <ArrowLeft size={18} /> Back
          </Link>
          <h1 className="text-2xl font-bold text-cyan-300 flex items-center gap-2">
            <ClipboardList className="text-cyan-300" /> Clinical Handoff
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-slate-400">Patient: {user}</span>
          <button onClick={onLogout} className="text-rose-400 font-medium hover:text-rose-300 flex items-center gap-1">
            <LogOut size={18} /> Logout
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto space-y-4">
          {loading && (
            <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-4 text-slate-300 inline-flex items-center gap-2">
              <Loader2 className="animate-spin" size={16} /> Generating handoff package...
            </div>
          )}

          {!loading && errorMessage && (
            <div className="rounded-lg border border-rose-600/60 bg-rose-950/30 p-4 text-rose-200 inline-flex items-center gap-2">
              <AlertTriangle size={16} /> {errorMessage}
            </div>
          )}

          {!loading && !errorMessage && (
            <>
              <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-cyan-300">Phase 3 Forecast + Handoff</p>
                    <h2 className="text-2xl font-bold text-white mt-1">Relapse Prevention and Clinician Transition</h2>
                    <p className="text-slate-300 mt-2 max-w-3xl">
                      This report combines risk formulation, CBT trajectory, and routine adherence into a handoff-ready snapshot.
                    </p>
                    <p className="text-xs text-slate-500 mt-2">
                      Generated: {formatDate(payload?.generated_at)} | Confidence: {forecast?.confidence || 'low'}
                    </p>
                  </div>
                  <div className="flex flex-col items-end gap-2">
                    <span className={`px-3 py-1 rounded-full border text-xs font-medium uppercase ${bandTone(forecast?.band)}`}>
                      Relapse {forecast?.relapse_probability_pct ?? 0}% ({forecast?.band || 'low'})
                    </span>
                    <button
                      type="button"
                      onClick={copyMarkdown}
                      className="inline-flex items-center gap-2 rounded-lg border border-cyan-600/60 bg-cyan-500/10 px-3 py-2 text-xs text-cyan-200 hover:bg-cyan-500/20"
                    >
                      <FileText size={13} /> Copy Handoff Markdown
                    </button>
                    {copyMessage && <p className="text-xs text-slate-300">{copyMessage}</p>}
                  </div>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
                    <TrendingUp size={14} /> Relapse Risk Drivers
                  </h3>
                  <div className="space-y-2 text-sm">
                    {(forecast?.contributors || []).map((item, idx) => (
                      <article key={`driver-${idx}`} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 flex items-center justify-between">
                        <p className="text-slate-200">{item.driver}</p>
                        <span className={`font-semibold ${Number(item.impact) >= 0 ? 'text-rose-300' : 'text-emerald-300'}`}>
                          {Number(item.impact) >= 0 ? '+' : ''}
                          {Number(item.impact).toFixed(1)}
                        </span>
                      </article>
                    ))}
                    {(forecast?.contributors || []).length === 0 && <p className="text-slate-400">No risk contributors available yet.</p>}
                  </div>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4 space-y-3">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 inline-flex items-center gap-2">
                    <Shield size={14} /> Triage Snapshot
                  </h3>
                  <p className="text-sm text-slate-200">Current risk level: <span className="text-cyan-300 font-semibold">{triage?.risk_level || 'stable'}</span></p>
                  <p className="text-sm text-slate-200">Current risk score: <span className="text-cyan-300 font-semibold">{triage?.risk_score ?? 0}</span></p>
                  <p className="text-sm text-slate-200">Relapse probability: <span className="text-cyan-300 font-semibold">{triage?.relapse_probability_pct ?? 0}%</span></p>
                  <p className="text-sm text-slate-200">Relapse band: <span className="text-cyan-300 font-semibold">{triage?.relapse_band || 'low'}</span></p>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Warning Signs</h3>
                  <ul className="space-y-1 text-sm text-slate-200">
                    {(forecast?.warning_signs || []).map((item, idx) => (
                      <li key={`warn-${idx}`}>• {item}</li>
                    ))}
                    {(forecast?.warning_signs || []).length === 0 && <li className="text-slate-400">No warning signs identified.</li>}
                  </ul>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Protective Signals</h3>
                  <ul className="space-y-1 text-sm text-slate-200">
                    {(forecast?.protective_signals || []).map((item, idx) => (
                      <li key={`protect-${idx}`}>• {item}</li>
                    ))}
                    {(forecast?.protective_signals || []).length === 0 && <li className="text-slate-400">No protective signals identified.</li>}
                  </ul>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Immediate Preventive Actions</h3>
                  <ul className="space-y-1 text-sm text-slate-200">
                    {(forecast?.preventive_actions || []).map((item, idx) => (
                      <li key={`action-${idx}`}>• {item}</li>
                    ))}
                    {(forecast?.preventive_actions || []).length === 0 && <li className="text-slate-400">No actions available yet.</li>}
                  </ul>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Screening Snapshot</h3>
                  <div className="space-y-2 text-sm">
                    {screeningRows.map((row) => (
                      <article key={row.type} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                        <p className="text-cyan-300 font-semibold">{row.type}</p>
                        <p className="text-slate-200">Score: {row.score}</p>
                        <p className="text-slate-300 text-xs">{row.severity} | trend: {row.trend}</p>
                      </article>
                    ))}
                    {screeningRows.length === 0 && <p className="text-slate-400">No screening snapshot available.</p>}
                  </div>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4 space-y-3">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-1">CBT + Adherence Snapshot</h3>
                  <p className="text-sm text-slate-200">CBT records: {cbtSnapshot.records_last_window ?? 0}</p>
                  <p className="text-sm text-slate-200">CBT trend: {cbtSnapshot.trend || 'insufficient_data'}</p>
                  <p className="text-sm text-slate-200">CBT improvement: {cbtSnapshot.improvement_pct ?? 0}%</p>
                  <p className="text-sm text-slate-200">Completion quality: {cbtSnapshot.completion_rate ?? 0}%</p>
                  <p className="text-sm text-slate-200">Streak: {cbtSnapshot.streak_days ?? 0} day(s)</p>
                  <p className="text-sm text-slate-200">Avg mood: {adherence.avg_mood ?? 0} | Avg stress: {adherence.avg_stress ?? 0}</p>
                  <p className="text-sm text-slate-200">Avg sleep: {adherence.avg_sleep_hours ?? 0}h | Check-ins: {adherence.checkin_count ?? 0}</p>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Handoff Priorities</h3>
                  <ul className="space-y-1 text-sm text-slate-200">
                    {(handoff?.handoff_priorities || []).map((item, idx) => (
                      <li key={`priority-${idx}`}>• {item}</li>
                    ))}
                    {(handoff?.handoff_priorities || []).length === 0 && <li className="text-slate-400">No priorities generated yet.</li>}
                  </ul>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Next 7-Day Plan</h3>
                  <ul className="space-y-1 text-sm text-slate-200">
                    {(handoff?.next_7_day_plan || []).map((item, idx) => (
                      <li key={`plan-${idx}`}>• {item}</li>
                    ))}
                    {(handoff?.next_7_day_plan || []).length === 0 && <li className="text-slate-400">No plan generated yet.</li>}
                  </ul>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Escalation Criteria</h3>
                  <ul className="space-y-1 text-sm text-slate-200">
                    {(handoff?.escalation_criteria || []).map((item, idx) => (
                      <li key={`esc-${idx}`}>• {item}</li>
                    ))}
                    {(handoff?.escalation_criteria || []).length === 0 && <li className="text-slate-400">No escalation criteria generated yet.</li>}
                  </ul>
                </div>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default ClinicalHandoffPage;

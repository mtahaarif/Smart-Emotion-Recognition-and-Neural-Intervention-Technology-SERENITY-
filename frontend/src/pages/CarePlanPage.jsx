import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  AlertTriangle,
  ArrowLeft,
  Brain,
  CheckCircle2,
  Compass,
  Loader2,
  LogOut,
  Shield,
  Target,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const riskTone = (value) => {
  const risk = String(value || '').toLowerCase();
  if (risk === 'elevated') return 'border-rose-500/60 bg-rose-950/30 text-rose-200';
  if (risk === 'monitor') return 'border-amber-500/60 bg-amber-950/30 text-amber-200';
  return 'border-emerald-500/60 bg-emerald-950/30 text-emerald-200';
};

const scoreBarTone = (band) => {
  const value = String(band || '').toLowerCase();
  if (value === 'high') return 'bg-cyan-400';
  if (value === 'low') return 'bg-slate-500';
  return 'bg-indigo-400';
};

const toTitle = (value) => {
  const text = String(value || '').replace(/_/g, ' ').trim();
  if (!text) return '';
  return text.charAt(0).toUpperCase() + text.slice(1);
};

const CarePlanPage = ({ user, onLogout }) => {
  const [loading, setLoading] = useState(true);
  const [errorMessage, setErrorMessage] = useState('');
  const [payload, setPayload] = useState(null);
  const [targetChecks, setTargetChecks] = useState({});
  const [savingCheckin, setSavingCheckin] = useState(false);
  const [checkinMessage, setCheckinMessage] = useState('');
  const [checkinForm, setCheckinForm] = useState({
    mood_rating: 5,
    stress_rating: 5,
    energy_rating: 5,
    sleep_hours: 7,
    note: '',
  });

  const personality = useMemo(() => {
    if (payload?.personality_profile && typeof payload.personality_profile === 'object') {
      return payload.personality_profile;
    }
    return {};
  }, [payload]);

  const routine = useMemo(() => {
    if (payload?.personalized_routine && typeof payload.personalized_routine === 'object') {
      return payload.personalized_routine;
    }
    return {};
  }, [payload]);

  const traits = useMemo(() => {
    const rows = Array.isArray(personality?.traits) ? personality.traits : [];
    return rows.slice().sort((a, b) => Number(b?.score || 0) - Number(a?.score || 0));
  }, [personality]);

  const weeklyTargets = useMemo(() => {
    return Array.isArray(routine?.weekly_targets) ? routine.weekly_targets : [];
  }, [routine]);

  const checkinSummary = useMemo(() => {
    if (payload?.checkin_summary && typeof payload.checkin_summary === 'object') {
      return payload.checkin_summary;
    }
    return {
      count: 0,
      avg_mood: 0,
      avg_stress: 0,
      avg_energy: 0,
      avg_sleep_hours: 0,
      recent_completed_targets: [],
    };
  }, [payload]);

  const recentCheckins = useMemo(() => {
    return Array.isArray(payload?.recent_checkins) ? payload.recent_checkins : [];
  }, [payload]);

  const loadCarePlan = async () => {
    setLoading(true);
    setErrorMessage('');
    try {
      const response = await axios.get(`${API_BASE_URL}/api/care-plan`, {
        params: { username: user },
      });
      setPayload(response.data || null);
      setTargetChecks({});
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to load personalized care plan.';
      setErrorMessage(String(backendMessage));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadCarePlan();
  }, [user]);

  const toggleTarget = (index) => {
    setTargetChecks((prev) => ({
      ...prev,
      [index]: !prev[index],
    }));
  };

  const updateCheckinField = (key, value) => {
    setCheckinForm((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const submitCheckin = async () => {
    if (savingCheckin) {
      return;
    }

    const completedTargets = weeklyTargets
      .map((target, idx) => (targetChecks[idx] ? String(target?.title || '').trim() : ''))
      .filter((item) => item);

    setSavingCheckin(true);
    setCheckinMessage('');
    try {
      const mood = Math.max(1, Math.min(10, Number(checkinForm.mood_rating) || 5));
      const stress = Math.max(1, Math.min(10, Number(checkinForm.stress_rating) || 5));
      const energy = Math.max(1, Math.min(10, Number(checkinForm.energy_rating) || 5));
      const sleepHours = Math.max(0, Math.min(24, Number(checkinForm.sleep_hours) || 0));

      await axios.post(`${API_BASE_URL}/api/care-plan/checkins`, {
        username: user,
        mood_rating: mood,
        stress_rating: stress,
        energy_rating: energy,
        sleep_hours: sleepHours,
        completed_targets: completedTargets,
        note: String(checkinForm.note || '').trim(),
      });
      setCheckinMessage('Daily check-in saved successfully.');
      await loadCarePlan();
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Unable to save check-in right now.';
      setCheckinMessage(String(backendMessage));
    } finally {
      setSavingCheckin(false);
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
            <Compass className="text-cyan-300" /> Personalized Care Plan
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
              <Loader2 className="animate-spin" size={16} /> Building your adaptive care plan...
            </div>
          )}

          {!loading && errorMessage && (
            <div className="rounded-lg border border-rose-600/60 bg-rose-950/30 p-4 text-rose-200 inline-flex items-center gap-2">
              <AlertTriangle size={16} /> {errorMessage}
            </div>
          )}

          {!loading && !errorMessage && payload && (
            <>
              <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-cyan-300">Clinical Focus</p>
                    <h2 className="text-2xl font-bold text-white mt-1">{routine?.focus_theme || 'Adaptive support plan'}</h2>
                    <p className="text-slate-300 mt-2 max-w-3xl">{payload?.follow_up_priority || 'Continue routine supportive follow-up.'}</p>
                    <p className="text-xs text-slate-500 mt-2">Cadence: {payload?.monitoring_cadence || 'Weekly review with symptom monitoring.'}</p>
                  </div>
                  <span className={`px-3 py-1 rounded-full border text-xs font-medium uppercase ${riskTone(payload?.risk_level)}`}>
                    {payload?.risk_level || 'stable'} (score: {payload?.risk_score ?? 0})
                  </span>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
                    <Brain size={14} /> Personality Pattern Estimate
                  </h3>
                  <p className="text-xs text-slate-500 mb-3">{personality?.disclaimer || 'Trait view unavailable.'}</p>
                  <div className="space-y-3">
                    {traits.map((trait) => {
                      const score = Number(trait?.score || 0);
                      return (
                        <article key={trait.key} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                          <div className="flex justify-between items-center text-sm">
                            <span className="text-slate-200 font-semibold">{trait.label}</span>
                            <span className="text-cyan-300">{score.toFixed(1)}</span>
                          </div>
                          <div className="w-full h-2 rounded bg-slate-800 mt-2 overflow-hidden">
                            <div className={`h-full ${scoreBarTone(trait?.band)}`} style={{ width: `${Math.max(0, Math.min(100, score))}%` }} />
                          </div>
                          <p className="text-xs text-slate-400 mt-2">{trait?.insight || ''}</p>
                        </article>
                      );
                    })}
                    {traits.length === 0 && <p className="text-slate-400 text-sm">No trait estimate available yet. Interact more and complete questionnaires.</p>}
                  </div>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
                    <Shield size={14} /> Safety Protocol
                  </h3>
                  <div className="space-y-3 text-sm">
                    <div>
                      <p className="text-slate-400 text-xs uppercase">Warning Signs</p>
                      <ul className="mt-1 space-y-1 text-slate-200">
                        {(routine?.safety_protocol?.warning_signs || []).map((row, idx) => (
                          <li key={`warn-${idx}`}>• {row}</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <p className="text-slate-400 text-xs uppercase">Immediate Steps</p>
                      <ul className="mt-1 space-y-1 text-slate-200">
                        {(routine?.safety_protocol?.immediate_steps || []).map((row, idx) => (
                          <li key={`step-${idx}`}>• {row}</li>
                        ))}
                      </ul>
                    </div>
                    <p className="text-rose-300 text-xs">{routine?.safety_protocol?.escalation || ''}</p>
                  </div>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Daily Routine Blueprint</h3>
                  <div className="space-y-3 text-sm">
                    {['morning', 'daytime', 'evening'].map((part) => (
                      <article key={part} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                        <p className="text-cyan-300 font-semibold">{toTitle(part)}</p>
                        <ul className="mt-2 space-y-1 text-slate-200">
                          {(routine?.daily_routine?.[part] || []).map((item, idx) => (
                            <li key={`${part}-${idx}`}>• {item}</li>
                          ))}
                        </ul>
                      </article>
                    ))}
                  </div>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4 space-y-4">
                  <div>
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
                      <Target size={14} /> Weekly Targets
                    </h3>
                    <div className="space-y-2 text-sm">
                      {weeklyTargets.map((target, idx) => (
                        <button
                          type="button"
                          key={`target-${idx}`}
                          onClick={() => toggleTarget(idx)}
                          className={`w-full text-left rounded-lg border p-3 transition ${targetChecks[idx] ? 'border-emerald-500/60 bg-emerald-950/20' : 'border-cyan-900/40 bg-slate-950/70'}`}
                        >
                          <p className="text-slate-100 font-medium inline-flex items-center gap-2">
                            {targetChecks[idx] ? <CheckCircle2 size={15} className="text-emerald-300" /> : <span className="w-3 h-3 rounded-full border border-slate-400 inline-block" />}
                            {target?.title || 'Target'}
                          </p>
                          <p className="text-xs text-cyan-300 mt-1">Metric: {target?.metric || 'Not set'}</p>
                          <p className="text-xs text-slate-400 mt-1">{target?.why || ''}</p>
                        </button>
                      ))}
                      {weeklyTargets.length === 0 && <p className="text-slate-400">No weekly targets available.</p>}
                    </div>
                  </div>

                  <div>
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Micro-Interventions</h3>
                    <ul className="space-y-1 text-sm text-slate-200">
                      {(routine?.micro_interventions || []).map((item, idx) => (
                        <li key={`micro-${idx}`}>• {item}</li>
                      ))}
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Monitoring Next Steps</h3>
                    <ul className="space-y-1 text-sm text-slate-200">
                      {(routine?.monitoring?.next_screenings || []).map((item, idx) => (
                        <li key={`screen-${idx}`}>• {item}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 space-y-3">
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300">Daily Check-In</h3>

                    <div className="grid grid-cols-2 gap-2 text-xs text-slate-300">
                      <label className="flex flex-col gap-1">
                        Mood (1-10)
                        <input
                          type="number"
                          min="1"
                          max="10"
                          value={checkinForm.mood_rating}
                          onChange={(e) => updateCheckinField('mood_rating', e.target.value)}
                          className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        Stress (1-10)
                        <input
                          type="number"
                          min="1"
                          max="10"
                          value={checkinForm.stress_rating}
                          onChange={(e) => updateCheckinField('stress_rating', e.target.value)}
                          className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        Energy (1-10)
                        <input
                          type="number"
                          min="1"
                          max="10"
                          value={checkinForm.energy_rating}
                          onChange={(e) => updateCheckinField('energy_rating', e.target.value)}
                          className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        />
                      </label>
                      <label className="flex flex-col gap-1">
                        Sleep Hours
                        <input
                          type="number"
                          min="0"
                          max="24"
                          step="0.5"
                          value={checkinForm.sleep_hours}
                          onChange={(e) => updateCheckinField('sleep_hours', e.target.value)}
                          className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        />
                      </label>
                    </div>

                    <label className="text-xs text-slate-300 flex flex-col gap-1">
                      Reflection Note
                      <textarea
                        value={checkinForm.note}
                        onChange={(e) => updateCheckinField('note', e.target.value)}
                        rows={3}
                        className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        placeholder="What felt difficult or helpful today?"
                      />
                    </label>

                    <button
                      type="button"
                      onClick={submitCheckin}
                      disabled={savingCheckin}
                      className="w-full rounded border border-cyan-500/60 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-60"
                    >
                      {savingCheckin ? 'Saving check-in...' : 'Save Daily Check-In'}
                    </button>
                    {checkinMessage && <p className="text-xs text-slate-300">{checkinMessage}</p>}
                  </div>

                  <div className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Recent Monitoring Snapshot</h3>
                    <p className="text-xs text-slate-400">Check-ins logged: {checkinSummary.count || 0}</p>
                    <p className="text-xs text-slate-300 mt-1">Avg mood: {checkinSummary.avg_mood || 0} | Avg stress: {checkinSummary.avg_stress || 0}</p>
                    <p className="text-xs text-slate-300">Avg energy: {checkinSummary.avg_energy || 0} | Avg sleep: {checkinSummary.avg_sleep_hours || 0}h</p>

                    {(checkinSummary.recent_completed_targets || []).length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs text-cyan-300">Recently completed targets</p>
                        <ul className="mt-1 space-y-1 text-xs text-slate-200">
                          {(checkinSummary.recent_completed_targets || []).map((item, idx) => (
                            <li key={`recent-target-${idx}`}>• {item}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  <div className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Latest Check-Ins</h3>
                    <div className="space-y-2 text-xs text-slate-300 max-h-44 overflow-auto pr-1">
                      {recentCheckins.slice(0, 6).map((entry) => (
                        <div key={entry.id} className="rounded border border-slate-800 p-2">
                          <p className="text-slate-400">{entry.created_at ? new Date(entry.created_at).toLocaleString() : 'Unknown time'}</p>
                          <p>Mood {entry.mood_rating} | Stress {entry.stress_rating} | Energy {entry.energy_rating} | Sleep {entry.sleep_hours}h</p>
                          {entry.note && <p className="text-slate-400 mt-1">{entry.note}</p>}
                        </div>
                      ))}
                      {recentCheckins.length === 0 && <p className="text-slate-400">No check-ins yet. Save your first daily check-in.</p>}
                    </div>
                  </div>
                </div>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
};

export default CarePlanPage;

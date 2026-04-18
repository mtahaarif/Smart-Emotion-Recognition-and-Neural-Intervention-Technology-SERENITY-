import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  AlertTriangle,
  ArrowLeft,
  Brain,
  CheckCircle2,
  FileText,
  Lightbulb,
  Loader2,
  LogOut,
  Shield,
  Target,
  TrendingUp,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const trendTone = (value) => {
  const trend = String(value || '').toLowerCase();
  if (trend === 'improving') return 'border-emerald-500/60 bg-emerald-950/25 text-emerald-200';
  if (trend === 'worsening') return 'border-rose-500/60 bg-rose-950/25 text-rose-200';
  return 'border-amber-500/60 bg-amber-950/25 text-amber-200';
};

const initialForm = {
  situation: '',
  automatic_thought: '',
  emotion_label: '',
  intensity_before: 6,
  evidence_for: '',
  evidence_against: '',
  balanced_thought: '',
  intensity_after: 4,
  action_plan: '',
};

const CBTWorkbenchPage = ({ user, onLogout }) => {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [successMessage, setSuccessMessage] = useState('');

  const [promptsPayload, setPromptsPayload] = useState(null);
  const [recordsPayload, setRecordsPayload] = useState(null);
  const [lastDistortions, setLastDistortions] = useState([]);
  const [form, setForm] = useState(initialForm);

  const prompts = useMemo(() => {
    if (promptsPayload?.prompts && typeof promptsPayload.prompts === 'object') {
      return promptsPayload.prompts;
    }
    return {};
  }, [promptsPayload]);

  const progress = useMemo(() => {
    if (recordsPayload?.weekly_progress && typeof recordsPayload.weekly_progress === 'object') {
      return recordsPayload.weekly_progress;
    }
    return {
      total_records: 0,
      avg_intensity_before: 0,
      avg_intensity_after: 0,
      avg_intensity_reduction: 0,
      improvement_pct: 0,
      completion_rate: 0,
      streak_days: 0,
      top_distortions: [],
      trend: 'insufficient_data',
    };
  }, [recordsPayload]);

  const records = useMemo(() => {
    return Array.isArray(recordsPayload?.records) ? recordsPayload.records : [];
  }, [recordsPayload]);

  const guidance = useMemo(() => {
    return Array.isArray(recordsPayload?.coaching_guidance) ? recordsPayload.coaching_guidance : [];
  }, [recordsPayload]);

  const loadData = async () => {
    setLoading(true);
    setErrorMessage('');
    try {
      const [promptsRes, recordsRes] = await Promise.all([
        axios.get(`${API_BASE_URL}/api/cbt/prompts`, { params: { username: user } }),
        axios.get(`${API_BASE_URL}/api/cbt/thought-records`, { params: { username: user, limit: 25 } }),
      ]);
      setPromptsPayload(promptsRes.data || null);
      setRecordsPayload(recordsRes.data || null);
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to load CBT workbench data.';
      setErrorMessage(String(backendMessage));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [user]);

  const updateField = (key, value) => {
    setForm((prev) => ({
      ...prev,
      [key]: value,
    }));
  };

  const submitRecord = async () => {
    if (saving) {
      return;
    }

    const situation = String(form.situation || '').trim();
    const automaticThought = String(form.automatic_thought || '').trim();
    const balancedThought = String(form.balanced_thought || '').trim();
    if (!situation || !automaticThought || !balancedThought) {
      setErrorMessage('Situation, automatic thought, and balanced thought are required.');
      return;
    }

    setSaving(true);
    setErrorMessage('');
    setSuccessMessage('');

    try {
      const response = await axios.post(`${API_BASE_URL}/api/cbt/thought-records`, {
        username: user,
        situation,
        automatic_thought: automaticThought,
        emotion_label: String(form.emotion_label || '').trim(),
        intensity_before: Number(form.intensity_before),
        evidence_for: String(form.evidence_for || '').trim(),
        evidence_against: String(form.evidence_against || '').trim(),
        balanced_thought: balancedThought,
        intensity_after: Number(form.intensity_after),
        action_plan: String(form.action_plan || '').trim(),
      });

      const detected = Array.isArray(response.data?.detected_distortions)
        ? response.data.detected_distortions
        : [];
      setLastDistortions(detected);
      setSuccessMessage('Thought record saved. Weekly progress has been updated.');
      setForm((prev) => ({
        ...initialForm,
        emotion_label: prev.emotion_label,
      }));
      await loadData();
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Unable to save CBT thought record.';
      setErrorMessage(String(backendMessage));
    } finally {
      setSaving(false);
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
            <FileText className="text-cyan-300" /> CBT Workbench
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
              <Loader2 className="animate-spin" size={16} /> Loading CBT workbench...
            </div>
          )}

          {!loading && errorMessage && (
            <div className="rounded-lg border border-rose-600/60 bg-rose-950/30 p-4 text-rose-200 inline-flex items-center gap-2">
              <AlertTriangle size={16} /> {errorMessage}
            </div>
          )}

          {!loading && successMessage && (
            <div className="rounded-lg border border-emerald-600/60 bg-emerald-950/30 p-4 text-emerald-200 inline-flex items-center gap-2">
              <CheckCircle2 size={16} /> {successMessage}
            </div>
          )}

          {!loading && !errorMessage && (
            <>
              <section className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-5">
                <div className="flex flex-wrap items-start justify-between gap-3">
                  <div>
                    <p className="text-xs uppercase tracking-wider text-cyan-300">CBT Session Focus</p>
                    <h2 className="text-2xl font-bold text-white mt-1">{prompts?.focus_hint || 'Cognitive restructuring session'}</h2>
                    <p className="text-slate-300 mt-2 max-w-3xl">
                      {prompts?.session_goal || 'Complete one guided thought record and track emotional intensity shift.'}
                    </p>
                  </div>
                  <span className={`px-3 py-1 rounded-full border text-xs font-medium uppercase ${trendTone(progress?.trend)}`}>
                    Weekly trend: {String(progress?.trend || 'insufficient_data').replace('_', ' ')}
                  </span>
                </div>
                <p className="text-xs text-rose-300 mt-3 inline-flex items-center gap-1">
                  <Shield size={12} /> {prompts?.safety_reminder || 'If you feel at immediate risk, contact emergency services.'}
                </p>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-3 gap-4">
                <div className="xl:col-span-2 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
                    <Brain size={14} /> Guided Prompts
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-sm">
                    <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                      <p className="text-cyan-300 text-xs uppercase">Opening</p>
                      <ul className="mt-2 space-y-1 text-slate-200">
                        {(prompts?.opening_prompts || []).map((item, idx) => (
                          <li key={`opening-${idx}`}>• {item}</li>
                        ))}
                      </ul>
                    </article>
                    <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                      <p className="text-cyan-300 text-xs uppercase">Evidence</p>
                      <ul className="mt-2 space-y-1 text-slate-200">
                        {(prompts?.evidence_prompts || []).map((item, idx) => (
                          <li key={`evidence-${idx}`}>• {item}</li>
                        ))}
                      </ul>
                    </article>
                    <article className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                      <p className="text-cyan-300 text-xs uppercase">Reframe</p>
                      <ul className="mt-2 space-y-1 text-slate-200">
                        {(prompts?.reframe_prompts || []).map((item, idx) => (
                          <li key={`reframe-${idx}`}>• {item}</li>
                        ))}
                      </ul>
                    </article>
                  </div>
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3 inline-flex items-center gap-2">
                    <TrendingUp size={14} /> Weekly Progress
                  </h3>
                  <div className="space-y-2 text-sm text-slate-200">
                    <p>Total records: {progress.total_records || 0}</p>
                    <p>Avg intensity before: {progress.avg_intensity_before || 0}</p>
                    <p>Avg intensity after: {progress.avg_intensity_after || 0}</p>
                    <p>Avg reduction: {progress.avg_intensity_reduction || 0}</p>
                    <p>Improvement: {progress.improvement_pct || 0}%</p>
                    <p>Completion quality: {progress.completion_rate || 0}%</p>
                    <p>Current streak: {progress.streak_days || 0} day(s)</p>
                  </div>
                  <div className="mt-3">
                    <p className="text-xs uppercase tracking-wider text-cyan-300">Top Distortions</p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {(progress.top_distortions || []).map((item, idx) => (
                        <span key={`dist-${idx}`} className="px-2 py-1 rounded-full border border-cyan-900/40 bg-slate-950/70 text-xs text-slate-200">
                          {String(item.distortion || '').replace(/_/g, ' ')} ({item.count})
                        </span>
                      ))}
                      {(progress.top_distortions || []).length === 0 && (
                        <span className="text-xs text-slate-400">No distortion trends yet</span>
                      )}
                    </div>
                  </div>
                </div>
              </section>

              <section className="grid grid-cols-1 xl:grid-cols-2 gap-4">
                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4 space-y-3">
                  <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-1 inline-flex items-center gap-2">
                    <Target size={14} /> New Thought Record
                  </h3>

                  <label className="text-xs text-slate-300 flex flex-col gap-1">
                    Situation
                    <textarea
                      value={form.situation}
                      onChange={(e) => updateField('situation', e.target.value)}
                      rows={3}
                      className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                      placeholder="Where and when did this happen?"
                    />
                  </label>

                  <label className="text-xs text-slate-300 flex flex-col gap-1">
                    Automatic Thought
                    <textarea
                      value={form.automatic_thought}
                      onChange={(e) => updateField('automatic_thought', e.target.value)}
                      rows={3}
                      className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                      placeholder="Write the thought exactly as it appeared."
                    />
                  </label>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <label className="text-xs text-slate-300 flex flex-col gap-1">
                      Emotion Label
                      <input
                        type="text"
                        value={form.emotion_label}
                        onChange={(e) => updateField('emotion_label', e.target.value)}
                        className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        placeholder="e.g., fear, sadness, anger"
                      />
                    </label>

                    <label className="text-xs text-slate-300 flex flex-col gap-1">
                      Intensity Before: {form.intensity_before}
                      <input
                        type="range"
                        min="0"
                        max="10"
                        value={form.intensity_before}
                        onChange={(e) => updateField('intensity_before', Number(e.target.value))}
                        className="accent-cyan-400"
                      />
                    </label>
                  </div>

                  <label className="text-xs text-slate-300 flex flex-col gap-1">
                    Evidence Supporting the Thought
                    <textarea
                      value={form.evidence_for}
                      onChange={(e) => updateField('evidence_for', e.target.value)}
                      rows={2}
                      className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-300 flex flex-col gap-1">
                    Evidence Against the Thought
                    <textarea
                      value={form.evidence_against}
                      onChange={(e) => updateField('evidence_against', e.target.value)}
                      rows={2}
                      className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                    />
                  </label>

                  <label className="text-xs text-slate-300 flex flex-col gap-1">
                    Balanced Thought
                    <textarea
                      value={form.balanced_thought}
                      onChange={(e) => updateField('balanced_thought', e.target.value)}
                      rows={3}
                      className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                      placeholder="Write a realistic and compassionate reframe."
                    />
                  </label>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-2">
                    <label className="text-xs text-slate-300 flex flex-col gap-1">
                      Intensity After: {form.intensity_after}
                      <input
                        type="range"
                        min="0"
                        max="10"
                        value={form.intensity_after}
                        onChange={(e) => updateField('intensity_after', Number(e.target.value))}
                        className="accent-emerald-400"
                      />
                    </label>

                    <label className="text-xs text-slate-300 flex flex-col gap-1">
                      24-Hour Action Plan
                      <input
                        type="text"
                        value={form.action_plan}
                        onChange={(e) => updateField('action_plan', e.target.value)}
                        className="rounded border border-cyan-900/50 bg-slate-900 px-2 py-1 text-sm text-slate-100"
                        placeholder="One small next action"
                      />
                    </label>
                  </div>

                  <button
                    type="button"
                    onClick={submitRecord}
                    disabled={saving}
                    className="w-full rounded border border-cyan-500/60 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-200 hover:bg-cyan-500/20 disabled:opacity-60"
                  >
                    {saving ? 'Saving thought record...' : 'Save Thought Record'}
                  </button>

                  {lastDistortions.length > 0 && (
                    <div className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3">
                      <p className="text-xs uppercase tracking-wider text-cyan-300 inline-flex items-center gap-1">
                        <Lightbulb size={12} /> Detected Distortions
                      </p>
                      <ul className="mt-2 space-y-2 text-xs text-slate-200">
                        {lastDistortions.map((item, idx) => (
                          <li key={`detect-${idx}`}>
                            <p className="font-semibold text-cyan-300">{item.label}</p>
                            <p className="text-slate-400">Evidence: {item.evidence}</p>
                            <p>{item.challenge_prompt}</p>
                          </li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4 space-y-4">
                  <div>
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Coaching Guidance</h3>
                    <ul className="space-y-1 text-sm text-slate-200">
                      {guidance.map((item, idx) => (
                        <li key={`guidance-${idx}`}>• {item}</li>
                      ))}
                      {guidance.length === 0 && <li className="text-slate-400">No guidance yet. Complete a thought record to begin.</li>}
                    </ul>
                  </div>

                  <div>
                    <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Recent Thought Records</h3>
                    <div className="space-y-2 max-h-[540px] overflow-auto pr-1">
                      {records.map((record) => (
                        <article key={record.id} className="rounded-lg border border-cyan-900/40 bg-slate-950/70 p-3 text-xs text-slate-200">
                          <p className="text-slate-400">{record.created_at ? new Date(record.created_at).toLocaleString() : 'Unknown time'}</p>
                          <p className="mt-1"><span className="text-cyan-300">Situation:</span> {record.situation}</p>
                          <p><span className="text-cyan-300">Thought:</span> {record.automatic_thought}</p>
                          <p><span className="text-cyan-300">Reframe:</span> {record.balanced_thought}</p>
                          <p className="mt-1 text-slate-400">
                            Intensity {record.intensity_before} → {record.intensity_after}
                            {' '}({record.intensity_before - record.intensity_after >= 0 ? '-' : '+'}
                            {Math.abs(record.intensity_before - record.intensity_after)})
                          </p>
                        </article>
                      ))}
                      {records.length === 0 && <p className="text-slate-400 text-sm">No thought records saved yet.</p>}
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

export default CBTWorkbenchPage;

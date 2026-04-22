import React, { useCallback, useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  CheckCircle2,
  Loader2,
  LogOut,
  RefreshCw,
  ShieldAlert,
  ClipboardList,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useClinical } from '../context/ClinicalContext';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const SERIES = [
  { key: 'phq9', label: 'Depression (PHQ-9)', color: '#818cf8' }, // Indigo
  { key: 'gad7', label: 'Anxiety (GAD-7)', color: '#fbbf24' },    // Amber
  { key: 'pcl5_scaled_27', label: 'Trauma (PCL-5 Scaled)', color: '#34d399' }, // Emerald
];

const parseDate = (value) => {
  if (!value) return null;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? null : parsed;
};

const formatChartTick = (value) => {
  const parsed = parseDate(value);
  if (!parsed) return String(value || 'Unknown');
  return parsed.toLocaleDateString('en-US', { 
    timeZone: 'Asia/Karachi', 
    month: 'short', 
    day: '2-digit' 
  });
};

const formatDateTime = (value) => {
  const parsed = parseDate(value);
  if (!parsed) return String(value || 'Unknown');
  return parsed.toLocaleString('en-US', { 
    timeZone: 'Asia/Karachi' 
  });
};
const TrajectoryTooltip = ({ active, payload, label }) => {
  if (!active || !Array.isArray(payload) || payload.length === 0) return null;
  const row = payload[0]?.payload || {};
  return (
    <div className="rounded-xl border border-slate-700 bg-slate-900/95 p-4 shadow-2xl">
      <p className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-3">{formatDateTime(row.timestamp || label)}</p>
      <div className="space-y-2">
        <p className="text-xs font-bold text-slate-300">PHQ-9 Score: <span className="font-black text-indigo-400 ml-1">{row.phq9 ?? 'N/A'}</span></p>
        <p className="text-xs font-bold text-slate-300">GAD-7 Score: <span className="font-black text-amber-400 ml-1">{row.gad7 ?? 'N/A'}</span></p>
        <p className="text-xs font-bold text-slate-300">PCL-5 Score: <span className="font-black text-emerald-400 ml-1">{row.pcl5 ?? 'N/A'}</span></p>
      </div>
    </div>
  );
};

const VelocityBadge = ({ label, value }) => {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return null;

  const classes = numeric > 3
    ? 'border-rose-900/50 bg-rose-950/30 text-rose-400'
    : numeric < 0
      ? 'border-emerald-900/50 bg-emerald-950/30 text-emerald-400'
      : 'border-slate-800 bg-slate-900/80 text-slate-400';

  const sign = numeric > 0 ? '+' : '';
  return (
    <span className={`inline-flex items-center rounded-lg border px-3 py-1 text-[10px] font-black uppercase tracking-widest ${classes}`}>
      {label}: {sign}{numeric}
    </span>
  );
};

const RationaleBadge = ({ text }) => {
  if (!text) return null;
  let tone = 'bg-emerald-950/30 text-emerald-400 border-emerald-900/50'; 
  const lower = text.toLowerCase();
  
  if (lower.includes('anxiety') || lower.includes('arousal') || lower.includes('panic') || lower.includes('tension')) {
    tone = 'bg-amber-950/30 text-amber-400 border-amber-900/50';
  } else if (lower.includes('depression') || lower.includes('motivation') || lower.includes('activation') || lower.includes('lethargy')) {
    tone = 'bg-indigo-950/30 text-indigo-400 border-indigo-900/50';
  } else if (lower.includes('trauma') || lower.includes('stress') || lower.includes('hypervigilance')) {
    tone = 'bg-rose-950/30 text-rose-400 border-rose-900/50';
  } else if (lower.includes('maintenance') || lower.includes('structuring')) {
    tone = 'bg-slate-800/80 text-slate-300 border-slate-700';
  }

  return (
    <span className={`mt-3 inline-block px-2.5 py-1 text-[9px] font-black uppercase tracking-widest border rounded-md shadow-sm ${tone}`}>
      {text}
    </span>
  );
};

const MBCHubPage = ({ user, onLogout }) => {
  const navigate = useNavigate();
  const { ingestBackendEvent, setActiveRiskScore, setCurrentTherapyMode } = useClinical();

  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  
  const [visibleSeries, setVisibleSeries] = useState({ phq9: true, gad7: true, pcl5_scaled_27: true });
  
  const [adherenceDateKey, setAdherenceDateKey] = useState(() => new Date().toISOString().slice(0, 10));
  const [adherenceChecks, setAdherenceChecks] = useState({});

  const username = String(user || localStorage.getItem('serenity_user') || '').trim();

  const fetchTrajectory = async (refresh = false) => {
    if (!username) return;
    refresh ? setIsRefreshing(true) : setIsLoading(true);
    setErrorMessage('');

    try {
      const response = await axios.get(`${API_BASE_URL}/api/mbc/trajectory`, { params: { username, refresh } });
      const payload = response.data || {};
      setData(payload);

      if (payload?.care_plan?.framework) setCurrentTherapyMode(String(payload.care_plan.framework));

      const scoreCandidates = [Number(payload?.latest_scores?.['PHQ-9'] ?? 0), Number(payload?.latest_scores?.['GAD-7'] ?? 0)].filter(Number.isFinite);
      if (scoreCandidates.length > 0) setActiveRiskScore(Math.max(...scoreCandidates));

      if (payload?.requires_safety_review === true) {
        ingestBackendEvent({ type: 'SAFETY_MODE', enabled: true, clinical: { requires_safety_review: true, isCrisisMode: true } });
      }
    } catch (error) {
      setErrorMessage('Unable to synchronize MBC trajectory data from the server.');
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => { fetchTrajectory(false); }, [username]);

  useEffect(() => {
    const today = new Date().toISOString().slice(0, 10);
    if (today !== adherenceDateKey) {
      setAdherenceDateKey(today);
      setAdherenceChecks({});
    }
  }, [adherenceDateKey]);

  const chartData = useMemo(() => {
    const rows = Array.isArray(data?.time_series) ? data.time_series : [];
    const perDay = new Map();
    
    rows.forEach((row) => {
      const parsed = parseDate(row.date || row.timestamp);
      if (!parsed) return;
      const dayStart = new Date(parsed.getFullYear(), parsed.getMonth(), parsed.getDate());
      const dayKey = dayStart.toISOString().slice(0, 10);
      
      // FIXED: Merge the data together for the same day instead of overwriting!
      const existing = perDay.get(dayKey) || { chartTimestamp: dayStart.getTime() };
      Object.entries(row).forEach(([key, val]) => {
        if (val !== undefined && val !== null) {
          existing[key] = val;
        }
      });
      
      perDay.set(dayKey, existing);
    });
    return Array.from(perDay.values()).sort((a, b) => a.chartTimestamp - b.chartTimestamp);
  }, [data]);

  const pendingAssessments = useMemo(() => (Array.isArray(data?.pending_assessments) ? data.pending_assessments : []), [data]);
  
  const routineBlueprint = useMemo(() => (Array.isArray(data?.care_plan?.daily_routine_blueprint) ? data.care_plan.daily_routine_blueprint : []), [data]);
  const interventions = useMemo(() => (Array.isArray(data?.care_plan?.micro_interventions) ? data.care_plan.micro_interventions : []), [data]);

  const toggleAdherenceItem = useCallback((checkKey) => {
    setAdherenceChecks((prev) => ({ ...prev, [checkKey]: !prev[checkKey] }));
  }, []);

  const adherenceStats = useMemo(() => {
    const allKeys = [...routineBlueprint.map((i) => `routine:${i.id}`), ...interventions.map((i) => `intervention:${i.id}`)];
    const total = allKeys.length;
    const completed = allKeys.filter((k) => Boolean(adherenceChecks[k])).length;
    return { total, completed, percent: total > 0 ? Math.round((completed / total) * 100) : 0 };
  }, [routineBlueprint, interventions, adherenceChecks]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500/30">
      
      {/* PROFESSIONAL NAV */}
      <nav className="border-b border-slate-800 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="text-slate-400 hover:text-white transition-colors p-2 hover:bg-slate-900 rounded-lg">
            <ArrowLeft size={20} />
          </Link>
          <div className="h-8 w-px bg-slate-800 mx-1 hidden md:block" />
          <h1 className="text-xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
            <Activity className="text-emerald-500" size={22} /> SERENITY <span className="text-slate-500 font-normal">Measurement-Based Care</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Active Patient</p>
            <p className="text-sm font-semibold text-slate-200">{username}</p>
          </div>
          <button onClick={onLogout} className="text-rose-400 text-xs font-black uppercase tracking-widest border border-rose-900/40 px-3 py-2 rounded-xl hover:bg-rose-950/30 transition-all flex items-center gap-2">
            <LogOut size={14} /> End Session
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6 max-w-[1400px] mx-auto w-full space-y-6">
        
        {/* HEADER SECTION */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8 shadow-sm">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="flex items-center gap-5">
              <div className="p-3 bg-indigo-500/10 rounded-2xl text-indigo-400 border border-indigo-500/20 hidden sm:block">
                <ClipboardList size={28} />
              </div>
              <div>
                <p className="text-[10px] uppercase font-black tracking-[0.2em] text-slate-500">Pillar 2</p>
                <h2 className="mt-1 text-2xl font-black tracking-tight text-slate-100">Dynamic MBC Dashboard</h2>
                <p className="mt-1 text-sm text-slate-400 max-w-2xl">
                  Longitudinal symptom trajectories driving an adaptive, measurement-based action queue.
                </p>
              </div>
            </div>

            <button onClick={() => fetchTrajectory(true)} disabled={isRefreshing} className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-5 py-2.5 text-xs font-black uppercase tracking-widest text-white hover:bg-indigo-500 disabled:opacity-50 transition-all shadow-lg shadow-indigo-900/20">
              {isRefreshing ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
              Refresh Trajectory
            </button>
          </div>

          {data?.requires_safety_review && (
            <div className="mt-6 rounded-2xl border-2 border-rose-600/50 bg-rose-950/30 px-5 py-4 text-rose-200 inline-flex items-center gap-3 shadow-[0_0_20px_rgba(225,29,72,0.1)]">
              <ShieldAlert size={20} className="text-rose-500" />
              <span className="text-sm font-bold">Clinical safety review indicated based on current symptom velocity and tracking history.</span>
            </div>
          )}
          
          {errorMessage && (
            <div className="mt-4 rounded-xl border border-rose-600/60 bg-rose-950/30 px-4 py-3 text-rose-200 text-sm inline-flex items-center gap-2">
              <AlertTriangle size={16} /> {errorMessage}
            </div>
          )}
        </section>

        {/* CHART SECTION */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-6">
            <div>
              <p className="text-[10px] uppercase font-black tracking-[0.2em] text-indigo-400">Trajectory Analysis</p>
              <h3 className="mt-1 text-lg font-bold text-white">Longitudinal Symptom Signals</h3>
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <VelocityBadge label="PHQ-9 Velocity" value={data?.velocity_delta?.['PHQ-9']} />
              <VelocityBadge label="GAD-7 Velocity" value={data?.velocity_delta?.['GAD-7']} />
            </div>
          </div>

          <div className="flex flex-wrap gap-3 mb-6">
            {SERIES.map((series) => (
              <button
                key={series.key}
                onClick={() => setVisibleSeries((prev) => ({ ...prev, [series.key]: !prev[series.key] }))}
                className={`rounded-xl border px-4 py-2 text-xs font-black tracking-widest uppercase transition-all shadow-sm ${
                  visibleSeries[series.key] ? 'bg-slate-800 text-white border-slate-600' : 'bg-slate-950/50 text-slate-500 border-slate-800 hover:border-slate-700'
                }`}
              >
                <span className="flex items-center gap-2">
                  <span className="h-2 w-2 rounded-full" style={{ backgroundColor: visibleSeries[series.key] ? series.color : '#334155' }} />
                  {series.label}
                </span>
              </button>
            ))}
          </div>

          {isLoading ? (
            <div className="h-80 rounded-2xl border border-slate-800 bg-slate-950/50 flex flex-col items-center justify-center text-slate-400 gap-3">
              <Loader2 size={24} className="animate-spin text-indigo-500" />
              <span className="text-xs font-black uppercase tracking-widest">Compiling historical data...</span>
            </div>
          ) : chartData.length === 0 ? (
            <div className="h-80 rounded-2xl border border-slate-800 bg-slate-950/50 flex items-center justify-center text-slate-500 text-sm font-medium">
              No longitudinal history available. Complete initial clinical assessments to establish a baseline.
            </div>
          ) : (
            <div className="h-80 rounded-2xl border border-slate-800 bg-slate-950/50 p-4">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 12, right: 18, left: 8, bottom: 10 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                  <XAxis dataKey="chartTimestamp" type="number" domain={['dataMin', 'dataMax']} tickFormatter={formatChartTick} stroke="#64748b" tickLine={false} axisLine={false} minTickGap={30} tick={{ fontSize: 11, fontWeight: 600 }} />
                  <YAxis stroke="#64748b" tickLine={false} axisLine={false} domain={[0, 27]} tick={{ fontSize: 11, fontWeight: 600 }} />
                  <Tooltip content={<TrajectoryTooltip />} cursor={{ stroke: '#334155', strokeWidth: 2 }} />
                  {SERIES.map((s) => visibleSeries[s.key] && (
                    <Line key={s.key} type="monotone" dataKey={s.key} stroke={s.color} strokeWidth={3} dot={{ r: 4, strokeWidth: 2, fill: '#020617' }} activeDot={{ r: 6 }} isAnimationActive={false} connectNulls />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </section>

        {/* DAILY ADHERENCE */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
          <div className="flex flex-wrap items-end justify-between gap-4 mb-4">
            <div>
              <p className="text-[10px] uppercase font-black tracking-[0.2em] text-emerald-400">Behavioral Maintenance</p>
              <h3 className="mt-1 text-lg font-bold text-white">Daily Adherence Tracking</h3>
            </div>
            <p className="text-sm font-black text-slate-300">
              {adherenceStats.completed} / {adherenceStats.total} <span className="text-slate-500 font-medium ml-1">Tasks ({adherenceStats.percent}%)</span>
            </p>
          </div>

          <div className="h-4 w-full overflow-hidden rounded-full bg-slate-950 border border-slate-800 shadow-inner">
            <div className="h-full bg-gradient-to-r from-indigo-500 to-emerald-400 transition-all duration-700 ease-out" style={{ width: `${adherenceStats.percent}%` }} />
          </div>
        </section>

        {/* DYNAMIC TASKS GRID */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          
          {/* ROUTINES */}
          <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
            <p className="text-[10px] uppercase font-black tracking-[0.2em] text-indigo-400">Structured Adaptation</p>
            <h3 className="mt-1 text-lg font-bold text-white mb-6">Prescribed Behavioral Routines</h3>
            <div className="space-y-4">
              {routineBlueprint.length === 0 ? <p className="text-sm text-slate-500 italic">No tailored routines currently assigned.</p> : (
                routineBlueprint.map((item) => {
                  const isChecked = Boolean(adherenceChecks[`routine:${item.id}`]);
                  return (
                    <label key={item.id} className={`rounded-2xl border transition-all duration-300 p-5 flex items-start gap-4 cursor-pointer ${isChecked ? 'bg-indigo-950/10 border-indigo-900/30' : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 hover:bg-slate-900/50 shadow-sm'}`}>
                      <input type="checkbox" checked={isChecked} onChange={() => toggleAdherenceItem(`routine:${item.id}`)} className="mt-1.5 h-5 w-5 shrink-0 rounded-md border-slate-600 bg-slate-900 text-indigo-500 focus:ring-indigo-500 cursor-pointer transition-colors" />
                      <div className={`transition-all duration-300 ${isChecked ? 'opacity-50 grayscale' : 'opacity-100'}`}>
                        <p className={`text-sm font-bold ${isChecked ? 'text-slate-400 line-through decoration-slate-500' : 'text-slate-100'}`}>{item.title}</p>
                        <p className={`text-xs mt-1.5 leading-relaxed ${isChecked ? 'text-slate-500 line-through decoration-slate-600' : 'text-slate-400'}`}>{item.description || item.objective}</p>
                        <RationaleBadge text={item.clinical_rationale} />
                      </div>
                    </label>
                  );
                })
              )}
            </div>
          </section>

          {/* INTERVENTIONS */}
          <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
            <p className="text-[10px] uppercase font-black tracking-[0.2em] text-amber-400">Symptom Regulation</p>
            <h3 className="mt-1 text-lg font-bold text-white mb-6">Acute Micro-Interventions</h3>
            <div className="space-y-4">
              {interventions.length === 0 ? <p className="text-sm text-slate-500 italic">No active interventions required.</p> : (
                interventions.map((item) => {
                  const isChecked = Boolean(adherenceChecks[`intervention:${item.id}`]);
                  return (
                    <label key={item.id} className={`rounded-2xl border transition-all duration-300 p-5 flex items-start gap-4 cursor-pointer ${isChecked ? 'bg-emerald-950/10 border-emerald-900/30' : 'bg-slate-950/50 border-slate-800 hover:border-slate-700 hover:bg-slate-900/50 shadow-sm'}`}>
                      <input type="checkbox" checked={isChecked} onChange={() => toggleAdherenceItem(`intervention:${item.id}`)} className="mt-1.5 h-5 w-5 shrink-0 rounded-md border-slate-600 bg-slate-900 text-emerald-500 focus:ring-emerald-500 cursor-pointer transition-colors" />
                      <div className={`transition-all duration-300 ${isChecked ? 'opacity-50 grayscale' : 'opacity-100'}`}>
                        <p className={`text-sm font-bold ${isChecked ? 'text-slate-400 line-through decoration-slate-500' : 'text-slate-100'}`}>{item.title}</p>
                        <p className={`text-xs mt-1.5 leading-relaxed ${isChecked ? 'text-slate-500 line-through decoration-slate-600' : 'text-slate-400'}`}>{item.description || item.objective}</p>
                        <RationaleBadge text={item.clinical_rationale} />
                      </div>
                    </label>
                  );
                })
              )}
            </div>
          </section>

        </div>

        {/* PENDING ASSESSMENTS */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
          <p className="text-[10px] uppercase font-black tracking-[0.2em] text-cyan-400">Assessment Cadence</p>
          <h3 className="mt-1 text-lg font-bold text-white mb-6">Pending Screenings</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-2">
            {pendingAssessments.length === 0 ? <p className="text-sm text-slate-500 italic">No assessment cadence data.</p> : (
              pendingAssessments.map((item) => {
                const needsAction = item.is_due || String(item.reason).toLowerCase().includes('overdue');
                return (
                  <div key={item.questionnaire_type} className={`rounded-2xl border p-5 transition-colors ${needsAction ? 'bg-rose-950/20 border-rose-900/50' : 'bg-slate-950/50 border-slate-800 hover:border-slate-700'}`}>
                    <div className="flex justify-between items-start mb-4">
                      <p className="text-base font-black text-slate-100">{item.questionnaire_type}</p>
                      <span className="text-[10px] font-black uppercase tracking-widest text-slate-400 bg-slate-900 px-2.5 py-1 rounded-md border border-slate-800">
                        {item.days_since_last === null ? 'No Record' : `Last: ${item.days_since_last} days`}
                      </span>
                    </div>
                    <p className="text-xs text-slate-400 mb-6 h-8 leading-relaxed">{item.reason}</p>
                    
                    {needsAction ? (
                      <button onClick={() => navigate('/questionnaires', { state: { questionnaireType: item.questionnaire_type } })} className="w-full rounded-xl bg-rose-600 py-3 text-xs font-black uppercase tracking-widest text-white hover:bg-rose-500 transition-colors shadow-lg shadow-rose-900/20">
                        Start Assessment
                      </button>
                    ) : (
                      <div className="w-full rounded-xl bg-slate-900/50 border border-slate-800 py-3 text-center text-xs font-black text-slate-500 uppercase tracking-widest flex items-center justify-center gap-2">
                        <CheckCircle2 size={14} /> Cadence Met
                      </div>
                    )}
                  </div>
                );
              })
            )}
          </div>
        </section>
      </main>
    </div>
  );
};

export default MBCHubPage;
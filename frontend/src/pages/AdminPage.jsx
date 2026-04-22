import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Brain,
  Clock3,
  Database,
  Download,
  FileText,
  Loader2,
  LogOut,
  MessageSquare,
  Shield,
  RefreshCw,
  CheckCircle2,
  Stethoscope,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const clamp = (value, min, max) => Math.max(min, Math.min(max, value));

const formatDate = (value) => {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString('en-US', { 
    timeZone: 'Asia/Karachi' 
  });
};

const acuityTone = (level) => {
  const key = String(level || '').toLowerCase();
  if (key === 'elevated' || key === 'high' || key === 'worsening') return 'border-rose-500/60 bg-rose-950/30 text-rose-200';
  if (key === 'monitor' || key === 'moderate') return 'border-amber-500/60 bg-amber-950/30 text-amber-200';
  return 'border-emerald-500/60 bg-emerald-950/30 text-emerald-200';
};

const MetricIcon = ({ id }) => {
  if (id === 'turns') return <MessageSquare size={15} />;
  if (id === 'care_plan_adherence') return <CheckCircle2 size={15} />;
  if (id === 'emotion_events') return <Brain size={15} />;
  if (id === 'questionnaire_entries') return <FileText size={15} />;
  if (id === 'risk_score') return <Shield size={15} />;
  if (id === 'distress_signals') return <AlertTriangle size={15} />;
  return <Database size={15} />;
};

const AdminPage = ({ user, onLogout }) => {
  const [overview, setOverview] = useState(null);
  const [isClinicalReportLoading, setIsClinicalReportLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');
  const [clinicalReportError, setClinicalReportError] = useState('');
  const [handoffError, setHandoffError] = useState('');
  const [isExporting, setIsExporting] = useState(false);
  const [limit, setLimit] = useState(300);
  const [chatVisibleCount, setChatVisibleCount] = useState(30);
  const [clinicalReport, setClinicalReport] = useState('');
  const [clinicalReportSource, setClinicalReportSource] = useState('');
  const [clinicalReportGeneratedAt, setClinicalReportGeneratedAt] = useState('');

  const metrics = useMemo(() => overview?.metrics || [], [overview]);
  const chats = useMemo(() => overview?.chats || [], [overview]);
  const timelineEvents = useMemo(() => overview?.timeline_events || [], [overview]);
  const protocolFidelity = useMemo(() => overview?.protocol_fidelity || [], [overview]);
  const questionnaireResults = useMemo(() => overview?.questionnaire_results || [], [overview]);
  const topEmotions = useMemo(() => overview?.top_emotions || [], [overview]);
  const profile = useMemo(() => overview?.profile || {}, [overview]);
  const clinical = useMemo(() => overview?.clinical_parameters || {}, [overview]);
  
  const displayedChats = useMemo(() => chats.slice(0, chatVisibleCount), [chats, chatVisibleCount]);

  const scoreRows = useMemo(() => {
    if (!profile?.latest_scores) return [];
    return Object.entries(profile.latest_scores).map(([type, score]) => ({
      type,
      score,
      severity: profile?.latest_severity?.[type] || 'unknown',
      trend: profile?.screening_trends?.[type] || 'insufficient_data',
    }));
  }, [profile]);

  const targetUserId = useMemo(() => {
    const raw = overview?.user_id ?? profile?.user_id;
    return Number.isFinite(Number(raw)) ? Number(raw) : null;
  }, [overview, profile]);

  const loadOverview = async (targetLimit = limit) => {
    const finalLimit = clamp(Number(targetLimit) || 300, 20, 3000);
    setErrorMessage('');
    setChatVisibleCount(30);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/admin/overview`, {
        params: { limit: finalLimit, username: user },
      });
      setOverview(response.data || null);
    } catch (error) {
      setErrorMessage("Data synchronization failed. Check backend connectivity.");
    }
  };

  const loadClinicalReport = async () => {
    setClinicalReportError('');
    setIsClinicalReportLoading(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/admin/clinical-report`, {
        params: { username: user },
      });
      setClinicalReport(response.data?.summary || '');
      setClinicalReportSource(response.data?.summary_source || 'fallback');
      setClinicalReportGeneratedAt(response.data?.generated_at || '');
    } catch (error) {
      setClinicalReportError("Clinical synthesis timed out. You may manually retry formulation.");
    } finally {
      setIsClinicalReportLoading(false);
    }
  };

  useEffect(() => {
    loadOverview(limit);
    loadClinicalReport();
  }, []);

  const handleGlobalRefresh = () => {
    loadOverview(limit);
    loadClinicalReport();
  };

  const exportClinicalHandoff = async () => {
    setHandoffError('');
    if (!targetUserId) return;
    setIsExporting(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/admin/handoff/${targetUserId}`);
      const blob = new Blob([response.data.markdown], { type: 'text/markdown' });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `${user}_clinical_SBAR_handoff.md`;
      link.click();
    } catch (error) {
      setHandoffError("SBAR Export failed.");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500/30">
      {/* PROFESSIONAL CLINICAL HEADER */}
      <nav className="border-b border-slate-800 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Link to="/dashboard" className="text-slate-400 hover:text-white transition-colors p-2 hover:bg-slate-900 rounded-lg">
            <ArrowLeft size={20} />
          </Link>
          <div className="h-8 w-px bg-slate-800 mx-1 hidden md:block" />
          <h1 className="text-xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
            <Activity className="text-emerald-500" size={22} /> SERENITY <span className="text-slate-500 font-normal">Case Observatory</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Authorized Clinician</p>
            <p className="text-sm font-semibold text-slate-200">{user}</p>
          </div>
          <button onClick={onLogout} className="text-rose-400 text-xs font-black uppercase tracking-widest border border-rose-900/40 px-3 py-2 rounded-xl hover:bg-rose-950/30 transition-all flex items-center gap-2">
            <LogOut size={14} /> End Session
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6 max-w-[1600px] mx-auto w-full space-y-6">
        
        {/* TARASOFF / DUTY TO WARN PULSING ALERT */}
        {profile?.duty_to_warn && (
          <div className="p-5 bg-rose-950/30 border-2 border-rose-600 rounded-2xl flex items-center gap-5 animate-pulse shadow-[0_0_40px_rgba(225,29,72,0.15)] ring-1 ring-rose-500/50">
            <div className="bg-rose-600 p-3 rounded-full text-white shadow-lg">
              <AlertTriangle size={32} />
            </div>
            <div>
              <h3 className="text-rose-100 font-black text-lg uppercase tracking-tighter">Urgent Alert: Duty to Warn (Tarasoff Rule)</h3>
              <p className="text-rose-200/90 text-sm font-medium">
                Autonomous acuity monitoring has flagged language indicating potential imminent harm to others. 
                Perform an immediate manual review of encounter transcripts and follow legal reporting requirements.
              </p>
            </div>
          </div>
        )}

        <div className="grid grid-cols-12 gap-6">
          
          {/* PRIMARY CLINICAL ANALYSIS COLUMN (LEFT: 8 Columns) */}
          <div className="col-span-12 lg:col-span-8 space-y-6">
            
            {/* DYNAMIC CASE FORMULATION */}
            <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm overflow-hidden relative">
              <div className="flex justify-between items-start mb-6">
                <div className="flex items-center gap-3">
                  <div className="p-2 bg-indigo-500/20 rounded-xl text-indigo-400">
                    <Stethoscope size={20} />
                  </div>
                  <div>
                    <h2 className="text-lg font-bold text-slate-100">Clinical Case Formulation</h2>
                    <p className="text-[10px] text-slate-500 mt-1 uppercase tracking-widest font-black">
                      Synthesis Engine: {clinicalReportSource === 'cloud_llm' ? 'Qwen 2.5 Inference' : 'Heuristic Fallback'} • {formatDate(clinicalReportGeneratedAt || overview?.generated_at)}
                    </p>
                  </div>
                </div>
                <div className="flex gap-2">
                   <button onClick={loadClinicalReport} title="Refresh Synthesis" className="p-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl transition-colors text-slate-400">
                    <RefreshCw size={16} className={isClinicalReportLoading ? 'animate-spin' : ''} />
                  </button>
                  <button onClick={exportClinicalHandoff} disabled={isExporting || !targetUserId} className="flex items-center gap-2 px-5 py-2.5 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 rounded-xl text-xs font-black uppercase tracking-widest transition-all shadow-lg shadow-indigo-900/20">
                    {isExporting ? <Loader2 size={14} className="animate-spin" /> : <Download size={14} />}
                    Export SBAR
                  </button>
                </div>
              </div>

              <div className={`prose prose-invert max-w-none p-6 rounded-2xl border border-slate-800/50 bg-slate-950/40 leading-relaxed text-slate-300 transition-all ${isClinicalReportLoading ? 'blur-sm grayscale opacity-50' : ''}`}>
                {clinicalReport ? (
                  <ReactMarkdown>{clinicalReport}</ReactMarkdown>
                ) : (
                  <div className="flex flex-col items-center py-10 text-slate-500 italic space-y-3">
                     <p>Formulation synthesis required. Cloud LLM may be initializing.</p>
                     <button onClick={loadClinicalReport} className="text-xs text-indigo-400 underline uppercase tracking-widest font-black">Retry Formulation</button>
                  </div>
                )}
              </div>

              {isClinicalReportLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-slate-950/10 backdrop-blur-[2px] z-10">
                  <div className="flex flex-col items-center gap-3">
                    <Loader2 className="animate-spin text-indigo-500" size={36} />
                    <p className="text-xs font-bold text-indigo-400 uppercase tracking-widest animate-pulse">Synthesizing clinical context...</p>
                  </div>
                </div>
              )}
              
              {clinicalReportError && !clinicalReport && (
                <div className="mt-4 p-3 bg-amber-950/20 border border-amber-900/50 rounded-xl text-amber-200 text-xs flex items-center gap-2">
                  <AlertTriangle size={14} /> {clinicalReportError}
                </div>
              )}
            </section>

            {/* MOVED: FRAMEWORK ENGAGEMENT FIDELITY */}
            <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 mb-5">Clinical Framework Engagement</h3>
              <div className="space-y-6">
                {protocolFidelity.length > 0 ? protocolFidelity.map((item, idx) => (
                  <div key={idx} className="space-y-2">
                    <div className="flex justify-between text-[11px] font-black uppercase tracking-tight">
                      <span className="text-slate-400">{item.label}</span>
                      <span className="text-indigo-400">{item.share}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800/50 rounded-full overflow-hidden">
                      <div 
                        className={`h-full rounded-full transition-all duration-1000 ${
                          item.tone === 'rose' ? 'bg-rose-500 shadow-[0_0_10px_rgba(244,63,94,0.3)]' : 
                          item.tone === 'amber' ? 'bg-amber-500 shadow-[0_0_10px_rgba(245,158,11,0.3)]' : 
                          item.tone === 'emerald' ? 'bg-emerald-500' :
                          'bg-indigo-500 shadow-[0_0_10px_rgba(99,102,241,0.3)]'
                        }`} 
                        style={{ width: `${item.share}%` }} 
                      />
                    </div>
                  </div>
                )) : <p className="text-slate-600 text-xs italic text-center">No routing activity captured.</p>}
              </div>
            </section>

            {/* ENCOUNTER TRANSCRIPTS */}
            <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500">Therapeutic Encounter Transcripts</h3>
                <span className="text-[10px] text-slate-600 font-bold uppercase">{chats?.length || 0} Total Turns</span>
              </div>
              <div className="space-y-4 max-h-[550px] overflow-y-auto pr-4 custom-scrollbar">
                {displayedChats.length > 0 ? displayedChats.map((chat, idx) => (
                  <div key={idx} className="p-5 rounded-2xl bg-slate-950/50 border border-slate-800/50 space-y-3 hover:border-slate-700/50 transition-colors">
                    <div className="flex justify-between text-[10px] uppercase font-black tracking-widest text-slate-500">
                      <span>{formatDate(chat.timestamp)}</span>
                      <span className={`px-2 py-0.5 rounded-lg border ${chat.dominant_emotion === 'Neutral' ? 'border-slate-800' : 'border-indigo-900/50 text-indigo-400 bg-indigo-950/20'}`}>
                        Clinical Affect: {chat.dominant_emotion}
                      </span>
                    </div>
                    <div className="text-sm leading-relaxed">
                      <p className="flex gap-3"><span className="text-emerald-500 font-black shrink-0">CLIENT:</span> <span className="text-slate-200">{chat.user_text}</span></p>
                      <p className="mt-3 flex gap-3"><span className="text-slate-500 font-black shrink-0 uppercase tracking-tighter">AI AGENT:</span> <span className="text-slate-400">{chat.assistant_text}</span></p>
                    </div>
                  </div>
                )) : <p className="text-slate-600 text-sm italic text-center py-10">No recent therapeutic encounters recorded in this window.</p>}
                
                {chats.length > chatVisibleCount && (
                   <button onClick={() => setChatVisibleCount(prev => prev + 30)} className="w-full py-4 text-xs font-black uppercase tracking-widest text-indigo-400 hover:text-indigo-300 transition-colors bg-slate-950/30 rounded-xl border border-slate-800/50 hover:bg-slate-900">
                     Load Historical Logs ({chats.length - chatVisibleCount} remaining)
                   </button>
                )}
              </div>
            </section>
          </div>

          {/* QUANTITATIVE INDICES COLUMN (RIGHT: 4 Columns) */}
          <div className="col-span-12 lg:col-span-4 space-y-6">
            
            {/* ACUITY INDICES */}
            <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 mb-5">Current Acuity Indices</h3>
              <div className="grid grid-cols-2 gap-4">
                {metrics.map((m, i) => (
                  <div key={i} className="p-4 rounded-2xl bg-slate-950/50 border border-slate-800/50 hover:bg-slate-900/50 transition-colors">
                    <p className="text-[10px] uppercase font-black text-slate-500 tracking-tighter flex items-center gap-2">
                       <MetricIcon id={m.id} /> {m.label || m.id}
                    </p>
                    <p className="text-2xl font-black text-emerald-400 mt-1">{m.value}</p>
                    {m.delta !== undefined && m.delta !== null && m.id !== 'care_plan_adherence' && (
                      <div className={`mt-2 inline-flex items-center text-[10px] font-black px-2 py-0.5 rounded-lg ${m.delta > 0 ? 'bg-rose-950/50 text-rose-400 border border-rose-900/50' : 'bg-emerald-950/50 text-emerald-400 border border-emerald-900/50'}`}>
                        {m.delta > 0 ? '↑' : '↓'} {Math.abs(m.delta)} {m.delta_label || 'Shift'}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>

             {/* MEASUREMENT-BASED SCREENING (PHQ/GAD/PCL) */}
             <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 mb-5">Screening Metrics (MBC)</h3>
              <div className="space-y-3 max-h-[260px] overflow-y-auto pr-2 custom-scrollbar">
                {scoreRows.length > 0 ? scoreRows.map((row) => (
                  <div key={row.type} className="p-4 rounded-2xl bg-slate-950/50 border border-slate-800/50 flex justify-between items-center group hover:border-slate-700 transition-colors">
                    <div>
                      <p className="text-xs font-black text-slate-100 group-hover:text-indigo-400 transition-colors">{row.type}</p>
                      <p className={`text-[10px] font-bold uppercase mt-1 px-2 py-0.5 rounded-md inline-block ${acuityTone(row.trend)}`}>{row.severity} • {row.trend}</p>
                    </div>
                    <p className="text-2xl font-black text-slate-300 group-hover:text-white transition-colors">{row.score}</p>
                  </div>
                )) : <p className="text-slate-600 text-xs italic text-center py-4">No measurement-based data available.</p>}
              </div>
            </section>

             {/* AFFECTIVE PRESENTATION (Emotion Distribution) */}
             <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 mb-5">Affective Presentation</h3>
              <div className="space-y-3 max-h-[220px] overflow-y-auto pr-2 custom-scrollbar">
                {topEmotions.length > 0 ? topEmotions.slice(0, 6).map((item) => (
                  <div key={item.emotion} className="p-3 rounded-xl bg-slate-950/50 border border-slate-800/50 flex items-center justify-between">
                    <span className="text-slate-300 text-sm font-semibold capitalize">{item.emotion}</span>
                    <span className="text-indigo-400 font-black">{item.count}</span>
                  </div>
                )) : <p className="text-slate-600 text-xs italic text-center py-4">No affective data captured.</p>}
              </div>
            </section>

            {/* LONGITUDINAL CLINICAL TIMELINE */}
            <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6">
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-slate-500 mb-5">Clinical Activity History</h3>
              <div className="space-y-6 border-l-2 border-slate-800 ml-2 pl-6 relative max-h-[400px] overflow-y-auto custom-scrollbar">
                {timelineEvents.length > 0 ? timelineEvents.slice(0, 10).map((ev, i) => (
                  <div key={i} className="relative">
                    <div className={`absolute -left-[32px] top-1 flex h-6 w-6 items-center justify-center rounded-full border-2 border-slate-950 shadow-sm ${
                       ev.kind === 'safety' ? 'bg-rose-500 shadow-rose-900/50 text-white' : 'bg-slate-800 text-indigo-400'
                    }`}>
                      <div className="scale-[0.6]">
                         {ev.kind === 'safety' ? <AlertTriangle /> : <Clock3 />}
                      </div>
                    </div>
                    <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">{formatDate(ev.timestamp).split(',')[0]}</p>
                    <h4 className="text-xs font-bold text-slate-200 mt-0.5 leading-tight">{ev.title}</h4>
                    <p className="text-[11px] text-slate-400 mt-1 leading-snug line-clamp-2">{ev.detail}</p>
                  </div>
                )) : <p className="text-slate-600 text-xs italic">No longitudinal activity available.</p>}
              </div>
            </section>

          </div>
        </div>
      </main>
    </div>
  );
};

export default AdminPage;
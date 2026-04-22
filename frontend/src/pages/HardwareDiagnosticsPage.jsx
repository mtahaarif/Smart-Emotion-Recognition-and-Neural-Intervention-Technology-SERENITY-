import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  ArrowLeft,
  Cpu,
  Gauge,
  LogOut,
  MemoryStick,
  RefreshCw,
  ShieldCheck,
  TerminalSquare
} from 'lucide-react';
import { Link } from 'react-router-dom';
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
const POLL_INTERVAL_MS = 2500;
const MAX_POINTS = 60;

const formatClock = (isoValue) => {
  const parsed = new Date(isoValue || Date.now());
  if (Number.isNaN(parsed.getTime())) return '--:--:--';
  return parsed.toLocaleTimeString('en-US', { 
    timeZone: 'Asia/Karachi',
    hour12: true 
  });
};

const toNumber = (value, fallback = 0) => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

const HardwareDiagnosticsPage = ({ user, onLogout }) => {
  const [currentMetrics, setCurrentMetrics] = useState(null);
  const [series, setSeries] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [errorMessage, setErrorMessage] = useState('');

  const pollMetrics = async (refreshTag = false) => {
    if (refreshTag) setIsRefreshing(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/diagnostics/metrics`, {
        params: user ? { username: user } : undefined,
      });
      const payload = response.data || {};

      const nextMetrics = {
        capturedAt: String(payload.captured_at || new Date().toISOString()),
        sttLatency: toNumber(payload.stt_latency_ms),
        serLatency: toNumber(payload.ser_latency_ms),
        ferLatency: toNumber(payload.fer_latency_ms),
        cpuThreadUsagePercent: toNumber(payload.cpu_thread_usage_percent),
        ramUsageMb: toNumber(payload.ram_usage_mb),
        xnnpackDelegateActive: Boolean(payload.xnnpack_delegate_active),
      };

      setCurrentMetrics(nextMetrics);
      setSeries((previous) => {
        const nextPoint = {
          time: formatClock(nextMetrics.capturedAt),
          stt: nextMetrics.sttLatency,
          ser: nextMetrics.serLatency,
          fer: nextMetrics.ferLatency,
          cpu: nextMetrics.cpuThreadUsagePercent,
          ram: nextMetrics.ramUsageMb,
        };
        return [...previous.slice(-(MAX_POINTS - 1)), nextPoint];
      });
      setErrorMessage('');
    } catch (error) {
      setErrorMessage("System telemetry unreachable. Verify hardware connection.");
    } finally {
      setIsLoading(false);
      setIsRefreshing(false);
    }
  };

  useEffect(() => {
    let active = true;
    let timerId = null;
    const runPoll = async () => { if (active) await pollMetrics(false); };
    runPoll();
    timerId = window.setInterval(runPoll, POLL_INTERVAL_MS);
    return () => { active = false; if (timerId) window.clearInterval(timerId); };
  }, [user]);

  const statusTone = useMemo(() => {
    if (!currentMetrics) return 'bg-slate-800 text-slate-300 border-slate-700';
    if (currentMetrics.cpuThreadUsagePercent > 85) return 'bg-rose-950/50 text-rose-400 border-rose-900/50';
    if (currentMetrics.cpuThreadUsagePercent > 65) return 'bg-amber-950/50 text-amber-400 border-amber-900/50';
    return 'bg-emerald-950/30 text-emerald-400 border-emerald-900/50';
  }, [currentMetrics]);

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
            <TerminalSquare className="text-emerald-500" size={22} /> SERENITY <span className="text-slate-500 font-normal">Edge Diagnostics</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">System Monitor</p>
            <p className="text-sm font-semibold text-slate-200">{user}</p>
          </div>
          <button onClick={onLogout} className="text-rose-400 text-xs font-black uppercase tracking-widest border border-rose-900/40 px-3 py-2 rounded-xl hover:bg-rose-950/30 transition-all flex items-center gap-2">
            <LogOut size={14} /> End Session
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6 max-w-[1400px] mx-auto w-full space-y-6">
        
        {/* HEADER */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8 shadow-sm">
          <div className="flex flex-wrap items-center justify-between gap-4">
            <div>
              <p className="text-[10px] uppercase font-black tracking-[0.2em] text-indigo-400 mb-1">Pillar 5</p>
              <h1 className="text-2xl font-black tracking-tight text-slate-100">Live Edge Telemetry</h1>
              <p className="mt-1 text-sm text-slate-400">Real-time hardware utilization and deep-learning inference latencies.</p>
            </div>

            <div className="flex flex-col sm:flex-row items-center gap-4">
              <div className={`px-4 py-2 rounded-xl border text-xs font-black uppercase tracking-widest shadow-sm flex items-center gap-2 ${statusTone}`}>
                <div className={`w-2 h-2 rounded-full ${currentMetrics ? 'bg-emerald-400 animate-pulse' : 'bg-slate-500'}`} />
                {currentMetrics ? 'Link Active' : 'Awaiting Data'}
                <span className="text-slate-500 font-bold ml-2">[{currentMetrics ? formatClock(currentMetrics.capturedAt) : '--:--'}]</span>
              </div>
              <button onClick={() => pollMetrics(true)} className="p-2.5 bg-slate-800 hover:bg-slate-700 rounded-xl transition-colors text-slate-400">
                <RefreshCw size={16} className={isRefreshing ? 'animate-spin' : ''} />
              </button>
            </div>
          </div>
          
          {errorMessage && (
            <div className="mt-4 rounded-xl border border-rose-600/60 bg-rose-950/30 px-4 py-3 text-rose-200 text-sm inline-flex items-center gap-2">
              <AlertTriangle size={16} /> {errorMessage}
            </div>
          )}
        </section>

        {/* METRICS GRID */}
        <section className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-col justify-center shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-slate-500 mb-2">Speech (STT) Latency</p>
             <p className="text-3xl font-black text-cyan-400 flex items-center gap-3">
               <Gauge size={28} className="text-cyan-500/50" /> {currentMetrics ? `${currentMetrics.sttLatency.toFixed(0)} ms` : '--'}
             </p>
          </div>
          
          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-col justify-center shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-slate-500 mb-2">Thread Utilization</p>
             <p className="text-3xl font-black text-amber-400 flex items-center gap-3">
               <Cpu size={28} className="text-amber-500/50" /> {currentMetrics ? `${currentMetrics.cpuThreadUsagePercent.toFixed(1)}%` : '--'}
             </p>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-col justify-center shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-slate-500 mb-2">Memory (RAM) Load</p>
             <p className="text-3xl font-black text-fuchsia-400 flex items-center gap-3">
               <MemoryStick size={28} className="text-fuchsia-500/50" /> {currentMetrics ? `${currentMetrics.ramUsageMb.toFixed(0)} MB` : '--'}
             </p>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-col justify-center shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-slate-500 mb-2">XNNPACK Delegate</p>
             <p className={`text-2xl font-black flex items-center gap-3 ${currentMetrics?.xnnpackDelegateActive ? 'text-emerald-400' : 'text-slate-500'}`}>
               <ShieldCheck size={28} className={currentMetrics?.xnnpackDelegateActive ? 'text-emerald-500/50' : 'opacity-50'} /> 
               {currentMetrics?.xnnpackDelegateActive ? 'ACTIVE' : 'INACTIVE'}
             </p>
          </div>
        </section>

        {/* CHARTS */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
          <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8 shadow-sm">
            <h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-cyan-400 mb-6 flex items-center gap-2">
              <Activity size={14} /> Inference Latency Stream (ms)
            </h2>

            <div className="h-72 w-full">
              {isLoading ? (
                <div className="h-full flex items-center justify-center text-slate-500 font-black uppercase tracking-widest text-xs">Awaiting diagnostics stream...</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 6, right: 14, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} minTickGap={30} />
                    <YAxis stroke="#64748b" tick={{ fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: '1rem', color: '#f8fafc', boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)' }} itemStyle={{ fontWeight: 800, fontSize: '12px' }} labelStyle={{ color: '#64748b', fontSize: '10px', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '20px', fontSize: '11px', fontWeight: 800 }} />
                    <Line type="monotone" dataKey="stt" name="Speech (STT)" stroke="#22d3ee" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="ser" name="Emotion (SER)" stroke="#a78bfa" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    <Line type="monotone" dataKey="fer" name="Face (FER)" stroke="#f97316" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>

          <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8 shadow-sm">
            <h2 className="text-[10px] font-black uppercase tracking-[0.2em] text-amber-400 mb-6 flex items-center gap-2">
              <Cpu size={14} /> Hardware Load Trend
            </h2>
            <div className="h-72 w-full">
              {isLoading ? (
                <div className="h-full flex items-center justify-center text-slate-500 font-black uppercase tracking-widest text-xs">Awaiting diagnostics stream...</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={series} margin={{ top: 6, right: 14, left: 0, bottom: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" vertical={false} />
                    <XAxis dataKey="time" stroke="#64748b" tick={{ fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} minTickGap={30} />
                    <YAxis yAxisId="left" stroke="#64748b" tick={{ fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} />
                    <YAxis yAxisId="right" orientation="right" stroke="#64748b" tick={{ fontSize: 10, fontWeight: 600 }} tickLine={false} axisLine={false} />
                    <Tooltip contentStyle={{ background: '#020617', border: '1px solid #1e293b', borderRadius: '1rem', color: '#f8fafc', boxShadow: '0 10px 25px -5px rgba(0, 0, 0, 0.5)' }} itemStyle={{ fontWeight: 800, fontSize: '12px' }} labelStyle={{ color: '#64748b', fontSize: '10px', fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.1em', marginBottom: '8px' }} />
                    <Legend wrapperStyle={{ paddingTop: '20px', fontSize: '11px', fontWeight: 800 }} />
                    <Line yAxisId="left" type="monotone" dataKey="cpu" name="CPU Utilization (%)" stroke="#fbbf24" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                    <Line yAxisId="right" type="monotone" dataKey="ram" name="Memory (MB)" stroke="#f472b6" strokeWidth={2.5} dot={false} isAnimationActive={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </section>
        </div>

      </main>
    </div>
  );
};

export default HardwareDiagnosticsPage;
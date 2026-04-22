import React from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Activity, 
  ShieldAlert, 
  MessageSquare, 
  ClipboardList, 
  BrainCircuit, 
  Cpu, 
  LogOut 
} from 'lucide-react';

const Dashboard = ({ user, onLogout }) => {
  const navigate = useNavigate();

  const modules = [
    {
      title: 'Clinical Assessment',
      description: 'Complete standardized diagnostic tools including PHQ-9, GAD-7, and PCL-5 to establish baseline acuity.',
      icon: <ClipboardList size={36} className="text-indigo-400" />,
      path: '/questionnaires',
      glow: 'group-hover:shadow-[0_0_30px_rgba(99,102,241,0.2)] border-indigo-900/30 group-hover:border-indigo-500/50',
      iconBg: 'bg-indigo-950/40',
    },
    {
      title: 'Live Support Session',
      description: 'Engage in a real-time therapeutic dialogue with autonomous facial and vocal emotion tracking.',
      icon: <MessageSquare size={36} className="text-emerald-400" />,
      path: '/session',
      glow: 'group-hover:shadow-[0_0_30px_rgba(16,185,129,0.2)] border-emerald-900/30 group-hover:border-emerald-500/50',
      iconBg: 'bg-emerald-950/40',
    },
    {
      title: 'Measurement-Based Care',
      description: 'Track longitudinal symptom trajectories and receive dynamically adapted behavioral interventions.',
      icon: <Activity size={36} className="text-cyan-400" />,
      path: '/mbc-hub',
      glow: 'group-hover:shadow-[0_0_30px_rgba(6,182,212,0.2)] border-cyan-900/30 group-hover:border-cyan-500/50',
      iconBg: 'bg-cyan-950/40',
    },
    {
      title: 'Safety & Coping Protocol',
      description: 'Access deterministic crisis intervention steps, tactile grounding, and emergency escalation pathways.',
      icon: <ShieldAlert size={36} className="text-rose-400" />,
      path: '/safety',
      glow: 'group-hover:shadow-[0_0_30px_rgba(244,63,94,0.2)] border-rose-900/30 group-hover:border-rose-500/50',
      iconBg: 'bg-rose-950/40',
    },
    {
      title: 'Admin Observatory',
      description: 'Review structured case formulations, aggregate acuity indices, and export SBAR clinical handoffs.',
      icon: <BrainCircuit size={36} className="text-amber-400" />,
      path: '/admin',
      glow: 'group-hover:shadow-[0_0_30px_rgba(245,158,11,0.2)] border-amber-900/30 group-hover:border-amber-500/50',
      iconBg: 'bg-amber-950/40',
    },
    {
      title: 'Edge Diagnostics',
      description: 'Monitor live system telemetry, STT/SER/FER inference latencies, and hardware resource utilization.',
      icon: <Cpu size={36} className="text-slate-400" />,
      path: '/diagnostics',
      glow: 'group-hover:shadow-[0_0_30px_rgba(148,163,184,0.2)] border-slate-700/50 group-hover:border-slate-400/50',
      iconBg: 'bg-slate-800/50',
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500/30">
      {/* PROFESSIONAL NAV */}
      <nav className="border-b border-slate-800 px-8 py-5 flex justify-between items-center sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Activity className="text-emerald-500" size={26} />
          <h1 className="text-2xl font-black tracking-tight text-slate-100 uppercase">
            SERENITY <span className="text-slate-500 font-normal">Dashboard</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Active Profile</p>
            <p className="text-sm font-semibold text-slate-200">{user}</p>
          </div>
          <button onClick={onLogout} className="text-rose-400 text-xs font-black uppercase tracking-widest border border-rose-900/40 px-4 py-2.5 rounded-xl hover:bg-rose-950/30 transition-all flex items-center gap-2">
            <LogOut size={16} /> End Session
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6 md:p-10 max-w-[1600px] mx-auto w-full">
        {/* HEADER */}
        <div className="mb-10 text-center space-y-4 max-w-3xl mx-auto">
          <p className="text-[10px] uppercase font-black tracking-[0.3em] text-indigo-500">Clinical AI Interface</p>
          <h2 className="text-4xl md:text-5xl font-black text-white tracking-tight">Select a Workflow</h2>
          <p className="text-base text-slate-400 leading-relaxed">
            Welcome to the SERENITY system. Select a module below to initiate care delivery, run assessments, or review clinical telemetry.
          </p>
        </div>

        {/* LARGE CENTERED CARDS GRID */}
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-8">
          {modules.map((mod, idx) => (
            <button
              key={idx}
              onClick={() => navigate(mod.path)}
              className={`group relative flex flex-col items-center text-center p-10 bg-slate-900/40 rounded-[2rem] border transition-all duration-500 hover:-translate-y-2 ${mod.glow}`}
            >
              {/* ICON WRAPPER */}
              <div className={`mb-6 p-6 rounded-[1.5rem] border border-slate-700/50 shadow-inner transition-transform duration-500 group-hover:scale-110 ${mod.iconBg}`}>
                {mod.icon}
              </div>
              
              {/* TEXT */}
              <h3 className="text-xl font-black text-slate-100 mb-3 tracking-wide">{mod.title}</h3>
              <p className="text-sm text-slate-400 leading-relaxed">{mod.description}</p>
            </button>
          ))}
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
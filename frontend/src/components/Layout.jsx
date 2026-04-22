import React from 'react';
import {
  Activity,
  AlertTriangle,
  BarChart3,
  Brain,
  Cpu,
  Loader2,
  LogOut,
  ShieldCheck,
  Wifi,
  WifiOff,
} from 'lucide-react';
import { NavLink, Outlet } from 'react-router-dom';
import { useClinical } from '../context/ClinicalContext';

const NAV_ITEMS = [
  { path: '/mbc-hub', label: 'MBC Hub', icon: BarChart3, danger: false },
  { path: '/session', label: 'Live Session', icon: Brain, danger: false },
  { path: '/safety', label: 'Safety Protocol', icon: AlertTriangle, danger: true },
  { path: '/admin', label: 'Admin Observatory', icon: ShieldCheck, danger: false },
  { path: '/diagnostics', label: 'Diagnostics', icon: Cpu, danger: false },
];

const CONNECTION_META = {
  connected: {
    label: 'Connected',
    dotClass: 'bg-emerald-500',
    textClass: 'text-emerald-700',
    Icon: Wifi,
  },
  connecting: {
    label: 'Connecting',
    dotClass: 'bg-amber-500',
    textClass: 'text-amber-700',
    Icon: Loader2,
    spinning: true,
  },
  disconnected: {
    label: 'Disconnected',
    dotClass: 'bg-rose-500',
    textClass: 'text-rose-700',
    Icon: WifiOff,
  },
};

const navClassName = ({ isActive }, danger) => {
  const base =
    'group flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors border';

  if (danger) {
    if (isActive) {
      return `${base} border-rose-200 bg-rose-100 text-rose-800`;
    }
    return `${base} border-transparent text-rose-700 hover:bg-rose-50 hover:border-rose-200`;
  }

  if (isActive) {
    return `${base} border-sky-200 bg-sky-100 text-sky-900`;
  }

  return `${base} border-transparent text-slate-700 hover:bg-slate-100 hover:border-slate-200`;
};

const Layout = ({ user, onLogout }) => {
  const { activeRiskScore, connectionStatus, currentTherapyMode, isCrisisMode } = useClinical();
  const connection = CONNECTION_META[connectionStatus] || CONNECTION_META.disconnected;

  return (
    <div className="min-h-screen bg-slate-100 text-slate-900">
      <div className="flex min-h-screen">
        <aside className="hidden md:flex md:w-72 flex-col border-r border-slate-200 bg-white">
          <div className="px-5 py-5 border-b border-slate-200">
            <p className="text-xs uppercase tracking-[0.16em] text-sky-700">SERENITY</p>
            <h1 className="mt-1 text-xl font-semibold text-slate-900 flex items-center gap-2">
              <Activity size={18} className="text-sky-700" /> Clinical Console
            </h1>
            <p className="mt-2 text-xs text-slate-500">Edge Autonomous Clinical Support</p>
          </div>

          <nav className="px-3 py-4 space-y-2">
            {NAV_ITEMS.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink key={item.path} to={item.path} className={(state) => navClassName(state, item.danger)}>
                  <Icon size={16} className={item.danger ? 'text-rose-600' : 'text-slate-500'} />
                  <span>{item.label}</span>
                </NavLink>
              );
            })}
          </nav>

          <div className="mt-auto px-4 py-4 border-t border-slate-200">
            <button
              type="button"
              onClick={onLogout}
              className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-slate-900 text-white px-4 py-2.5 text-sm font-medium hover:bg-slate-800"
            >
              <LogOut size={15} /> Logout
            </button>
          </div>
        </aside>

        <div className="flex-1 min-w-0 flex flex-col">
          <header className="h-16 border-b border-slate-200 bg-white px-4 md:px-6 flex items-center justify-between gap-4">
            <div className="min-w-0">
              <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Patient</p>
              <p className="text-sm md:text-base font-medium text-slate-900 truncate">{user}</p>
            </div>

            <div className="flex items-center gap-4 md:gap-5">
              <div className="hidden sm:block text-right">
                <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Therapy Mode</p>
                <p className="text-sm font-medium text-slate-800">{currentTherapyMode}</p>
              </div>

              <div className="text-right">
                <p className="text-xs uppercase tracking-[0.14em] text-slate-500">Risk Score</p>
                <p className={`text-sm font-semibold ${isCrisisMode ? 'text-rose-700' : 'text-slate-800'}`}>
                  {activeRiskScore}
                </p>
              </div>

              <div className={`inline-flex items-center gap-2 rounded-full px-3 py-1.5 bg-slate-100 ${connection.textClass}`}>
                <span className={`h-2.5 w-2.5 rounded-full ${connection.dotClass}`} />
                <connection.Icon size={14} className={connection.spinning ? 'animate-spin' : ''} />
                <span className="text-xs font-semibold">{connection.label}</span>
              </div>
            </div>
          </header>

          <nav className="md:hidden border-b border-slate-200 bg-white px-3 py-3 overflow-x-auto">
            <div className="flex items-center gap-2 min-w-max">
              {NAV_ITEMS.map((item) => {
                const Icon = item.icon;
                return (
                  <NavLink
                    key={item.path}
                    to={item.path}
                    className={(state) =>
                      `${navClassName(state, item.danger)} whitespace-nowrap py-2 px-3 rounded-lg text-xs`
                    }
                  >
                    <Icon size={14} />
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}
            </div>
          </nav>

          <main className="flex-1 p-4 md:p-6 overflow-auto">
            <Outlet />
          </main>
        </div>
      </div>
    </div>
  );
};

export default Layout;

import React from 'react';
import { Activity, ClipboardList, LogOut, Radar, Shield } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

const Dashboard = ({ user, onLogout }) => {
  const navigate = useNavigate();

  const navCards = [
    {
      id: 'questionnaires',
      title: 'Questionnaires',
      subtitle: 'PHQ-9, GAD-7, PCL-5',
      description: 'Complete one assessment or all assessments and store dated results in your local database.',
      icon: ClipboardList,
      to: '/questionnaires',
      buttonClass: 'bg-indigo-700 hover:bg-indigo-600',
    },
    {
      id: 'live-emotion',
      title: 'Live Emotion Session',
      subtitle: 'Voice + visual support',
      description: 'Start a real-time support conversation with microphone, camera, and streaming response feedback.',
      icon: Radar,
      to: '/emotion/live',
      buttonClass: 'bg-cyan-700 hover:bg-cyan-600',
    },
    {
      id: 'admin',
      title: 'Admin Observatory',
      subtitle: 'Local analytics dashboard',
      description: 'Review chats, sessions, emotions, questionnaire outcomes, and mental health trend summaries.',
      icon: Shield,
      to: '/admin',
      buttonClass: 'bg-emerald-700 hover:bg-emerald-600',
    },
  ];

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <nav className="border-b border-cyan-900/50 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/95 backdrop-blur">
        <h1 className="text-2xl font-bold text-cyan-300 flex items-center gap-2">
          <Activity className="text-cyan-300" /> SERENITY Dashboard
        </h1>
        <div className="flex items-center gap-4">
          <span className="text-slate-400">Patient: <b>{user}</b></span>
          <button onClick={onLogout} className="text-rose-400 font-medium hover:text-rose-300 flex items-center gap-1">
            <LogOut size={18} /> Logout
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6">
        <div className="max-w-6xl mx-auto">
          <div className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-5 mb-4">
            <p className="text-xs uppercase tracking-wider text-cyan-300">Session Hub</p>
            <h2 className="text-3xl font-bold text-white mt-2">Welcome, {user}</h2>
            <p className="text-slate-300 mt-2 max-w-3xl">
              Choose where to continue: fill clinical questionnaires, launch a live support session, or monitor full local insights from the admin observatory.
            </p>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
            {navCards.map((card) => {
              const Icon = card.icon;
              return (
                <article
                  key={card.id}
                  className="rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-5 flex flex-col min-h-[280px]"
                >
                  <div className="inline-flex items-center justify-center w-11 h-11 rounded-lg bg-slate-950 border border-cyan-900/60 mb-4 text-cyan-300">
                    <Icon size={20} />
                  </div>
                  <h3 className="text-xl font-semibold text-white">{card.title}</h3>
                  <p className="text-cyan-300 text-sm mt-1">{card.subtitle}</p>
                  <p className="text-slate-300 text-sm mt-3 flex-1">{card.description}</p>

                  <button
                    type="button"
                    onClick={() => navigate(card.to)}
                    className={`mt-5 inline-flex items-center justify-center rounded-lg px-4 py-2 font-semibold ${card.buttonClass}`}
                  >
                    Open {card.title}
                  </button>
                </article>
              );
            })}
          </div>
        </div>
      </main>
    </div>
  );
};

export default Dashboard;
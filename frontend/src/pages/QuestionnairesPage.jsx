import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { 
  Activity, 
  ArrowLeft, 
  CheckCircle2, 
  Loader2, 
  LogOut, 
  Save, 
  ClipboardList, 
  AlertTriangle,
  RefreshCw // FIXED: Added the missing import that caused the blank screen!
} from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const normalizeHistoryDate = (value) => {
  if (!value) return 'N/A';
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return parsed.toLocaleString('en-US', { 
    timeZone: 'Asia/Karachi' 
  });
};

const QuestionnairesPage = ({ user, onLogout }) => {
  // Allow the MBC Hub to pass in a specific questionnaire to pre-select
  const location = useLocation();
  const preSelectedType = location.state?.questionnaireType;

  const [templates, setTemplates] = useState([]);
  const [selectedTypes, setSelectedTypes] = useState(preSelectedType ? [preSelectedType] : ['PHQ-9', 'GAD-7', 'PCL-5']);
  const [answersByType, setAnswersByType] = useState({});
  const [history, setHistory] = useState([]);
  const [isLoadingTemplates, setIsLoadingTemplates] = useState(true);
  const [isLoadingHistory, setIsLoadingHistory] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [statusMessage, setStatusMessage] = useState('');
  const [errorMessage, setErrorMessage] = useState('');

  const selectedTemplates = useMemo(
    () => templates.filter((item) => selectedTypes.includes(item.type)),
    [templates, selectedTypes]
  );

  const refreshHistory = async () => {
    setIsLoadingHistory(true);
    try {
      const response = await axios.get(`${API_BASE_URL}/api/questionnaires/history`, {
        params: { username: user, limit: 120 },
      });
      setHistory(Array.isArray(response.data?.results) ? response.data.results : []);
    } catch (error) {
      setErrorMessage("Failed to load submission history.");
    } finally {
      setIsLoadingHistory(false);
    }
  };

  useEffect(() => {
    let isMounted = true;
    const loadTemplates = async () => {
      setIsLoadingTemplates(true);
      try {
        const response = await axios.get(`${API_BASE_URL}/api/questionnaires/templates`);
        if (isMounted) {
          setTemplates(Array.isArray(response.data?.questionnaires) ? response.data.questionnaires : []);
          setStatusMessage('Diagnostic tools initialized and ready.');
        }
      } catch (error) {
        if (isMounted) setErrorMessage("Failed to load clinical templates.");
      } finally {
        if (isMounted) setIsLoadingTemplates(false);
      }
    };

    loadTemplates();
    refreshHistory();
    return () => { isMounted = false; };
  }, [user]);

  const toggleQuestionnaire = (type) => {
    setSelectedTypes((prev) => {
      if (prev.includes(type)) return prev.length === 1 ? prev : prev.filter((item) => item !== type);
      return [...prev, type];
    });
  };

  const setAnswer = (type, questionId, value) => {
    setAnswersByType((prev) => ({
      ...prev,
      [type]: { ...(prev[type] || {}), [questionId]: Number(value) },
    }));
  };

  const validateAnswers = () => {
    if (selectedTemplates.length === 0) return 'Select at least one assessment.';
    for (const q of selectedTemplates) {
      const missing = (q.questions || []).find((item) => (answersByType[q.type] || {})[item.id] === undefined);
      if (missing) return `Please complete all items in ${q.type} before saving.`;
    }
    return '';
  };

  const submitSelectedQuestionnaires = async () => {
    setStatusMessage(''); setErrorMessage('');
    const validationError = validateAnswers();
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setIsSubmitting(true);
    try {
      for (const q of selectedTemplates) {
        const orderedAnswers = (q.questions || []).map((question) => Number((answersByType[q.type] || {})[question.id] ?? 0));
        await axios.post(`${API_BASE_URL}/api/questionnaires/submit`, {
          username: user, questionnaire_type: q.type, answers: orderedAnswers, submitted_at: new Date().toISOString(),
        });
      }
      setStatusMessage('Clinical assessment securely logged and evaluated.');
      setAnswersByType({});
      await refreshHistory();
    } catch (error) {
      setErrorMessage("Failed to securely save assessment results.");
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans antialiased selection:bg-indigo-500/30">
      
      {/* PROFESSIONAL NAV */}
      <nav className="border-b border-slate-800 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/90 backdrop-blur-md">
        <div className="flex items-center gap-4">
          <Link to="/mbc-hub" className="text-slate-400 hover:text-white transition-colors p-2 hover:bg-slate-900 rounded-lg">
            <ArrowLeft size={20} />
          </Link>
          <div className="h-8 w-px bg-slate-800 mx-1 hidden md:block" />
          <h1 className="text-xl font-bold tracking-tight text-slate-100 flex items-center gap-2">
            <ClipboardList className="text-emerald-500" size={22} /> SERENITY <span className="text-slate-500 font-normal">Clinical Assessment</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Active Patient</p>
            <p className="text-sm font-semibold text-slate-200">{user}</p>
          </div>
          <button onClick={onLogout} className="text-rose-400 text-xs font-black uppercase tracking-widest border border-rose-900/40 px-3 py-2 rounded-xl hover:bg-rose-950/30 transition-all flex items-center gap-2">
            <LogOut size={14} /> End Session
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6 max-w-[1400px] mx-auto w-full">
        <div className="grid grid-cols-1 xl:grid-cols-12 gap-6">
          
          {/* LEFT COLUMN: ASSESSMENTS */}
          <section className="xl:col-span-8 space-y-6">
            
            {/* TOOL SELECTION HEADER */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8 shadow-sm">
              <p className="text-[10px] uppercase font-black tracking-[0.2em] text-indigo-400 mb-5">Select Diagnostic Tools</p>
              
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                {['PHQ-9', 'GAD-7', 'PCL-5'].map((type) => {
                  const selected = selectedTypes.includes(type);
                  return (
                    <button 
                      key={type} 
                      onClick={() => toggleQuestionnaire(type)} 
                      className={`rounded-2xl border px-5 py-4 text-left transition-all duration-300 ${
                        selected 
                          ? 'border-indigo-500 bg-indigo-900/20 text-indigo-200 shadow-[0_0_15px_rgba(99,102,241,0.1)]' 
                          : 'border-slate-800 bg-slate-950/50 text-slate-400 hover:border-slate-700 hover:bg-slate-900'
                      }`}
                    >
                      <p className={`font-black text-lg ${selected ? 'text-indigo-300' : 'text-slate-300'}`}>{type}</p>
                      <p className="text-[10px] font-bold uppercase tracking-widest mt-2">
                        {selected ? 'Active' : 'Tap to include'}
                      </p>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* ASSESSMENT FORMS */}
            {isLoadingTemplates ? (
              <div className="h-40 rounded-[2rem] border border-slate-800 bg-slate-950/50 flex flex-col items-center justify-center text-slate-400 gap-3">
                <Loader2 size={24} className="animate-spin text-indigo-500" />
                <span className="text-xs font-black uppercase tracking-widest">Loading diagnostic templates...</span>
              </div>
            ) : (
              <div className="space-y-6">
                {selectedTemplates.map((q) => {
                  const responseMap = answersByType[q.type] || {};
                  return (
                    <div key={q.type} className="rounded-[2rem] border border-slate-800 bg-slate-900/40 p-6 lg:p-8">
                      <h3 className="text-xl font-black tracking-tight text-white">{q.title}</h3>
                      <p className="text-slate-400 text-sm mt-2 mb-8 leading-relaxed">{q.description}</p>

                      <div className="space-y-6">
                        {(q.questions || []).map((question) => (
                          <div key={question.id} className="rounded-2xl border border-slate-800/50 bg-slate-950/50 p-5 hover:border-slate-700 transition-colors">
                            <p className="text-sm font-medium mb-4 leading-relaxed text-slate-200">
                              <span className="text-indigo-400 font-black mr-2">Q{question.id}.</span> {question.text}
                            </p>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                              {(q.options || []).map((opt) => {
                                const selected = responseMap[question.id] === opt.value;
                                return (
                                  <button 
                                    key={opt.value} 
                                    onClick={() => setAnswer(q.type, question.id, opt.value)} 
                                    className={`rounded-xl border px-3 py-3 text-xs font-bold transition-all duration-300 ${
                                      selected 
                                        ? 'border-emerald-500/50 bg-emerald-900/20 text-emerald-300 shadow-[0_0_10px_rgba(16,185,129,0.1)]' 
                                        : 'border-slate-800 bg-slate-900/40 text-slate-400 hover:border-slate-600 hover:bg-slate-800 hover:text-slate-200'
                                    }`}
                                  >
                                    {opt.label}
                                  </button>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>
                  );
                })}
              </div>
            )}

            {/* ACTION FOOTER */}
            <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-wrap gap-4 items-center justify-between">
              <div className="flex gap-3">
                <button 
                  onClick={submitSelectedQuestionnaires} 
                  disabled={isSubmitting || isLoadingTemplates || selectedTemplates.length === 0} 
                  className="inline-flex items-center gap-2 rounded-xl bg-indigo-600 px-6 py-3 text-xs font-black uppercase tracking-widest text-white hover:bg-indigo-500 disabled:opacity-50 transition-all shadow-lg shadow-indigo-900/20"
                >
                  {isSubmitting ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />} 
                  Save Results
                </button>
                <button 
                  onClick={refreshHistory} 
                  disabled={isLoadingHistory} 
                  className="inline-flex items-center gap-2 rounded-xl bg-slate-800 border border-slate-700 px-6 py-3 text-xs font-black uppercase tracking-widest text-slate-300 hover:bg-slate-700 disabled:opacity-50 transition-all"
                >
                  <RefreshCw size={16} className={isLoadingHistory ? "animate-spin" : ""} /> Refresh
                </button>
              </div>

              {statusMessage && <p className="text-xs font-black uppercase tracking-widest text-emerald-400 flex items-center gap-2"><CheckCircle2 size={14}/> {statusMessage}</p>}
              {errorMessage && <p className="text-xs font-black uppercase tracking-widest text-rose-400 flex items-center gap-2"><AlertTriangle size={14}/> {errorMessage}</p>}
            </div>
          </section>

          {/* RIGHT COLUMN: HISTORY */}
          <section className="xl:col-span-4 bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8 h-fit sticky top-24">
            <div className="flex justify-between items-center mb-6">
              <h2 className="text-[10px] uppercase font-black tracking-[0.2em] text-cyan-400">Longitudinal Submissions</h2>
              <button onClick={refreshHistory} disabled={isLoadingHistory} className="text-slate-500 hover:text-slate-300 transition-colors">
                <RefreshCw size={14} className={isLoadingHistory ? "animate-spin" : ""} />
              </button>
            </div>
            
            <div className="space-y-4 max-h-[60vh] overflow-y-auto pr-2 custom-scrollbar">
              {isLoadingHistory ? (
                 <div className="py-10 flex justify-center"><Loader2 className="animate-spin text-indigo-500" size={24} /></div>
              ) : history.length === 0 ? (
                <p className="text-sm text-slate-500 italic text-center py-10">No clinical submissions recorded yet.</p>
              ) : (
                history.map((entry) => {
                  // FIXED: Safe lowercasing to prevent crashes on null severity
                  const severityStr = String(entry?.severity || '').toLowerCase();
                  const isSevere = severityStr.includes('severe') || severityStr.includes('high') || severityStr.includes('elevated');
                  
                  return (
                    <div key={entry.id} className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5 hover:border-slate-700 transition-colors">
                      <div className="flex justify-between items-start mb-2">
                        <p className="text-sm font-black text-slate-100">{entry.questionnaire_type}</p>
                        <p className="text-xl font-black text-indigo-400">{entry.total_score}</p>
                      </div>
                      <p className={`text-[10px] font-bold uppercase tracking-widest mb-3 ${isSevere ? 'text-rose-400' : 'text-emerald-400'}`}>
                        Severity: {entry.severity || 'Unknown'}
                      </p>
                      <p className="text-[10px] text-slate-500 font-bold uppercase tracking-widest border-t border-slate-800/50 pt-3">
                        {normalizeHistoryDate(entry.created_at)}
                      </p>
                    </div>
                  );
                })
              )}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default QuestionnairesPage;
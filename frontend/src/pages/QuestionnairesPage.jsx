import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { Activity, ArrowLeft, CheckCircle2, FileText, Loader2, LogOut, Save } from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const normalizeHistoryDate = (value) => {
  if (!value) {
    return 'Unknown date';
  }
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) {
    return String(value);
  }
  return parsed.toLocaleString();
};

const QuestionnairesPage = ({ user, onLogout }) => {
  const [templates, setTemplates] = useState([]);
  const [selectedTypes, setSelectedTypes] = useState(['PHQ-9', 'GAD-7', 'PCL-5']);
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
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to load questionnaire history.';
      setErrorMessage(backendMessage);
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
        const rows = Array.isArray(response.data?.questionnaires) ? response.data.questionnaires : [];
        if (isMounted) {
          setTemplates(rows);
          setStatusMessage('Questionnaires are ready. You can submit one, two, or all three.');
        }
      } catch (error) {
        const backendMessage =
          error.response?.data?.detail ||
          error.response?.data?.error ||
          'Failed to load questionnaire templates.';
        if (isMounted) {
          setErrorMessage(backendMessage);
        }
      } finally {
        if (isMounted) {
          setIsLoadingTemplates(false);
        }
      }
    };

    loadTemplates();
    refreshHistory();

    return () => {
      isMounted = false;
    };
  }, [user]);

  const toggleQuestionnaire = (questionnaireType) => {
    setSelectedTypes((prev) => {
      if (prev.includes(questionnaireType)) {
        if (prev.length === 1) {
          return prev;
        }
        return prev.filter((item) => item !== questionnaireType);
      }
      return [...prev, questionnaireType];
    });
  };

  const setAnswer = (questionnaireType, questionId, value) => {
    setAnswersByType((prev) => {
      const current = prev[questionnaireType] || {};
      return {
        ...prev,
        [questionnaireType]: {
          ...current,
          [questionId]: Number(value),
        },
      };
    });
  };

  const validateAnswers = () => {
    if (selectedTemplates.length === 0) {
      return 'Select at least one questionnaire.';
    }

    for (const questionnaire of selectedTemplates) {
      const responseMap = answersByType[questionnaire.type] || {};
      const missingQuestion = questionnaire.questions.find((item) => responseMap[item.id] === undefined);
      if (missingQuestion) {
        return `Complete all questions in ${questionnaire.type} before submitting.`;
      }
    }

    return '';
  };

  const submitSelectedQuestionnaires = async () => {
    setStatusMessage('');
    setErrorMessage('');

    const validationError = validateAnswers();
    if (validationError) {
      setErrorMessage(validationError);
      return;
    }

    setIsSubmitting(true);
    try {
      for (const questionnaire of selectedTemplates) {
        const responseMap = answersByType[questionnaire.type] || {};
        const orderedAnswers = questionnaire.questions.map((question) => Number(responseMap[question.id] ?? 0));

        await axios.post(`${API_BASE_URL}/api/questionnaires/submit`, {
          username: user,
          questionnaire_type: questionnaire.type,
          answers: orderedAnswers,
          submitted_at: new Date().toISOString(),
        });
      }

      setStatusMessage('Questionnaire responses saved locally and linked to your profile.');
      setAnswersByType({});
      await refreshHistory();
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        'Failed to save questionnaire results.';
      setErrorMessage(backendMessage);
    } finally {
      setIsSubmitting(false);
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
            <FileText className="text-cyan-300" /> Questionnaire Center
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
        <div className="max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-12 gap-4">
          <section className="xl:col-span-8 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h2 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Select Questionnaires</h2>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-4">
              {['PHQ-9', 'GAD-7', 'PCL-5'].map((questionnaireType) => {
                const selected = selectedTypes.includes(questionnaireType);
                return (
                  <button
                    key={questionnaireType}
                    type="button"
                    onClick={() => toggleQuestionnaire(questionnaireType)}
                    className={`rounded-lg border px-3 py-2 text-left transition ${
                      selected
                        ? 'border-cyan-500 bg-cyan-900/40 text-cyan-200'
                        : 'border-cyan-900/50 bg-slate-950 text-slate-300 hover:border-cyan-700'
                    }`}
                  >
                    <p className="font-semibold">{questionnaireType}</p>
                    <p className="text-xs mt-1">{selected ? 'Selected' : 'Tap to include in this submission'}</p>
                  </button>
                );
              })}
            </div>

            {isLoadingTemplates ? (
              <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-5 text-slate-300 inline-flex items-center gap-2">
                <Loader2 className="animate-spin" size={16} /> Loading questionnaires...
              </div>
            ) : (
              <div className="space-y-6">
                {selectedTemplates.map((questionnaire) => {
                  const responseMap = answersByType[questionnaire.type] || {};
                  return (
                    <div key={questionnaire.type} className="rounded-xl border border-cyan-900/50 bg-slate-950/70 p-4">
                      <h3 className="text-lg text-cyan-300 font-semibold">{questionnaire.title}</h3>
                      <p className="text-slate-400 text-sm mb-4">{questionnaire.description}</p>

                      <div className="space-y-4">
                        {questionnaire.questions.map((question) => (
                          <div key={`${questionnaire.type}-${question.id}`} className="rounded-lg border border-cyan-900/40 bg-slate-900/60 p-3">
                            <p className="text-sm mb-2">
                              <span className="text-cyan-300 font-semibold">Q{question.id}.</span> {question.text}
                            </p>
                            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                              {questionnaire.options.map((option) => {
                                const selected = responseMap[question.id] === option.value;
                                return (
                                  <button
                                    key={`${questionnaire.type}-${question.id}-${option.value}`}
                                    type="button"
                                    onClick={() => setAnswer(questionnaire.type, question.id, option.value)}
                                    className={`rounded-md border px-2 py-2 text-xs text-left ${
                                      selected
                                        ? 'border-emerald-500 bg-emerald-900/30 text-emerald-200'
                                        : 'border-cyan-900/40 bg-slate-950 text-slate-300 hover:border-cyan-700'
                                    }`}
                                  >
                                    {option.label}
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

            <div className="mt-5 flex flex-wrap gap-3">
              <button
                type="button"
                onClick={submitSelectedQuestionnaires}
                disabled={isSubmitting || isLoadingTemplates}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 px-4 py-2 font-semibold disabled:opacity-60"
              >
                {isSubmitting ? <Loader2 className="animate-spin" size={16} /> : <Save size={16} />}
                Save Selected Questionnaires
              </button>

              <button
                type="button"
                onClick={refreshHistory}
                disabled={isLoadingHistory}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-700 hover:bg-cyan-600 px-4 py-2 font-semibold disabled:opacity-60"
              >
                <Activity size={16} /> Refresh History
              </button>
            </div>

            {statusMessage && (
              <div className="mt-4 rounded-lg border border-emerald-600/60 bg-emerald-950/30 p-3 text-emerald-200 text-sm inline-flex items-center gap-2">
                <CheckCircle2 size={16} /> {statusMessage}
              </div>
            )}

            {errorMessage && (
              <div className="mt-4 rounded-lg border border-rose-600/60 bg-rose-950/30 p-3 text-rose-200 text-sm">
                {errorMessage}
              </div>
            )}
          </section>

          <section className="xl:col-span-4 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h2 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Local History</h2>

            <div className="space-y-3 max-h-[72vh] overflow-auto pr-1">
              {isLoadingHistory && (
                <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3 text-slate-300 inline-flex items-center gap-2">
                  <Loader2 className="animate-spin" size={14} /> Fetching submissions...
                </div>
              )}

              {!isLoadingHistory && history.length === 0 && (
                <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3 text-slate-400">
                  No questionnaire submissions yet.
                </div>
              )}

              {history.map((entry) => (
                <div key={entry.id} className="rounded-lg border border-cyan-900/50 bg-slate-950/70 p-3">
                  <p className="text-cyan-300 font-semibold">{entry.questionnaire_type}</p>
                  <p className="text-sm text-slate-300">Score: {entry.total_score}</p>
                  <p className="text-sm text-emerald-300">Severity: {entry.severity}</p>
                  <p className="text-xs text-slate-500 mt-1">{normalizeHistoryDate(entry.created_at)}</p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default QuestionnairesPage;

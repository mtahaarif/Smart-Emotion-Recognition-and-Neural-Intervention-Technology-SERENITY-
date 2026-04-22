import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle,
  ArrowLeft,
  Download,
  HeartPulse,
  LifeBuoy,
  Loader2,
  LogOut,
  MapPin,
  Phone,
  RotateCcw,
  Share2,
  ShieldCheck,
  ShieldAlert,
  Volume2,
  VolumeX,
  Wind,
  CheckSquare,
  Stethoscope,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
const TRUSTED_CONTACT_PLACEHOLDER = '+920000000000';
const TOTAL_BREATH_CIRCUITS = 5;

const groundingSequence = [
  { id: 'see', count: 5, prompt: 'Tap one button for each of the 5 things you can see.', cue: 'Identify edges, colors, or specific objects in your immediate environment.' },
  { id: 'feel', count: 4, prompt: 'Tap one button for each of the 4 things you can feel.', cue: 'Focus on tactile sensations: pressure points, fabric texture, or temperature.' },
  { id: 'hear', count: 3, prompt: 'Tap one button for each of the 3 things you can hear.', cue: 'Isolate distinct auditory inputs, both near and far.' },
  { id: 'smell', count: 2, prompt: 'Tap one button for each of the 2 things you can smell.', cue: 'Notice ambient scents, fresh air, or clothing.' },
  { id: 'taste', count: 1, prompt: 'Tap when you identify 1 thing you can taste.', cue: 'Bring awareness to present taste or take a sip of water.' },
];

const createTapFlags = (count) => Array.from({ length: count }, () => false);
const buildMapLink = (latitude, longitude) => `https://maps.google.com/?q=${latitude},${longitude}`;

const SafetyPlanPage = ({ user, onLogout }) => {
  const navigate = useNavigate();
  const contactsRef = useRef(null);
  const [targetUserId, setTargetUserId] = useState(null);
  
  // Base State
  const [resolveError, setResolveError] = useState('');
  const [handoffError, setHandoffError] = useState('');
  const [shareNotice, setShareNotice] = useState('');
  const [locationNotice, setLocationNotice] = useState('');
  const [isResolvingUserId, setIsResolvingUserId] = useState(false);
  const [isDownloadingReport, setIsDownloadingReport] = useState(false);
  const [isSharingSos, setIsSharingSos] = useState(false);

  // Grounding State
  const [groundingStepIndex, setGroundingStepIndex] = useState(0);
  const [groundingTaps, setGroundingTaps] = useState(() => createTapFlags(groundingSequence[0].count));
  const [groundingComplete, setGroundingComplete] = useState(false);

  // Breathing State
  const [audioGuideEnabled, setAudioGuideEnabled] = useState(false);
  const [audioError, setAudioError] = useState('');
  const breathTimerRef = useRef(null);
  const [breathPhase, setBreathPhase] = useState('IDLE');
  const [breathCount, setBreathCount] = useState(0);

  // CALM Protocol State
  const [calmChecks, setCalmChecks] = useState([false, false, false]);
  const isEnvironmentSecured = calmChecks.every(Boolean);

  // C-SSRS Protocol State
  const [cssrsStep, setCssrsStep] = useState(0);
  const [riskLevel, setRiskLevel] = useState('Unassessed'); 

  const userKey = String(user || localStorage.getItem('serenity_user') || '').trim();

  useEffect(() => {
    const resolveTargetUserId = async () => {
      setResolveError('');
      if (!userKey) {
        setTargetUserId(null);
        setResolveError('No active user found. Sign in again to enable SOS report export.');
        return;
      }
      setIsResolvingUserId(true);
      try {
        const response = await axios.get(`${API_BASE_URL}/api/admin/overview`, { params: { username: userKey, limit: 20 } });
        const rawUserId = response.data?.user_id ?? response.data?.profile?.user_id;
        const parsedUserId = Number(rawUserId);
        if (!Number.isFinite(parsedUserId)) throw new Error('Unable to resolve user_id.');
        setTargetUserId(parsedUserId);
      } catch (error) {
        setResolveError('Failed to prepare SOS report export.');
        setTargetUserId(null);
      } finally {
        setIsResolvingUserId(false);
      }
    };
    resolveTargetUserId();
  }, [userKey]);

  // --- GROUNDING LOGIC ---
  const currentGroundingStep = useMemo(() => groundingSequence[Math.min(groundingStepIndex, groundingSequence.length - 1)], [groundingStepIndex]);
  const groundingTappedCount = useMemo(() => groundingTaps.filter(Boolean).length, [groundingTaps]);

  const handleGroundingTap = useCallback((tapIndex) => {
    setGroundingTaps((previous) => {
      if (previous[tapIndex]) return previous;
      const next = [...previous];
      next[tapIndex] = true;
      return next;
    });
  }, []);

  const resetGroundingStepper = useCallback(() => {
    setGroundingStepIndex(0);
    setGroundingComplete(false);
    setGroundingTaps(createTapFlags(groundingSequence[0].count));
  }, []);

  useEffect(() => {
    if (groundingComplete || groundingTaps.length === 0 || !groundingTaps.every(Boolean)) return;
    const nextStepIndex = groundingStepIndex + 1;
    const advanceTimer = window.setTimeout(() => {
      if (nextStepIndex >= groundingSequence.length) {
        setGroundingComplete(true);
        return;
      }
      setGroundingStepIndex(nextStepIndex);
      setGroundingTaps(createTapFlags(groundingSequence[nextStepIndex].count));
    }, 220);
    return () => window.clearTimeout(advanceTimer);
  }, [groundingComplete, groundingStepIndex, groundingTaps]);

  // --- BREATHING LOGIC ---
  const stopSpeechOutput = useCallback(() => {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) window.speechSynthesis.cancel();
  }, []);

  const stopAudioGuide = useCallback(() => {
    if (breathTimerRef.current) {
      window.clearTimeout(breathTimerRef.current);
      breathTimerRef.current = null;
    }
    stopSpeechOutput();
  }, [stopSpeechOutput]);

  useEffect(() => {
    if (!audioGuideEnabled) stopSpeechOutput();
  }, [audioGuideEnabled, stopSpeechOutput]);

  const speakAudioCue = useCallback((text) => {
    if (!audioGuideEnabled) return;
    if (typeof window === 'undefined' || !('speechSynthesis' in window)) {
      setAudioError('Audio guide is not supported.');
      setAudioGuideEnabled(false);
      return;
    }
    try {
      const synth = window.speechSynthesis;
      const utterance = new window.SpeechSynthesisUtterance(text);
      utterance.rate = 0.86;
      synth.cancel();
      synth.speak(utterance);
    } catch {
      setAudioError('Audio guide failed.');
      setAudioGuideEnabled(false);
    }
  }, [audioGuideEnabled]);

  useEffect(() => {
    if (breathPhase === 'IDLE' || breathPhase === 'COMPLETE') {
      if (breathTimerRef.current) window.clearTimeout(breathTimerRef.current);
      return;
    }

    if (breathPhase === 'INHALE') {
      speakAudioCue('Inhale. Two. Three. Four.');
      breathTimerRef.current = window.setTimeout(() => setBreathPhase('HOLD'), 4000);
    } else if (breathPhase === 'HOLD') {
      speakAudioCue('Hold...');
      breathTimerRef.current = window.setTimeout(() => setBreathPhase('EXHALE'), 4000);
    } else if (breathPhase === 'EXHALE') {
      speakAudioCue('Exhale...');
      breathTimerRef.current = window.setTimeout(() => {
        setBreathCount((prev) => {
          const completed = prev + 1;
          if (completed >= TOTAL_BREATH_CIRCUITS) {
            setBreathPhase('COMPLETE');
            return TOTAL_BREATH_CIRCUITS;
          }
          setBreathPhase('INHALE');
          return completed;
        });
      }, 6000);
    }
    return () => window.clearTimeout(breathTimerRef.current);
  }, [breathPhase, speakAudioCue]);

  useEffect(() => stopAudioGuide, [stopAudioGuide]);

  // --- SOS LOGIC ---
  const logCrisisEvent = async (severity) => {
    try {
      await axios.post(`${API_BASE_URL}/api/crisis/log`, { user_id: targetUserId, severity: severity });
    } catch (e) {
      console.error("Failed to log crisis timestamp", e);
    }
  };

  const fetchSosReportPayload = useCallback(async () => {
    if (!targetUserId) throw new Error('No user_id available yet.');
    const response = await axios.get(`${API_BASE_URL}/api/admin/handoff/${targetUserId}`);
    const markdown = String(response.data?.markdown || '').trim();
    if (!markdown) throw new Error('Empty handoff report.');
    return { blob: new Blob([`${markdown}\n`], { type: 'text/markdown;charset=utf-8' }), fileName: `${userKey}_clinical_handoff.md` };
  }, [targetUserId, userKey]);

  const resolveLocationLink = useCallback(() => {
    if (!navigator.geolocation) return Promise.resolve('');
    return new Promise((resolve) => {
      navigator.geolocation.getCurrentPosition(
        (position) => resolve(buildMapLink(position.coords.latitude, position.coords.longitude)),
        () => resolve(''),
        { enableHighAccuracy: true, timeout: 8000 }
      );
    });
  }, []);

  const shareSosReport = async () => {
    setHandoffError('');
    setIsSharingSos(true);
    try {
      await logCrisisEvent(riskLevel); 
      const { blob, fileName } = await fetchSosReportPayload();
      const locationLink = await resolveLocationLink();

      const emergencyText = locationLink
        ? `Emergency: I am in severe distress. My C-SSRS acute risk level is assessed as [${riskLevel.toUpperCase()}]. My location: ${locationLink}.`
        : `Emergency: I am in severe distress. My C-SSRS acute risk level is assessed as [${riskLevel.toUpperCase()}]. Location unavailable; please call me immediately.`;

      const smsHref = `sms:${TRUSTED_CONTACT_PLACEHOLDER}?body=${encodeURIComponent(`${emergencyText}\n\nPlease contact emergency services.`)}`;
      window.location.href = smsHref;
      setShareNotice('Opened SMS emergency draft.');
    } catch (error) {
      setHandoffError('Failed to prepare SOS sharing.');
    } finally {
      setIsSharingSos(false);
    }
  };

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
            <Activity className="text-emerald-500" size={22} /> SERENITY <span className="text-slate-500 font-normal">Active Safety & Coping Toolkit</span>
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

      <main className="flex-1 p-6 max-w-[1200px] mx-auto w-full space-y-6">
        
        {/* HEADER */}
        <section className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-8 shadow-sm">
          <div className="flex items-center gap-3 mb-2">
            <div className="p-2 bg-indigo-500/20 rounded-xl text-indigo-400">
               <Stethoscope size={24} />
            </div>
            <h1 className="text-2xl font-black text-slate-100 tracking-tight">Clinical Safety & Stabilization Plan</h1>
          </div>
          <p className="text-sm text-slate-400 mt-2 max-w-3xl leading-relaxed">
            This secure, structured environment provides evidence-based techniques to down-regulate physiological arousal, ensure immediate environmental safety, and facilitate professional handoff if required.
          </p>
        </section>

        {/* ROW 1: GROUNDING & CALM */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* TACTILE GROUNDING */}
          <article className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-col">
            <h3 className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400 mb-5 flex items-center gap-2">
              <ShieldCheck size={16} /> Tactile Grounding (5-4-3-2-1)
            </h3>
            <p className="text-sm text-slate-400 mb-4">
              Interrupt acute distress cycles through forced sensory engagement and motor focus.
            </p>
            
            <div className="mt-auto bg-slate-950/50 rounded-2xl border border-slate-800 p-5">
              {groundingComplete ? (
                <div className="py-4 text-center space-y-2">
                  <div className="mx-auto w-12 h-12 bg-emerald-500/20 rounded-full flex items-center justify-center text-emerald-400 mb-3">
                    <CheckSquare size={24} />
                  </div>
                  <p className="text-sm font-black text-emerald-400 uppercase tracking-widest">Grounding Complete</p>
                  <p className="text-xs text-slate-400">Maintain focus on your breath. Proceed to the next module.</p>
                </div>
              ) : (
                <>
                  <p className="text-[10px] font-black uppercase tracking-[0.15em] text-slate-500 mb-2">
                    Phase {groundingStepIndex + 1} of {groundingSequence.length}
                  </p>
                  <p className="text-base font-bold text-slate-200 mb-1">{currentGroundingStep.prompt}</p>
                  <p className="text-xs text-slate-500 mb-5">{currentGroundingStep.cue}</p>
                  <div className="flex flex-wrap gap-3">
                    {groundingTaps.map((isTapped, tapIndex) => (
                      <button
                        key={`tap-${tapIndex}`}
                        onClick={() => handleGroundingTap(tapIndex)}
                        disabled={isTapped}
                        className={`h-12 w-12 rounded-full border-2 text-sm font-black transition-all ${
                          isTapped 
                            ? 'border-emerald-500/50 bg-emerald-900/30 text-emerald-400 scale-95' 
                            : 'border-slate-700 bg-slate-800 text-slate-300 hover:bg-slate-700 hover:border-slate-500 active:scale-95'
                        }`}
                      >
                        {isTapped ? 'OK' : tapIndex + 1}
                      </button>
                    ))}
                  </div>
                </>
              )}
              <button onClick={resetGroundingStepper} className="mt-6 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-slate-500 hover:text-slate-300 transition-colors">
                <RotateCcw size={12} /> Reset Sequence
              </button>
            </div>
          </article>

          {/* CALM PROTOCOL */}
          <article className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 flex flex-col">
            <h3 className="text-xs font-black uppercase tracking-[0.2em] text-blue-400 mb-5 flex items-center gap-2">
              <LifeBuoy size={16} /> CALM: Environmental Safety
            </h3>
            <p className="text-sm text-slate-400 mb-4">
              Counseling on Access to Lethal Means (CALM). Physically confirm your environment is secure before proceeding.
            </p>
            
            <div className="space-y-3 mt-auto">
              {[
                "I have moved to a shared, public, or supervised room.",
                "I have secured or moved away from medications, sharp objects, or firearms.",
                "I am sitting in a grounded, secure position."
              ].map((text, idx) => (
                <label key={idx} className="flex items-start gap-4 p-4 rounded-xl border border-slate-800 bg-slate-950/50 cursor-pointer hover:border-slate-700 transition-colors">
                  <input 
                    type="checkbox" 
                    checked={calmChecks[idx]} 
                    onChange={() => {
                      const newChecks = [...calmChecks];
                      newChecks[idx] = !newChecks[idx];
                      setCalmChecks(newChecks);
                    }}
                    className="mt-0.5 h-5 w-5 rounded border-slate-600 text-emerald-500 focus:ring-emerald-500 focus:ring-offset-slate-900 bg-slate-900 transition-all cursor-pointer"
                  />
                  <span className="text-sm font-medium text-slate-300">{text}</span>
                </label>
              ))}
            </div>

            <div className="mt-5">
              {isEnvironmentSecured ? (
                <div className="inline-flex items-center gap-2 rounded-xl bg-emerald-950/30 px-4 py-2 text-xs font-black uppercase tracking-widest text-emerald-400 border border-emerald-900/50 shadow-sm">
                  <ShieldCheck size={14} /> Environment Secured
                </div>
              ) : (
                <div className="inline-flex items-center gap-2 rounded-xl bg-amber-950/20 px-4 py-2 text-xs font-black uppercase tracking-widest text-amber-500 border border-amber-900/30">
                  <AlertTriangle size={14} /> Security Confirmation Pending
                </div>
              )}
            </div>
          </article>
        </div>

        {/* ROW 2: BREATHING VISUALIZER */}
        <article className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
           <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 mb-8">
            <div>
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-indigo-400 flex items-center gap-2">
                <Wind size={16} /> Paced Respiration (Autonomic Regulation)
              </h3>
              <p className="text-sm text-slate-400 mt-2">
                Synchronized 4-4-6 state machine designed to trigger the parasympathetic nervous system.
              </p>
            </div>
            <button
              onClick={() => setAudioGuideEnabled(!audioGuideEnabled)}
              className={`shrink-0 inline-flex items-center gap-2 rounded-xl px-4 py-2 text-xs font-black uppercase tracking-widest transition-colors ${
                audioGuideEnabled ? 'border border-indigo-500/50 bg-indigo-900/30 text-indigo-300' : 'border border-slate-700 bg-slate-800 text-slate-400 hover:bg-slate-700'
              }`}
            >
              {audioGuideEnabled ? <Volume2 size={14} /> : <VolumeX size={14} />}
              {audioGuideEnabled ? 'Audio Guide: ON' : 'Audio Guide: OFF'}
            </button>
          </div>
          
          <div className="flex flex-col items-center gap-8 py-4">
            <div className="flex flex-col items-center gap-3">
              <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500">Therapeutic Circuit Progress</span>
              <div className="flex gap-2">
                {[...Array(TOTAL_BREATH_CIRCUITS)].map((_, i) => (
                  <div key={i} className={`h-2 w-10 rounded-full transition-colors duration-500 ${i < breathCount ? 'bg-indigo-500 shadow-[0_0_8px_rgba(99,102,241,0.5)]' : 'bg-slate-800'}`} />
                ))}
              </div>
            </div>

            <div
              className="grid place-items-center rounded-full border border-indigo-500/30 bg-[radial-gradient(circle_at_30%_30%,rgba(99,102,241,0.15),rgba(30,27,75,0.4)_55%,rgba(2,6,23,0.95)_100%)] shadow-[0_0_0_15px_rgba(99,102,241,0.03)]"
              style={{
                width: 'clamp(14rem, 40vw, 18rem)',
                aspectRatio: '1 / 1',
                transform: `scale(${breathPhase === 'INHALE' || breathPhase === 'HOLD' ? 1 : breathPhase === 'IDLE' ? 0.78 : 0.74})`,
                transition: `transform ${breathPhase === 'EXHALE' ? '6000ms' : '4000ms'} cubic-bezier(0.4, 0, 0.2, 1)`,
                willChange: 'transform'
              }}
            >
              <div className="text-center">
                <p className="text-xl font-black tracking-[0.2em] text-indigo-200 uppercase">
                  {breathPhase === 'IDLE' ? 'READY' : breathPhase}
                </p>
                {breathPhase !== 'IDLE' && breathPhase !== 'COMPLETE' && (
                   <p className="text-[10px] font-bold text-indigo-400/80 uppercase tracking-widest mt-2">
                     {breathPhase === 'EXHALE' ? '6 Seconds' : '4 Seconds'}
                   </p>
                )}
              </div>
            </div>

            <div className="h-12 flex items-center justify-center">
              {breathPhase === 'IDLE' ? (
                <button onClick={() => { setBreathCount(0); setBreathPhase('INHALE'); }} className="rounded-xl bg-indigo-600 px-8 py-3 text-sm font-black uppercase tracking-widest text-white hover:bg-indigo-500 transition-colors shadow-lg shadow-indigo-900/20">
                  Initiate Breathing Cycle
                </button>
              ) : breathPhase === 'COMPLETE' ? (
                <div className="flex items-center gap-4">
                  <button onClick={() => { setBreathCount(0); setBreathPhase('IDLE'); }} className="rounded-xl border-2 border-indigo-600/50 px-6 py-2.5 text-xs font-black uppercase tracking-widest text-indigo-300 hover:bg-indigo-950/30 transition-colors">
                    Repeat Cycle
                  </button>
                </div>
              ) : (
                <button onClick={() => { setBreathPhase('IDLE'); setBreathCount(0); stopAudioGuide(); }} className="text-xs font-black uppercase tracking-widest text-slate-500 hover:text-slate-300 transition-colors">
                  Halt Intervention
                </button>
              )}
            </div>
          </div>
        </article>

        {/* ROW 3: C-SSRS TRIAGE */}
        <article className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
          <h3 className="text-xs font-black uppercase tracking-[0.2em] text-amber-400 mb-2 flex items-center gap-2">
            <Activity size={16} /> Clinical Triage (C-SSRS Risk Assessment)
          </h3>
          <p className="text-sm text-slate-400 mb-6">Determine the necessary level of clinical escalation through structured risk formulation.</p>

          <div className="bg-slate-950/50 rounded-2xl border border-slate-800 p-6 min-h-[140px] flex flex-col justify-center">
            {cssrsStep === 0 && (
              <div className="text-center">
                <button onClick={() => setCssrsStep(1)} className="rounded-xl bg-amber-600 px-6 py-3 text-sm font-black uppercase tracking-widest text-white hover:bg-amber-500 transition-colors shadow-lg shadow-amber-900/20">
                  Begin Risk Assessment
                </button>
              </div>
            )}

            {cssrsStep === 1 && (
              <div className="max-w-2xl mx-auto w-full space-y-6">
                <p className="font-semibold text-lg text-slate-200 text-center">1. In the past 24 hours, have you had thoughts of ending your life or harming others?</p>
                <div className="flex justify-center gap-4">
                  <button onClick={() => setCssrsStep(2)} className="w-32 py-3 bg-slate-800 rounded-xl hover:bg-slate-700 text-white font-black uppercase tracking-widest transition-colors">Yes</button>
                  <button onClick={() => { setRiskLevel('Low'); setCssrsStep(4); }} className="w-32 py-3 bg-slate-800 rounded-xl hover:bg-slate-700 text-white font-black uppercase tracking-widest transition-colors">No</button>
                </div>
              </div>
            )}

            {cssrsStep === 2 && (
              <div className="max-w-2xl mx-auto w-full space-y-6">
                <p className="font-semibold text-lg text-slate-200 text-center">2. Do you have a specific plan or immediate access to lethal means?</p>
                <div className="flex justify-center gap-4">
                  <button onClick={() => setCssrsStep(3)} className="w-32 py-3 bg-slate-800 rounded-xl hover:bg-slate-700 text-white font-black uppercase tracking-widest transition-colors">Yes</button>
                  <button onClick={() => { setRiskLevel('Moderate'); setCssrsStep(4); }} className="w-32 py-3 bg-slate-800 rounded-xl hover:bg-slate-700 text-white font-black uppercase tracking-widest transition-colors">No</button>
                </div>
              </div>
            )}

            {cssrsStep === 3 && (
              <div className="max-w-2xl mx-auto w-full space-y-6">
                <p className="font-semibold text-lg text-slate-200 text-center">3. Do you have the intention to act on this plan right now?</p>
                <div className="flex justify-center gap-4">
                  <button onClick={() => { setRiskLevel('High'); setCssrsStep(4); logCrisisEvent('High'); }} className="w-32 py-3 bg-rose-600 rounded-xl hover:bg-rose-500 text-white font-black uppercase tracking-widest transition-colors shadow-lg shadow-rose-900/30">Yes</button>
                  <button onClick={() => { setRiskLevel('Moderate'); setCssrsStep(4); }} className="w-32 py-3 bg-slate-800 rounded-xl hover:bg-slate-700 text-white font-black uppercase tracking-widest transition-colors">No</button>
                </div>
              </div>
            )}

            {cssrsStep === 4 && (
              <div className="text-center space-y-4">
                <div className={`inline-flex items-center gap-3 px-6 py-3 rounded-xl border-2 ${riskLevel === 'High' ? 'bg-rose-950/30 border-rose-500/50 text-rose-400' : riskLevel === 'Moderate' ? 'bg-amber-950/30 border-amber-500/50 text-amber-400' : 'bg-emerald-950/30 border-emerald-500/50 text-emerald-400'}`}>
                  {riskLevel === 'High' && <ShieldAlert size={20} />}
                  <span className="font-black text-lg uppercase tracking-widest">
                    {riskLevel === 'High' ? 'CRITICAL ACUITY ACTIVATED' : `Triage Result: ${riskLevel} Risk`}
                  </span>
                </div>
                <p className="text-sm text-slate-300 max-w-xl mx-auto">
                  {riskLevel === 'High' ? "Immediate clinical dispatch is required. Please utilize Tier 3 emergency contacts below or execute the SOS Handoff." : "Acuity is stable. Please continue to utilize the sensory grounding tools above or contact a Tier 1 support person if distress increases."}
                </p>
                <button onClick={() => setCssrsStep(0)} className="mt-2 text-[10px] font-black uppercase tracking-widest text-slate-500 hover:text-slate-300 transition-colors">Restart Assessment</button>
              </div>
            )}
          </div>
        </article>

        {/* ROW 4: TIERED ESCALATION & SOS */}
        <article ref={contactsRef} className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 lg:p-8">
          <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4 mb-6">
            <div>
              <h3 className="text-xs font-black uppercase tracking-[0.2em] text-rose-400 flex items-center gap-2">
                <Phone size={16} /> Tiered Clinical Escalation & Support
              </h3>
              <p className="text-sm text-slate-400 mt-2">Trigger direct communications or export a geolocated SBAR protocol.</p>
            </div>
            <button
              onClick={shareSosReport}
              disabled={isSharingSos || isResolvingUserId || !targetUserId}
              className="shrink-0 inline-flex items-center gap-2 rounded-xl bg-rose-600 px-6 py-3 text-xs font-black uppercase tracking-widest text-white hover:bg-rose-500 transition-colors disabled:opacity-50 shadow-lg shadow-rose-900/20"
            >
              {isSharingSos ? <Loader2 size={16} className="animate-spin" /> : <Share2 size={16} />}
              Execute SOS Handoff
            </button>
          </div>

          <div className="grid gap-4 md:grid-cols-3">
            <div className="rounded-2xl border border-slate-800 bg-slate-950/50 p-5 hover:bg-slate-900/50 transition-colors">
              <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Tier 1: Personal</p>
              <p className="mt-1 text-sm font-bold text-slate-200">Trusted Contact</p>
              <a href={`tel:${TRUSTED_CONTACT_PLACEHOLDER}`} className="mt-3 block text-xl font-black text-indigo-400 hover:text-indigo-300 transition-colors">{TRUSTED_CONTACT_PLACEHOLDER}</a>
            </div>
            
            <div className="rounded-2xl border border-amber-900/30 bg-amber-950/10 p-5 hover:bg-amber-950/20 transition-colors">
              <p className="text-[10px] font-black text-amber-600/80 uppercase tracking-widest">Tier 2: Triage</p>
              <p className="mt-1 text-sm font-bold text-amber-200">Umang Helpline (PK)</p>
              <a href="tel:03111186264" className="mt-3 block text-xl font-black text-amber-400 hover:text-amber-300 transition-colors">0311-1186264</a>
            </div>
            
            <div className="rounded-2xl border border-rose-900/30 bg-rose-950/10 p-5 hover:bg-rose-950/20 transition-colors">
              <p className="text-[10px] font-black text-rose-600/80 uppercase tracking-widest">Tier 3: Dispatch</p>
              <p className="mt-1 text-sm font-bold text-rose-200">Immediate Rescue</p>
              <div className="mt-3 flex items-center gap-4">
                <a href="tel:1122" className="text-xl font-black text-rose-400 hover:text-rose-300 transition-colors">1122</a>
                <span className="text-slate-700">|</span>
                <a href="tel:115" className="text-xl font-black text-rose-400 hover:text-rose-300 transition-colors">115</a>
              </div>
            </div>
          </div>
          
          {(handoffError || shareNotice) && (
            <div className="mt-4 text-center">
              {handoffError && <p className="text-xs font-bold text-rose-400 uppercase tracking-widest">{handoffError}</p>}
              {shareNotice && <p className="text-xs font-bold text-emerald-400 uppercase tracking-widest">{shareNotice}</p>}
            </div>
          )}
        </article>
      </main>
    </div>
  );
};

export default SafetyPlanPage;
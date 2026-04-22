import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  AlertTriangle, // FIXED: Added missing icon
  ArrowLeft,
  Camera,
  Loader2,       // FIXED: Added missing icon (Caused the crash on 'Send'!)
  LogOut,
  MessageCircle,
  Mic,
  Send,
  Square,
  Video,         // FIXED: Added missing icon
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { useClinical } from '../context/ClinicalContext';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
const SHOW_PROVISIONAL_ASSISTANT_TEXT = (import.meta.env.VITE_SHOW_PROVISIONAL_ASSISTANT_TEXT || 'true').toLowerCase() === 'true';

const formatTime = (date = new Date()) => date.toLocaleTimeString('en-US', { 
  timeZone: 'Asia/Karachi' 
});

const stripStarredSegments = (value, preserveEdges = false) => {
  const source = String(value || '');
  if (!source) return '';
  const starIndex = source.indexOf('*');
  const cleaned = (starIndex === -1 ? source : source.slice(0, starIndex))
    .replace(/[ \t]+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1');
  return preserveEdges ? cleaned : cleaned.trim();
};

const cleanStreamToken = (token, reachedFirstAsterisk) => {
  if (reachedFirstAsterisk) return { cleanToken: '', reachedFirstAsterisk: true };
  const source = String(token || '');
  if (!source) return { cleanToken: '', reachedFirstAsterisk: false };
  const starIndex = source.indexOf('*');
  const visible = starIndex === -1 ? source : source.slice(0, starIndex);
  const cleanToken = visible
    .replace(/[ \t]+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1');
  return { cleanToken, reachedFirstAsterisk: starIndex !== -1 };
};

const hardClean = (value, preserveEdges = false) => {
  let cleaned = stripStarredSegments(value, true);
  cleaned = cleaned.replace(/([a-z])([A-Z])/g, '$1 $2');
  cleaned = cleaned.replace(/\s+/g, ' ');
  return preserveEdges ? cleaned : cleaned.trim();
};

const FRAMEWORK_PHASES = {
  DBT_Distress_Tolerance: ['Stabilization and Immediate Grounding', 'Crisis Survival Skills', 'Emotion Regulation During Peak Distress', 'Post-Crisis Recovery Plan'],
  CBT_Restructuring: ['Identify Automatic Thoughts', 'Label Cognitive Distortion', 'Evidence Examination', 'Balanced Reframe and Action'],
  ACT_Defusion: ['Notice Thought-Emotion Loop', 'Defusion from Narrative', 'Acceptance and Present-Moment Contact', 'Values-Aligned Micro-Action'],
  Supportive_Stabilization: ['Emotional Check-In', 'Clarify Needs and Stressors', 'Coping Plan and Commitment'],
};

const FRAMEWORK_ALIAS = { DBT: 'DBT_Distress_Tolerance', CBT: 'CBT_Restructuring', ACT: 'ACT_Defusion', SUPPORTIVE: 'Supportive_Stabilization' };
const FRAMEWORK_LABELS = { DBT_Distress_Tolerance: 'DBT', CBT_Restructuring: 'CBT', ACT_Defusion: 'ACT', Supportive_Stabilization: 'SUPPORTIVE' };

const NEGATIVE_EMOTIONS = new Set(['angry', 'sad', 'fear', 'disgust']);
const REGULATED_EMOTIONS = new Set(['calm', 'happy', 'neutral']);

const normalizeFrameworkKey = (value) => FRAMEWORK_ALIAS[String(value || '').trim().toUpperCase()] || String(value || '').trim() || 'Supportive_Stabilization';
const getFrameworkPhases = (frameworkKey) => FRAMEWORK_PHASES[normalizeFrameworkKey(frameworkKey)] || FRAMEWORK_PHASES.Supportive_Stabilization;

const UnifiedEmotionPage = ({ user, onLogout }) => {
  const { ingestBackendEvent, currentTherapyMode, setCrisisMode } = useClinical();
  const navigate = useNavigate();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const videoStreamRef = useRef(null);
  const audioStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const endSessionAfterUnitRef = useRef(false);
  const sessionActiveRef = useRef(false);
  const streamAudioQueueRef = useRef([]);
  const streamAudioActiveRef = useRef(false);
  const streamAudioElementRef = useRef(null);
  const streamAudioCurrentSequenceRef = useRef(null);
  const streamUiFrameRef = useRef(null);
  const streamUiPendingTextRef = useRef('');
  const streamUiPendingTurnRef = useRef(null);
  const streamReachedFirstAsteriskRef = useRef(false);

  const [cameraEnabled, setCameraEnabled] = useState(false);
  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [sessionActive, setSessionActive] = useState(false);
  const [isRecording, setIsRecording] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const [statusText, setStatusText] = useState('Start a session to begin.');
  const [promptText, setPromptText] = useState('');
  const [dominantEmotion, setDominantEmotion] = useState('Neutral');
  const [speechEmotion, setSpeechEmotion] = useState('Neutral');
  const [faceEmotion, setFaceEmotion] = useState('Neutral');
  const [transcription, setTranscription] = useState('');
  const [assistantText, setAssistantText] = useState('');

  const [notices, setNotices] = useState([]);
  const [logs, setLogs] = useState([]);
  const [emotionCounts, setEmotionCounts] = useState({});
  const [conversationTurns, setConversationTurns] = useState([]);
  const [phaseIndex, setPhaseIndex] = useState(0);
  const [currentPhase, setCurrentPhase] = useState('Emotional Check-In');

  const greetingLine = useMemo(() => `${user}`, [user]);
  const activeFrameworkKey = useMemo(() => normalizeFrameworkKey(currentTherapyMode), [currentTherapyMode]);
  const activeFrameworkLabel = useMemo(() => FRAMEWORK_LABELS[activeFrameworkKey] || String(currentTherapyMode || 'SUPPORTIVE'), [activeFrameworkKey, currentTherapyMode]);

  const applyClinicalProtocolStatus = (payload = {}) => {
    const nextFramework = normalizeFrameworkKey(payload?.framework || payload?.mode || activeFrameworkKey);
    const phases = getFrameworkPhases(nextFramework);
    const explicitPhase = String(payload?.phase || '').trim();
    const explicitIndex = Number(payload?.phase_index);

    if (Number.isFinite(explicitIndex)) {
      const boundedIndex = Math.max(0, Math.min(explicitIndex, phases.length - 1));
      setPhaseIndex(boundedIndex);
      setCurrentPhase(explicitPhase || phases[boundedIndex] || 'Emotional Check-In');
      return;
    }

    if (explicitPhase) {
      const mappedIndex = phases.findIndex((phase) => phase.toLowerCase() === explicitPhase.toLowerCase());
      if (mappedIndex >= 0) setPhaseIndex(mappedIndex);
      setCurrentPhase(explicitPhase);
    }
  };

  const advancePhaseAutonomously = (detectedDistortion = '') => {
    setPhaseIndex((previous) => {
      const phases = getFrameworkPhases(activeFrameworkKey);
      const next = Math.min(previous + 1, phases.length - 1);
      setCurrentPhase(phases[next] || 'Emotional Check-In');
      return next;
    });

    const normalizedDistortion = String(detectedDistortion || '').trim();
    if (normalizedDistortion) pushNotice(`Detected cognitive distortion: ${normalizedDistortion}`);
  };

  useEffect(() => {
    const phases = getFrameworkPhases(activeFrameworkKey);
    const boundedIndex = Math.max(0, Math.min(phaseIndex, phases.length - 1));
    if (boundedIndex !== phaseIndex) {
      setPhaseIndex(boundedIndex);
      return;
    }
    const expectedPhase = phases[boundedIndex] || 'Emotional Check-In';
    const phaseBelongsToFramework = phases.some((phase) => phase.toLowerCase() === String(currentPhase || '').toLowerCase());
    if (!currentPhase || !phaseBelongsToFramework) setCurrentPhase(expectedPhase);
  }, [activeFrameworkKey, phaseIndex, currentPhase]);

  useEffect(() => { sessionActiveRef.current = sessionActive; }, [sessionActive]);

  const pushNotice = (message) => {
    if (!message) return;
    setNotices((prev) => [message, ...prev.filter((item) => item !== message)].slice(0, 4));
  };

  const addEmotionLog = (emotion) => {
    const normalized = (emotion || 'Neutral').toLowerCase();
    setLogs((prev) => [{ time: formatTime(), emotion: normalized }, ...prev].slice(0, 20));
    setEmotionCounts((prev) => ({ ...prev, [normalized]: (prev[normalized] || 0) + 1 }));
  };

  const appendConversationTurn = ({ userText, assistantReply, emotion, source }) => {
    const timestamp = formatTime();
    const turnId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setConversationTurns((prev) => [
      ...prev,
      { id: turnId, time: timestamp, userText: userText || '(no text recognized)', assistantReply: assistantReply || '', emotion: (emotion || 'Neutral').toLowerCase(), source },
    ]);
    return turnId;
  };

  const reconcileLatestEmotionLog = (fromEmotion, toEmotion) => {
    const from = String(fromEmotion || '').trim().toLowerCase();
    const to = String(toEmotion || '').trim().toLowerCase();
    if (!from || !to || from === to) return;

    setLogs((prev) => {
      if (prev.length === 0) return prev;
      const [head, ...rest] = prev;
      return [{ ...head, emotion: to }, ...rest];
    });

    setEmotionCounts((prev) => {
      const next = { ...prev };
      if (next[from]) {
        next[from] = Math.max(0, next[from] - 1);
        if (next[from] === 0) delete next[from];
      }
      next[to] = (next[to] || 0) + 1;
      return next;
    });
  };

  const updateConversationTurn = (turnId, updates) => {
    if (!turnId) return;
    setConversationTurns((prev) => prev.map((turn) => turn.id === turnId ? { ...turn, ...updates, emotion: updates?.emotion ? String(updates.emotion).toLowerCase() : turn.emotion } : turn));
  };

  const flushStreamUiUpdate = () => {
    streamUiFrameRef.current = null;
    const nextText = String(streamUiPendingTextRef.current || '');
    const nextTurnId = streamUiPendingTurnRef.current;
    setAssistantText(nextText);
    if (nextTurnId) updateConversationTurn(nextTurnId, { assistantReply: nextText });
  };

  const scheduleStreamUiUpdate = (text, turnId = null) => {
    streamUiPendingTextRef.current = String(text || '');
    if (turnId) streamUiPendingTurnRef.current = turnId;
    if (streamUiFrameRef.current !== null) return;
    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      streamUiFrameRef.current = window.requestAnimationFrame(flushStreamUiUpdate);
      return;
    }
    streamUiFrameRef.current = setTimeout(flushStreamUiUpdate, 16);
  };

  const cancelStreamUiUpdate = () => {
    if (streamUiFrameRef.current === null) return;
    if (typeof window !== 'undefined' && typeof window.cancelAnimationFrame === 'function') {
      window.cancelAnimationFrame(streamUiFrameRef.current);
    } else {
      clearTimeout(streamUiFrameRef.current);
    }
    streamUiFrameRef.current = null;
    streamUiPendingTextRef.current = '';
    streamUiPendingTurnRef.current = null;
  };

  const playNextStreamAudioSegment = () => {
    const nextSegment = streamAudioQueueRef.current.shift();
    if (!nextSegment) {
      streamAudioActiveRef.current = false;
      streamAudioElementRef.current = null;
      streamAudioCurrentSequenceRef.current = null;
      return;
    }

    const nextAudioBase64 = String(nextSegment.audioBase64 || '').trim();
    if (!nextAudioBase64) { playNextStreamAudioSegment(); return; }

    const parsedSequence = Number(nextSegment.sequence);
    streamAudioCurrentSequenceRef.current = Number.isFinite(parsedSequence) ? parsedSequence : null;
    streamAudioActiveRef.current = true;
    const audio = new Audio(`data:audio/mpeg;base64,${nextAudioBase64}`);
    streamAudioElementRef.current = audio;
    audio.onended = () => playNextStreamAudioSegment();
    audio.onerror = () => playNextStreamAudioSegment();
    audio.play().catch(() => playNextStreamAudioSegment());
  };

  const resetStreamAudioPlayback = () => {
    streamAudioQueueRef.current = [];
    streamAudioActiveRef.current = false;
    streamAudioCurrentSequenceRef.current = null;
    if (streamAudioElementRef.current) { streamAudioElementRef.current.pause(); streamAudioElementRef.current = null; }
    if (typeof window !== 'undefined' && window.speechSynthesis) window.speechSynthesis.cancel();
  };

  const enqueueStreamAudioSegment = (audioBase64, sequence = null) => {
    const normalized = String(audioBase64 || '').trim();
    if (!normalized) return;
    const parsedSequence = Number(sequence);
    streamAudioQueueRef.current.push({ audioBase64: normalized, sequence: Number.isFinite(parsedSequence) ? parsedSequence : null });
    if (!streamAudioActiveRef.current) playNextStreamAudioSegment();
  };

  const trimStreamAudioPlayback = (maxSequence) => {
    const parsedMax = Number(maxSequence);
    if (!Number.isFinite(parsedMax)) return;
    streamAudioQueueRef.current = streamAudioQueueRef.current.filter((segment) => {
      const segmentSequence = Number(segment.sequence);
      if (!Number.isFinite(segmentSequence)) return true;
      return segmentSequence <= parsedMax;
    });

    const currentSequence = streamAudioCurrentSequenceRef.current;
    if (Number.isFinite(currentSequence) && currentSequence > parsedMax) {
      if (streamAudioElementRef.current) { streamAudioElementRef.current.pause(); streamAudioElementRef.current = null; }
      streamAudioActiveRef.current = false;
      streamAudioCurrentSequenceRef.current = null;
      if (streamAudioQueueRef.current.length > 0) playNextStreamAudioSegment();
    }
  };

  const toFriendlyNotices = (rawErrors = []) => {
    const mapped = rawErrors.map((error) => {
      const lower = String(error).toLowerCase();
      if (lower.includes('rag model unavailable') || lower.includes('rag initialization failed') || lower.includes('llm generation failed') || lower.includes('llm timeout')) return 'LLM fallback triggered for this turn. Retry if needed.';
      if (lower.includes('tts failed') || lower.includes('tts timeout') || lower.includes('tts disabled')) return null;
      if (lower.includes('audio upload was empty')) return 'No speech captured. Speak clearly before pressing Stop.';
      return String(error);
    });
    return [...new Set(mapped.filter(Boolean))].slice(0, 4);
  };

  const playTtsIfAvailable = async (payload) => {
    const segments = Array.isArray(payload?.tts_audio_segments_base64) ? payload.tts_audio_segments_base64.filter(Boolean) : [];
    if (segments.length > 0) {
      try {
        for (const segment of segments) {
          const audio = new Audio(`data:audio/mpeg;base64,${segment}`);
          await new Promise((resolve, reject) => { audio.onended = () => resolve(); audio.onerror = () => reject(new Error('Audio playback failed.')); audio.play().catch(reject); });
        }
        return;
      } catch { pushNotice('Audio playback blocked by browser.'); }
    }
    if (payload?.tts_audio_base64) {
      try {
        const audio = new Audio(`data:audio/mpeg;base64,${payload.tts_audio_base64}`);
        await audio.play();
        return;
      } catch { pushNotice('Audio playback blocked by browser.'); }
    }
    const fallbackText = hardClean(payload?.llm_response || '');
    if (!fallbackText || typeof window === 'undefined' || !window.speechSynthesis) return;
    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(fallbackText);
      utterance.lang = 'en-US';
      window.speechSynthesis.speak(utterance);
    } catch { }
  };

  const updateFromPayload = async (payload, fallbackUserText = '', source = 'voice', existingTurnId = null) => {
    const cleanedAssistantReply = hardClean(payload?.llm_response || '');
    setDominantEmotion(payload.dominant_emotion || 'Neutral');
    setSpeechEmotion(payload.speech_emotion || 'Neutral');
    setFaceEmotion(payload.face_emotion || 'Neutral');
    setTranscription(payload.transcription || '');
    setAssistantText(cleanedAssistantReply);
    addEmotionLog(payload.dominant_emotion || 'Neutral');

    const userText = (payload.transcription || fallbackUserText || '').trim();
    if (existingTurnId) {
      updateConversationTurn(existingTurnId, { userText, assistantReply: cleanedAssistantReply, emotion: payload.dominant_emotion });
    } else {
      appendConversationTurn({ userText, assistantReply: cleanedAssistantReply, emotion: payload.dominant_emotion, source });
    }
    setNotices(toFriendlyNotices(payload.errors || []));
    await playTtsIfAvailable(payload);
  };

  const stopStreams = () => {
    if (videoStreamRef.current) { videoStreamRef.current.getTracks().forEach((track) => track.stop()); videoStreamRef.current = null; }
    if (audioStreamRef.current) { audioStreamRef.current.getTracks().forEach((track) => track.stop()); audioStreamRef.current = null; }
    if (videoRef.current) videoRef.current.srcObject = null;
    setIsCameraOn(false);
    setIsMicOn(false);
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop();
      cancelStreamUiUpdate();
      resetStreamAudioPlayback();
      stopStreams();
    };
  }, []);

  const streamNdjson = async ({ url, method = 'POST', headers = {}, body, onEvent }) => {
    const response = await fetch(url, { method, headers, body });
    if (!response.ok) throw new Error(`Request failed with status ${response.status}`);
    if (!response.body) throw new Error('Streaming response body is not available.');

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        try { await onEvent(JSON.parse(trimmed)); } catch { }
      }
    }
    const tail = buffer.trim();
    if (tail) {
      try { await onEvent(JSON.parse(tail)); } catch { }
    }
  };

  const ensureVideoStream = async () => {
    if (!cameraEnabled) return null;
    if (videoStreamRef.current && videoStreamRef.current.getVideoTracks().length > 0) return videoStreamRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoStreamRef.current = stream;
    if (videoRef.current) videoRef.current.srcObject = stream;
    setIsCameraOn(true);
    return stream;
  };

  const ensureAudioStream = async () => {
    if (audioStreamRef.current && audioStreamRef.current.getAudioTracks().length > 0) return audioStreamRef.current;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    audioStreamRef.current = stream;
    setIsMicOn(true);
    return stream;
  };

  const captureSnapshot = () => {
    if (!videoRef.current || !canvasRef.current || !videoRef.current.videoWidth) return '';
    const ctx = canvasRef.current.getContext('2d');
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    return canvasRef.current.toDataURL('image/jpeg');
  };

  const buildRecorder = (stream) => {
    const preferredTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
    const selectedType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type));
    return selectedType ? new MediaRecorder(stream, { mimeType: selectedType }) : new MediaRecorder(stream);
  };

  const submitInteraction = async ({ imageBase64 = '', audioBlob = null, textFallback = '', existingTurnId = null }) => {
    const formData = new FormData();
    formData.append('username', user || 'anonymous');
    if (imageBase64) formData.append('image', imageBase64);
    if (textFallback) formData.append('user_message', textFallback);
    if (audioBlob) formData.append('file', audioBlob, 'interaction.webm');
    const response = await axios.post(`${API_BASE_URL}/api/interact`, formData, { timeout: 120000 });
    await updateFromPayload(response.data, textFallback, 'voice', existingTurnId);
  };

  const submitInteractionStream = async ({ imageBase64 = '', audioBlob = null, textFallback = '' }) => {
    const formData = new FormData();
    formData.append('username', user || 'anonymous');
    if (imageBase64) formData.append('image', imageBase64);
    if (textFallback) formData.append('user_message', textFallback);
    if (audioBlob) formData.append('file', audioBlob, 'interaction.webm');

    let turnId = null;
    let liveAssistantText = '';
    let emotionLoggedForTurn = false;
    let provisionalEmotion = '';

    resetStreamAudioPlayback();
    streamReachedFirstAsteriskRef.current = false;

    await streamNdjson({
      url: `${API_BASE_URL}/api/interact/stream`,
      method: 'POST',
      body: formData,
      onEvent: async (event) => {
        ingestBackendEvent(event);
        const eventType = String(event?.type || '').toLowerCase();

        if (eventType === 'clinical_protocol_status') { applyClinicalProtocolStatus(event); return; }
        if (eventType === 'protocol_control') {
          if (event?.advance_phase === true) advancePhaseAutonomously(event?.detected_distortion || '');
          else if (event?.detected_distortion) pushNotice(`Detected cognitive distortion: ${String(event.detected_distortion)}`);
          return;
        }
        if (eventType === 'transcription' || eventType === 'user_text') {
          const text = String(event?.text || textFallback || '').trim();
          setTranscription(text);
          if (!turnId && text) {
            turnId = appendConversationTurn({ userText: text, assistantReply: '', emotion: dominantEmotion, source: 'voice' });
          } else if (turnId && text) {
            updateConversationTurn(turnId, { userText: text });
          }
          return;
        }
        if (eventType === 'emotion_partial') {
          const provisional = String(event?.speech_emotion || event?.face_emotion || '').trim();
          if (event?.speech_emotion) setSpeechEmotion(String(event.speech_emotion));
          if (event?.face_emotion) setFaceEmotion(String(event.face_emotion));
          if (provisional && !emotionLoggedForTurn) {
            addEmotionLog(provisional);
            provisionalEmotion = provisional;
            emotionLoggedForTurn = true;
            if (turnId) updateConversationTurn(turnId, { emotion: provisional });
          }
          return;
        }
        if (eventType === 'emotion') {
          const dominant = String(event?.dominant_emotion || 'Neutral');
          setDominantEmotion(dominant);
          setSpeechEmotion(String(event?.speech_emotion || 'Neutral'));
          setFaceEmotion(String(event?.face_emotion || 'Neutral'));
          if (!emotionLoggedForTurn) { addEmotionLog(dominant); emotionLoggedForTurn = true; } 
          else { reconcileLatestEmotionLog(provisionalEmotion, dominant); }
          if (turnId) updateConversationTurn(turnId, { emotion: dominant });
          return;
        }
        if (eventType === 'assistant_delta') {
          let rawToken = String(event?.delta || '');
          if (!rawToken && event?.text) rawToken = event.text.startsWith(liveAssistantText) ? event.text.slice(liveAssistantText.length) : event.text;
          const { cleanToken, reachedFirstAsterisk } = cleanStreamToken(rawToken, streamReachedFirstAsteriskRef.current);
          streamReachedFirstAsteriskRef.current = reachedFirstAsterisk;
          if (!cleanToken) return;

          const isFirstVisibleToken = !liveAssistantText;
          liveAssistantText += cleanToken;
          if (SHOW_PROVISIONAL_ASSISTANT_TEXT) {
            if (!turnId) turnId = appendConversationTurn({ userText: textFallback || '(processing...)', assistantReply: liveAssistantText, emotion: dominantEmotion, source: 'voice' });
            if (isFirstVisibleToken) { cancelStreamUiUpdate(); setAssistantText(liveAssistantText); updateConversationTurn(turnId, { assistantReply: liveAssistantText }); }
            else scheduleStreamUiUpdate(liveAssistantText, turnId);
          }
          return;
        }
        if (eventType === 'assistant_replace') {
          liveAssistantText = hardClean(String(event?.text || liveAssistantText));
          streamReachedFirstAsteriskRef.current = false;
          if (liveAssistantText) { cancelStreamUiUpdate(); setAssistantText(liveAssistantText); if (turnId) updateConversationTurn(turnId, { assistantReply: liveAssistantText }); }
          return;
        }
        if (eventType === 'assistant_tts_reset') { resetStreamAudioPlayback(); return; }
        if (eventType === 'assistant_tts_trim') { trimStreamAudioPlayback(event?.max_sequence); return; }
        if (eventType === 'assistant_sentence_tts') { enqueueStreamAudioSegment(event?.audio_base64, event?.sequence); return; }
        if (eventType === 'error') {
          const friendly = toFriendlyNotices([event?.message]).filter(Boolean);
          if (friendly.length > 0) friendly.forEach(pushNotice);
          else if (event?.message) pushNotice(String(event.message));
          return;
        }
        if (eventType === 'final') {
          const finalReply = hardClean(event?.llm_response || liveAssistantText || '');
          streamReachedFirstAsteriskRef.current = false;
          applyClinicalProtocolStatus(event?.clinical || {});
          setTranscription(message);
          setAssistantText(finalReply);
          updateConversationTurn(turnId, { userText: message, assistantReply: finalReply, emotion: String(event?.dominant_emotion || 'Neutral') });
          if (event?.tts_audio_base64) enqueueStreamAudioSegment(event.tts_audio_base64, Number.MAX_SAFE_INTEGER);
          setStatusText('Response complete. Ready for next input.');
        }
      },
    });
    return turnId;
  };

  const startSession = () => {
    endSessionAfterUnitRef.current = false;
    setSessionActive(true);
    setConversationTurns([]);
    setLogs([]);
    setEmotionCounts({});
    setNotices([]);
    setTranscription('');
    cancelStreamUiUpdate();
    setAssistantText('');
    streamReachedFirstAsteriskRef.current = false;
    resetStreamAudioPlayback();
    setStatusText('Session active. Press Speak to initiate dialogue.');
  };

  const closeSessionNow = () => {
    endSessionAfterUnitRef.current = false;
    setSessionActive(false);
    setIsRecording(false);
    setIsProcessing(false);
    streamReachedFirstAsteriskRef.current = false;
    cancelStreamUiUpdate();
    resetStreamAudioPlayback();
    stopStreams();
    setStatusText('Session terminated. Press Start Session to reconnect.');
  };

  const handleSessionToggle = () => {
    if (!sessionActive) { startSession(); return; }
    if (isRecording) {
      endSessionAfterUnitRef.current = true;
      setStatusText('Concluding current dialogue turn before terminating session...');
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop();
      return;
    }
    closeSessionNow();
  };

  const startVoiceUnit = async () => {
    if (!sessionActive) { pushNotice('Initiate a session first.'); return; }
    if (isRecording || isProcessing) return;

    setNotices([]);
    setStatusText('Recording... Press Stop to finalize audio input.');

    try {
      const stream = await ensureAudioStream();
      let captureWithCamera = false;
      if (cameraEnabled) {
        try { await ensureVideoStream(); captureWithCamera = true; } 
        catch (videoError) { captureWithCamera = false; setIsCameraOn(false); pushNotice(`Vision offline: ${videoError.message}`); }
      }

      const recorder = buildRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
      recorder.onerror = () => { setIsRecording(false); pushNotice('Audio capture failed.'); };
      recorder.onstop = async () => {
        setIsRecording(false);
        setIsProcessing(true);
        setStatusText('Analyzing affective presentation and synthesizing response...');

        const audioBlob = new Blob(chunksRef.current, { type: chunksRef.current[0]?.type || 'audio/webm' });
        let streamedTurnId = null;
        chunksRef.current = [];

        try {
          if (!audioBlob || audioBlob.size === 0) { pushNotice('No audio detected.'); } 
          else {
            const imageBase64 = captureWithCamera ? captureSnapshot() : '';
            streamedTurnId = await submitInteractionStream({ imageBase64, audioBlob, textFallback: promptText.trim() });
          }
        } catch (error) {
          try {
            const imageBase64 = captureWithCamera ? captureSnapshot() : '';
            await submitInteraction({ imageBase64, audioBlob, textFallback: promptText.trim(), existingTurnId: streamedTurnId });
          } catch (fallbackError) { pushNotice('Backend connection failed. System operating in degraded mode.'); }
        } finally {
          setIsProcessing(false);
          if (endSessionAfterUnitRef.current) closeSessionNow();
          else if (sessionActiveRef.current) setStatusText('Response complete. Ready for next input.');
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      setIsMicOn(false);
      setStatusText('Microphone permission required for session.');
    }
  };

  const stopVoiceUnit = () => { if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') mediaRecorderRef.current.stop(); };
  const handleSpeakToggle = async () => { if (isRecording) stopVoiceUnit(); else await startVoiceUnit(); };

  const toggleVision = async () => {
    if (isRecording || isProcessing) { pushNotice('Vision toggle locked during active transcription.'); return; }
    if (cameraEnabled) {
      setCameraEnabled(false);
      stopStreams();
      setStatusText('Vision Disabled. Operating in Voice-Only mode.');
      return;
    }
    setCameraEnabled(true);
    try {
      await ensureVideoStream();
      setStatusText('Vision Enabled. Multi-modal affective tracking active.');
    } catch (error) {
      setCameraEnabled(false);
      setIsCameraOn(false);
      setStatusText('Vision hardware unavailable.');
    }
  };

  const sendTextPrompt = async () => {
    const message = promptText.trim();
    let turnId = null;
    if (!sessionActive) { pushNotice('Initiate a session first.'); return; }
    if (!message || isRecording || isProcessing) return;

    setNotices([]);
    setIsProcessing(true);
    setStatusText('Synthesizing response...');

    try {
      turnId = appendConversationTurn({ userText: message, assistantReply: '', emotion: dominantEmotion, source: 'text' });
      let liveAssistantText = '';
      resetStreamAudioPlayback();
      streamReachedFirstAsteriskRef.current = false;

      await streamNdjson({
        url: `${API_BASE_URL}/api/chat/stream`,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: user || 'anonymous', message }),
        onEvent: async (event) => {
          ingestBackendEvent(event);
          const eventType = String(event?.type || '').toLowerCase();
          if (eventType === 'clinical_protocol_status') { applyClinicalProtocolStatus(event); return; }
          if (eventType === 'protocol_control') {
            if (event?.advance_phase === true) advancePhaseAutonomously(event?.detected_distortion || '');
            else if (event?.detected_distortion) pushNotice(`Detected cognitive distortion: ${String(event.detected_distortion)}`);
            return;
          }
          if (eventType === 'emotion') {
            const dominant = String(event?.dominant_emotion || 'Neutral');
            setDominantEmotion(dominant); setSpeechEmotion(String(event?.speech_emotion || 'Neutral')); setFaceEmotion(String(event?.face_emotion || 'Neutral'));
            addEmotionLog(dominant); updateConversationTurn(turnId, { emotion: dominant });
            return;
          }
          if (eventType === 'assistant_delta') {
            let rawToken = String(event?.delta || '');
            if (!rawToken && event?.text) rawToken = event.text.startsWith(liveAssistantText) ? event.text.slice(liveAssistantText.length) : event.text;
            const { cleanToken, reachedFirstAsterisk } = cleanStreamToken(rawToken, streamReachedFirstAsteriskRef.current);
            streamReachedFirstAsteriskRef.current = reachedFirstAsterisk;
            if (!cleanToken) return;

            const isFirstVisibleToken = !liveAssistantText;
            liveAssistantText += cleanToken;
            if (SHOW_PROVISIONAL_ASSISTANT_TEXT) {
              if (isFirstVisibleToken) { cancelStreamUiUpdate(); setAssistantText(liveAssistantText); updateConversationTurn(turnId, { assistantReply: liveAssistantText }); }
              else scheduleStreamUiUpdate(liveAssistantText, turnId);
            }
            return;
          }
          if (eventType === 'assistant_replace') {
            liveAssistantText = hardClean(String(event?.text || liveAssistantText));
            streamReachedFirstAsteriskRef.current = false;
            if (liveAssistantText) { cancelStreamUiUpdate(); setAssistantText(liveAssistantText); updateConversationTurn(turnId, { assistantReply: liveAssistantText }); }
            return;
          }
          if (eventType === 'assistant_tts_reset') { resetStreamAudioPlayback(); return; }
          if (eventType === 'assistant_tts_trim') { trimStreamAudioPlayback(event?.max_sequence); return; }
          if (eventType === 'assistant_sentence_tts') { enqueueStreamAudioSegment(event?.audio_base64, event?.sequence); return; }
          if (eventType === 'error') {
            const friendly = toFriendlyNotices([event?.message]).filter(Boolean);
            if (friendly.length > 0) friendly.forEach(pushNotice);
            else if (event?.message) pushNotice(String(event.message));
            return;
          }
          if (eventType === 'final') {
            const finalReply = hardClean(event?.llm_response || liveAssistantText || '');
            streamReachedFirstAsteriskRef.current = false;
            applyClinicalProtocolStatus(event?.clinical || {});
            setTranscription(message);
            setAssistantText(finalReply);
            updateConversationTurn(turnId, { userText: message, assistantReply: finalReply, emotion: String(event?.dominant_emotion || 'Neutral') });
            if (event?.tts_audio_base64) enqueueStreamAudioSegment(event.tts_audio_base64, Number.MAX_SAFE_INTEGER);
            setStatusText('Response complete. Ready for next input.');
          }
        },
      });
      setPromptText('');
    } catch (error) {
      try {
        const response = await axios.post(`${API_BASE_URL}/api/chat`, { username: user || 'anonymous', message }, { timeout: 120000 });
        await updateFromPayload(response.data, message, 'text', turnId);
        setPromptText('');
      } catch (fallbackError) { pushNotice('Backend unreachable.'); setStatusText('Transmission failed.'); }
    } finally { setIsProcessing(false); }
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
            <MessageCircle className="text-emerald-500" size={22} /> SERENITY <span className="text-slate-500 font-normal">Live Support Session</span>
          </h1>
        </div>
        <div className="flex items-center gap-6">
          <div className="text-right hidden sm:block">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">Active Patient</p>
            <p className="text-sm font-semibold text-slate-200">{greetingLine}</p>
          </div>
          <button onClick={onLogout} className="text-rose-400 text-xs font-black uppercase tracking-widest border border-rose-900/40 px-3 py-2 rounded-xl hover:bg-rose-950/30 transition-all flex items-center gap-2">
            <LogOut size={14} /> End Session
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6 max-w-[1600px] mx-auto w-full grid grid-cols-1 xl:grid-cols-12 gap-6">
        
        {/* LEFT COLUMN: CLINICAL STATUS & LOGS */}
        <section className="xl:col-span-3 space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm">
            <p className="text-[10px] uppercase font-black tracking-[0.2em] text-indigo-400 mb-4">Hardware Telemetry</p>
            <div className="space-y-3">
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-950/50 border border-slate-800">
                <span className="text-xs font-bold text-slate-400">Connection</span>
                <span className={`text-xs font-black uppercase tracking-widest ${sessionActive ? 'text-emerald-400' : 'text-slate-500'}`}>{sessionActive ? 'Active' : 'Offline'}</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-950/50 border border-slate-800">
                <span className="text-xs font-bold text-slate-400">Microphone</span>
                <span className={`text-xs font-black uppercase tracking-widest ${isMicOn ? 'text-sky-400' : 'text-slate-500'}`}>{isMicOn ? 'Enabled' : 'Disabled'}</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-950/50 border border-slate-800">
                <span className="text-xs font-bold text-slate-400">Vision Mode</span>
                <span className={`text-xs font-black uppercase tracking-widest ${cameraEnabled ? 'text-amber-400' : 'text-slate-500'}`}>{cameraEnabled ? 'Active' : 'Voice-Only'}</span>
              </div>
              <div className="flex justify-between items-center p-3 rounded-xl bg-slate-950/50 border border-slate-800">
                <span className="text-xs font-bold text-slate-400">Acuity Status</span>
                <span className={`text-xs font-black uppercase tracking-widest ${isRecording ? 'text-rose-400 animate-pulse' : isProcessing ? 'text-indigo-400 animate-pulse' : 'text-emerald-400'}`}>
                  {isRecording ? 'Listening' : isProcessing ? 'Synthesizing' : 'Standby'}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-emerald-400 mb-4">Affective Check-In Log</p>
             <div className="space-y-2 max-h-[220px] overflow-y-auto pr-2 custom-scrollbar">
                {logs.length === 0 && <p className="text-xs text-slate-500 italic py-2">No clinical affect registered.</p>}
                {logs.map((entry, index) => (
                  <div key={`${entry.time}-${index}`} className="text-xs flex justify-between p-2 rounded-lg hover:bg-slate-800/50 transition-colors">
                    <span className="text-slate-500 font-bold">[{entry.time}]</span>
                    <span className="text-emerald-400 font-black uppercase tracking-widest">{entry.emotion}</span>
                  </div>
                ))}
              </div>
          </div>

          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-amber-400 mb-4">Emotional Distribution</p>
             <div className="space-y-3 max-h-[200px] overflow-y-auto pr-2 custom-scrollbar">
                {Object.keys(emotionCounts).length === 0 && <p className="text-xs text-slate-500 italic py-2">Insufficient data.</p>}
                {Object.entries(emotionCounts).sort((a, b) => b[1] - a[1]).map(([emotion, count]) => (
                  <div key={emotion} className="flex justify-between items-center p-2 border border-slate-800 rounded-lg bg-slate-950/50">
                    <span className="text-slate-300 text-xs font-bold capitalize">{emotion}</span>
                    <span className="text-amber-400 font-black">x{count}</span>
                  </div>
                ))}
              </div>
          </div>

          {/* ACTIVE PROTOCOL STATUS */}
          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm">
             <p className="text-[10px] uppercase font-black tracking-[0.2em] text-indigo-400 mb-4">Active Protocol Status</p>
             <div className="p-4 rounded-2xl bg-indigo-950/20 border border-indigo-900/40">
                <p className="text-xs font-bold text-slate-400 mb-1">Clinical Framework: <span className="font-black text-indigo-300 uppercase tracking-widest ml-1">{activeFrameworkLabel}</span></p>
                <p className="text-xs font-bold text-slate-400">Current Phase ({phaseIndex + 1}): <span className="font-black text-emerald-300 uppercase tracking-tight ml-1">{currentPhase}</span></p>
             </div>
           </div>

        </section>

        {/* CENTER COLUMN: VIDEO & CONTROLS */}
        <section className="xl:col-span-5 space-y-6 flex flex-col">
           <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm flex-1 flex flex-col">
              <div className="flex justify-between items-center mb-4">
                <p className="text-[10px] uppercase font-black tracking-[0.2em] text-slate-500">Live Sensory Input</p>
                {cameraEnabled && (
                  <span className="flex items-center gap-1.5 px-2 py-0.5 rounded border border-rose-900/50 bg-rose-950/30 text-[9px] font-black uppercase tracking-widest text-rose-400 animate-pulse">
                    <div className="w-1.5 h-1.5 rounded-full bg-rose-500" /> Live
                  </span>
                )}
              </div>
              
              {/* FIXED: Replaced aspect-video with flex-1 and min-height to fill the remaining space */}
              <div className="relative flex-1 w-full min-h-[300px] rounded-[1.5rem] overflow-hidden border-2 border-slate-800 bg-slate-950 shadow-inner group">
                <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
                <canvas ref={canvasRef} className="hidden" />
                {!isCameraOn && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-500 gap-3">
                    <Video size={32} className="opacity-50" />
                    <span className="text-xs font-black uppercase tracking-widest">Vision Mode Offline</span>
                  </div>
                )}
                {/* HUD Overlay */}
                <div className="absolute top-4 left-4 bg-slate-950/80 backdrop-blur-md px-4 py-3 rounded-xl border border-slate-700/50 shadow-lg">
                  <p className="text-[9px] uppercase font-black tracking-[0.2em] text-slate-400 mb-1">Current Affect</p>
                  <p className="text-2xl font-black text-white capitalize leading-none">{dominantEmotion}</p>
                  <div className="mt-2 flex gap-3 text-[10px] font-bold text-slate-400 uppercase tracking-wider">
                    <span>S: <span className="text-indigo-400">{speechEmotion}</span></span>
                    <span>F: <span className="text-emerald-400">{faceEmotion}</span></span>
                  </div>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-3 gap-4">
                <button onClick={toggleVision} disabled={isRecording || isProcessing} className={`inline-flex items-center justify-center gap-2 rounded-xl border px-4 py-3 text-xs font-black uppercase tracking-widest transition-all ${cameraEnabled ? 'bg-amber-950/30 border-amber-900/50 text-amber-400 hover:bg-amber-900/40' : 'bg-slate-800 border-slate-700 text-slate-300 hover:bg-slate-700 disabled:opacity-50'}`}>
                  <Camera size={16} /> {cameraEnabled ? 'Disable Vision' : 'Enable Vision'}
                </button>

                <button onClick={handleSpeakToggle} disabled={!sessionActive || isProcessing} className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-xs font-black uppercase tracking-widest transition-all shadow-lg ${isRecording ? 'bg-rose-600 hover:bg-rose-500 text-white shadow-rose-900/20' : 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-indigo-900/20 disabled:opacity-50'}`}>
                  {isRecording ? <Square size={16} /> : <Mic size={16} />}
                  {isRecording ? 'Stop Recording' : 'Push to Speak'}
                </button>

                <button onClick={handleSessionToggle} className={`inline-flex items-center justify-center gap-2 rounded-xl px-4 py-3 text-xs font-black uppercase tracking-widest transition-all ${sessionActive ? 'bg-rose-950/40 border border-rose-900/50 text-rose-400 hover:bg-rose-900/40' : 'bg-emerald-600 hover:bg-emerald-500 text-white shadow-lg shadow-emerald-900/20'}`}>
                  {sessionActive ? <Square size={14} /> : <Activity size={14} />}
                  {sessionActive ? 'End Session' : 'Start Session'}
                </button>
              </div>
           </div>
        </section>

        {/* RIGHT COLUMN: TRANSCRIPTS & TEXT FALLBACK */}
        <section className="xl:col-span-4 flex flex-col space-y-6">
          <div className="bg-slate-900/40 border border-slate-800 rounded-[2rem] p-6 shadow-sm flex-1 flex flex-col">
            <div className="flex items-center justify-between mb-5">
              <p className="text-[10px] uppercase font-black tracking-[0.2em] text-cyan-400">Therapeutic Dialogue</p>
              {isProcessing && <Loader2 size={14} className="text-cyan-400 animate-spin" />}
            </div>

            <div className="mb-4 rounded-xl border border-slate-800 bg-slate-950/50 p-4">
              <p className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-1">System Status</p>
              <p className="text-sm font-medium text-slate-200">{statusText}</p>
            </div>

            <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar min-h-[300px]">
              {conversationTurns.length === 0 && (
                <div className="flex flex-col items-center justify-center h-full text-slate-500 space-y-3 py-10">
                  <MessageCircle size={32} className="opacity-50" />
                  <p className="text-xs font-black uppercase tracking-widest text-center">No dialogue established.<br/>Start a session and press speak.</p>
                </div>
              )}

              {conversationTurns.map((turn) => (
                <div key={turn.id} className="space-y-3">
                  <div className="rounded-2xl bg-slate-950 border border-slate-800 p-4 ml-8 relative shadow-sm">
                    <div className="absolute -left-3 top-4 w-6 h-6 bg-slate-800 rounded-full border-4 border-slate-950 flex items-center justify-center"><span className="text-[8px] font-black text-slate-400">YOU</span></div>
                    <div className="flex justify-between items-start mb-2">
                      <p className="text-[9px] font-black uppercase tracking-widest text-slate-500">{turn.time} • {turn.source}</p>
                    </div>
                    <p className="text-sm text-slate-200 leading-relaxed">{turn.userText}</p>
                  </div>

                  <div className="rounded-2xl bg-indigo-950/10 border border-indigo-900/30 p-4 mr-8 relative shadow-sm">
                    <div className="absolute -right-3 top-4 w-6 h-6 bg-indigo-600 rounded-full border-4 border-slate-950 flex items-center justify-center"><Activity size={10} className="text-white" /></div>
                    <div className="flex justify-between items-start mb-2">
                      <p className="text-[9px] font-black uppercase tracking-widest text-indigo-400">SERENITY AI</p>
                      <span className="text-[8px] font-black uppercase tracking-widest text-slate-500 border border-slate-800 px-1.5 py-0.5 rounded">Affect: {turn.emotion}</span>
                    </div>
                    <p className="text-sm text-slate-200 leading-relaxed">{turn.assistantReply}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* TEXT FALLBACK INPUT */}
            <div className="mt-4 pt-4 border-t border-slate-800">
              <div className="relative">
                <textarea
                  value={promptText}
                  onChange={(e) => setPromptText(e.target.value)}
                  placeholder="Type a message manually if you prefer..."
                  className="w-full rounded-2xl bg-slate-950 border border-slate-700/50 pl-4 pr-12 py-3 text-sm text-slate-200 focus:outline-none focus:border-indigo-500 transition-colors resize-none custom-scrollbar"
                  rows={2}
                />
                <button
                  onClick={sendTextPrompt}
                  disabled={!sessionActive || isRecording || isProcessing || !promptText.trim()}
                  className="absolute right-2 bottom-2 p-2 bg-indigo-600 text-white rounded-xl hover:bg-indigo-500 disabled:opacity-50 transition-colors shadow-md"
                >
                  <Send size={16} />
                </button>
              </div>
            </div>

            {notices.length > 0 && (
              <div className="mt-4 rounded-xl border border-amber-900/50 bg-amber-950/20 p-3 text-amber-200 text-xs font-medium space-y-1">
                {notices.map((notice, index) => (
                  <p key={`${notice}-${index}`} className="flex items-start gap-2"><AlertTriangle size={14} className="shrink-0 mt-0.5" /> {notice}</p>
                ))}
              </div>
            )}
          </div>
        </section>

      </main>
    </div>
  );
};

export default UnifiedEmotionPage;
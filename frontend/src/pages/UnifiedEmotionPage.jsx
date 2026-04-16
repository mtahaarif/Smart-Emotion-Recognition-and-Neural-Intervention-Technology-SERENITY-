import React, { useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import {
  Activity,
  ArrowLeft,
  LogOut,
  Mic,
  Square,
  Camera,
  MessageCircle,
  Send,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
const SHOW_PROVISIONAL_ASSISTANT_TEXT = (import.meta.env.VITE_SHOW_PROVISIONAL_ASSISTANT_TEXT || 'true').toLowerCase() === 'true';

const formatTime = (date = new Date()) => date.toLocaleTimeString();

const stripStarredSegments = (value, preserveEdges = false) => {
  const source = String(value || '');
  if (!source) {
    return '';
  }

  const starIndex = source.indexOf('*');
  const cleaned = (starIndex === -1 ? source : source.slice(0, starIndex))
    .replace(/[ \t]+/g, ' ')
    .replace(/\s+([,.;:!?])/g, '$1');
  return preserveEdges ? cleaned : cleaned.trim();
};

const cleanStreamToken = (token, reachedFirstAsterisk) => {
  if (reachedFirstAsterisk) {
    return { cleanToken: '', reachedFirstAsterisk: true };
  }

  const source = String(token || '');
  if (!source) {
    return { cleanToken: '', reachedFirstAsterisk: false };
  }

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

const UnifiedEmotionPage = ({ user, onLogout }) => {
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

  const greetingLine = useMemo(() => `Patient: ${user}`, [user]);

  useEffect(() => {
    sessionActiveRef.current = sessionActive;
  }, [sessionActive]);

  const pushNotice = (message) => {
    if (!message) return;
    setNotices((prev) => [message, ...prev.filter((item) => item !== message)].slice(0, 4));
  };

  const addEmotionLog = (emotion) => {
    const normalized = (emotion || 'Neutral').toLowerCase();
    setLogs((prev) => [
      { time: formatTime(), emotion: normalized },
      ...prev,
    ].slice(0, 20));

    setEmotionCounts((prev) => ({
      ...prev,
      [normalized]: (prev[normalized] || 0) + 1,
    }));
  };

  const appendConversationTurn = ({ userText, assistantReply, emotion, source }) => {
    const timestamp = formatTime();
    const turnId = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    setConversationTurns((prev) => [
      ...prev,
      {
        id: turnId,
        time: timestamp,
        userText: userText || '(no text recognized)',
        assistantReply: assistantReply || '',
        emotion: (emotion || 'Neutral').toLowerCase(),
        source,
      },
    ]);
    return turnId;
  };

  const reconcileLatestEmotionLog = (fromEmotion, toEmotion) => {
    const from = String(fromEmotion || '').trim().toLowerCase();
    const to = String(toEmotion || '').trim().toLowerCase();
    if (!from || !to || from === to) {
      return;
    }

    setLogs((prev) => {
      if (prev.length === 0) {
        return prev;
      }
      const [head, ...rest] = prev;
      return [{ ...head, emotion: to }, ...rest];
    });

    setEmotionCounts((prev) => {
      const next = { ...prev };
      if (next[from]) {
        next[from] = Math.max(0, next[from] - 1);
        if (next[from] === 0) {
          delete next[from];
        }
      }
      next[to] = (next[to] || 0) + 1;
      return next;
    });
  };

  const updateConversationTurn = (turnId, updates) => {
    if (!turnId) return;
    setConversationTurns((prev) =>
      prev.map((turn) =>
        turn.id === turnId
          ? {
              ...turn,
              ...updates,
              emotion: updates?.emotion ? String(updates.emotion).toLowerCase() : turn.emotion,
            }
          : turn
      )
    );
  };

  const flushStreamUiUpdate = () => {
    streamUiFrameRef.current = null;
    const nextText = String(streamUiPendingTextRef.current || '');
    const nextTurnId = streamUiPendingTurnRef.current;
    setAssistantText(nextText);
    if (nextTurnId) {
      updateConversationTurn(nextTurnId, { assistantReply: nextText });
    }
  };

  const scheduleStreamUiUpdate = (text, turnId = null) => {
    streamUiPendingTextRef.current = String(text || '');
    if (turnId) {
      streamUiPendingTurnRef.current = turnId;
    }

    if (streamUiFrameRef.current !== null) {
      return;
    }

    if (typeof window !== 'undefined' && typeof window.requestAnimationFrame === 'function') {
      streamUiFrameRef.current = window.requestAnimationFrame(flushStreamUiUpdate);
      return;
    }

    streamUiFrameRef.current = setTimeout(flushStreamUiUpdate, 16);
  };

  const cancelStreamUiUpdate = () => {
    if (streamUiFrameRef.current === null) {
      return;
    }

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
    if (!nextAudioBase64) {
      playNextStreamAudioSegment();
      return;
    }

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

    if (streamAudioElementRef.current) {
      streamAudioElementRef.current.pause();
      streamAudioElementRef.current = null;
    }

    if (typeof window !== 'undefined' && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }
  };

  const enqueueStreamAudioSegment = (audioBase64, sequence = null) => {
    const normalized = String(audioBase64 || '').trim();
    if (!normalized) {
      return;
    }

    const parsedSequence = Number(sequence);
    streamAudioQueueRef.current.push({
      audioBase64: normalized,
      sequence: Number.isFinite(parsedSequence) ? parsedSequence : null,
    });
    if (!streamAudioActiveRef.current) {
      playNextStreamAudioSegment();
    }
  };

  const trimStreamAudioPlayback = (maxSequence) => {
    const parsedMax = Number(maxSequence);
    if (!Number.isFinite(parsedMax)) {
      return;
    }

    streamAudioQueueRef.current = streamAudioQueueRef.current.filter((segment) => {
      const segmentSequence = Number(segment.sequence);
      if (!Number.isFinite(segmentSequence)) {
        return true;
      }
      return segmentSequence <= parsedMax;
    });

    const currentSequence = streamAudioCurrentSequenceRef.current;
    if (Number.isFinite(currentSequence) && currentSequence > parsedMax) {
      if (streamAudioElementRef.current) {
        streamAudioElementRef.current.pause();
        streamAudioElementRef.current = null;
      }
      streamAudioActiveRef.current = false;
      streamAudioCurrentSequenceRef.current = null;
      if (streamAudioQueueRef.current.length > 0) {
        playNextStreamAudioSegment();
      }
    }
  };

  const toFriendlyNotices = (rawErrors = []) => {
    const mapped = rawErrors.map((error) => {
      const lower = String(error).toLowerCase();
      if (lower.includes('rag model unavailable')) {
        return 'RAG model unavailable. Serenity is using a safe fallback reply.';
      }
      if (lower.includes('rag initialization failed')) {
        return 'RAG initialization failed. Serenity used a safe fallback response.';
      }
      if (lower.includes('llm generation failed') || lower.includes('llm timeout')) {
        return 'LLM response used a safety fallback for this turn. Please retry if needed.';
      }
      if (
        lower.includes('tts failed') ||
        lower.includes('tts timeout') ||
        lower.includes('tts service unavailable') ||
        lower.includes('tts temporarily disabled') ||
        lower.includes('tts disabled by configuration')
      ) {
        // TTS issues are handled by browser speech fallback where available.
        return null;
      }
      if (lower.includes('audio upload was empty')) {
        return 'No speech was captured. Please speak clearly before pressing Stop.';
      }
      return String(error);
    });

    return [...new Set(mapped.filter(Boolean))].slice(0, 4);
  };

  const playTtsIfAvailable = async (payload) => {
    const segments = Array.isArray(payload?.tts_audio_segments_base64)
      ? payload.tts_audio_segments_base64.filter(Boolean)
      : [];

    if (segments.length > 0) {
      try {
        for (const segment of segments) {
          // Play each sentence chunk in order; chunks were synthesized while LLM was still generating.
          const audio = new Audio(`data:audio/mpeg;base64,${segment}`);
          await new Promise((resolve, reject) => {
            audio.onended = () => resolve();
            audio.onerror = () => reject(new Error('Audio playback failed.'));
            audio.play().catch(reject);
          });
        }
        return;
      } catch {
        pushNotice('Audio playback was blocked by browser policy.');
      }
    }

    if (payload?.tts_audio_base64) {
      try {
        const audio = new Audio(`data:audio/mpeg;base64,${payload.tts_audio_base64}`);
        await audio.play();
        return;
      } catch {
        pushNotice('Audio playback was blocked by browser policy.');
      }
    }

    // Browser fallback when backend TTS is unavailable.
    const fallbackText = hardClean(payload?.llm_response || '');
    if (!fallbackText || typeof window === 'undefined' || !window.speechSynthesis) {
      return;
    }

    try {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(fallbackText);
      utterance.lang = 'en-US';
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    } catch {
      // Browser speech fallback is best-effort only.
    }
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
      updateConversationTurn(existingTurnId, {
        userText,
        assistantReply: cleanedAssistantReply,
        emotion: payload.dominant_emotion,
      });
    } else {
      appendConversationTurn({
        userText,
        assistantReply: cleanedAssistantReply,
        emotion: payload.dominant_emotion,
        source,
      });
    }

    setNotices(toFriendlyNotices(payload.errors || []));
    await playTtsIfAvailable(payload);
  };

  const stopStreams = () => {
    if (videoStreamRef.current) {
      videoStreamRef.current.getTracks().forEach((track) => track.stop());
      videoStreamRef.current = null;
    }

    if (audioStreamRef.current) {
      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsCameraOn(false);
    setIsMicOn(false);
  };

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      cancelStreamUiUpdate();
      resetStreamAudioPlayback();
      stopStreams();
    };
  }, []);

  const streamNdjson = async ({ url, method = 'POST', headers = {}, body, onEvent }) => {
    const response = await fetch(url, {
      method,
      headers,
      body,
    });

    if (!response.ok) {
      let detail = `Request failed with status ${response.status}`;
      try {
        const text = await response.text();
        if (text) {
          try {
            const payload = JSON.parse(text);
            detail = payload?.detail || payload?.error || detail;
          } catch {
            detail = text;
          }
        }
      } catch {
        // Ignore body parsing error and keep status-based detail.
      }
      throw new Error(detail);
    }

    if (!response.body) {
      throw new Error('Streaming response body is not available.');
    }

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

        try {
          const event = JSON.parse(trimmed);
          // eslint-disable-next-line no-await-in-loop
          await onEvent(event);
        } catch {
          // Ignore malformed chunks and continue parsing subsequent events.
        }
      }
    }

    const tail = buffer.trim();
    if (tail) {
      try {
        const event = JSON.parse(tail);
        await onEvent(event);
      } catch {
        // Ignore malformed trailing chunk.
      }
    }
  };

  const ensureVideoStream = async () => {
    if (!cameraEnabled) {
      return null;
    }

    if (videoStreamRef.current && videoStreamRef.current.getVideoTracks().length > 0) {
      return videoStreamRef.current;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
    videoStreamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
    }

    setIsCameraOn(true);
    return stream;
  };

  const ensureAudioStream = async () => {
    if (audioStreamRef.current && audioStreamRef.current.getAudioTracks().length > 0) {
      return audioStreamRef.current;
    }

    const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
    audioStreamRef.current = stream;
    setIsMicOn(true);
    return stream;
  };

  const captureSnapshot = () => {
    if (!videoRef.current || !canvasRef.current) {
      return '';
    }

    if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      return '';
    }

    const ctx = canvasRef.current.getContext('2d');
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    return canvasRef.current.toDataURL('image/jpeg');
  };

  const buildRecorder = (stream) => {
    const preferredTypes = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4'];
    const selectedType = preferredTypes.find((type) => MediaRecorder.isTypeSupported(type));

    if (selectedType) {
      return new MediaRecorder(stream, { mimeType: selectedType });
    }

    return new MediaRecorder(stream);
  };

  const submitInteraction = async ({ imageBase64 = '', audioBlob = null, textFallback = '', existingTurnId = null }) => {
    const formData = new FormData();
    formData.append('username', user || 'anonymous');

    if (imageBase64) {
      formData.append('image', imageBase64);
    }
    if (textFallback) {
      formData.append('user_message', textFallback);
    }
    if (audioBlob) {
      formData.append('file', audioBlob, 'interaction.webm');
    }

    const response = await axios.post(`${API_BASE_URL}/api/interact`, formData, { timeout: 120000 });
    await updateFromPayload(response.data, textFallback, 'voice', existingTurnId);
  };

  const submitInteractionStream = async ({ imageBase64 = '', audioBlob = null, textFallback = '' }) => {
    const formData = new FormData();
    formData.append('username', user || 'anonymous');
    if (imageBase64) {
      formData.append('image', imageBase64);
    }
    if (textFallback) {
      formData.append('user_message', textFallback);
    }
    if (audioBlob) {
      formData.append('file', audioBlob, 'interaction.webm');
    }

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
        const eventType = String(event?.type || '').toLowerCase();

        if (eventType === 'transcription') {
          const text = String(event?.text || '').trim();
          setTranscription(text);
          if (!turnId && text) {
            turnId = appendConversationTurn({
              userText: text,
              assistantReply: '',
              emotion: dominantEmotion,
              source: 'voice',
            });
          } else if (turnId && text) {
            updateConversationTurn(turnId, { userText: text });
          }
          return;
        }

        if (eventType === 'user_text') {
          const text = String(event?.text || textFallback || '').trim();
          setTranscription(text);
          if (!turnId) {
            turnId = appendConversationTurn({
              userText: text,
              assistantReply: '',
              emotion: dominantEmotion,
              source: 'voice',
            });
          } else {
            updateConversationTurn(turnId, { userText: text });
          }
          return;
        }

        if (eventType === 'emotion_partial') {
          const provisional = String(event?.speech_emotion || event?.face_emotion || '').trim();
          if (event?.speech_emotion) {
            setSpeechEmotion(String(event.speech_emotion));
          }
          if (event?.face_emotion) {
            setFaceEmotion(String(event.face_emotion));
          }

          if (provisional && !emotionLoggedForTurn) {
            addEmotionLog(provisional);
            provisionalEmotion = provisional;
            emotionLoggedForTurn = true;
            if (turnId) {
              updateConversationTurn(turnId, { emotion: provisional });
            }
          }
          return;
        }

        if (eventType === 'emotion') {
          const dominant = String(event?.dominant_emotion || 'Neutral');
          setDominantEmotion(dominant);
          setSpeechEmotion(String(event?.speech_emotion || 'Neutral'));
          setFaceEmotion(String(event?.face_emotion || 'Neutral'));
          if (!emotionLoggedForTurn) {
            addEmotionLog(dominant);
            emotionLoggedForTurn = true;
          } else {
            reconcileLatestEmotionLog(provisionalEmotion, dominant);
          }
          if (turnId) {
            updateConversationTurn(turnId, { emotion: dominant });
          }
          return;
        }

        if (eventType === 'assistant_delta') {
          let rawToken = String(event?.delta || '');
          if (!rawToken) {
            const candidateText = String(event?.text || '');
            if (candidateText) {
              rawToken = candidateText.startsWith(liveAssistantText)
                ? candidateText.slice(liveAssistantText.length)
                : candidateText;
            }
          }

          const { cleanToken, reachedFirstAsterisk } = cleanStreamToken(rawToken, streamReachedFirstAsteriskRef.current);
          streamReachedFirstAsteriskRef.current = reachedFirstAsterisk;
          if (!cleanToken) {
            return;
          }

          const isFirstVisibleToken = !liveAssistantText;
          liveAssistantText += cleanToken;
          if (SHOW_PROVISIONAL_ASSISTANT_TEXT) {
            if (!turnId) {
              turnId = appendConversationTurn({
                userText: textFallback || '(processing...)',
                assistantReply: liveAssistantText,
                emotion: dominantEmotion,
                source: 'voice',
              });
            }
            if (isFirstVisibleToken) {
              cancelStreamUiUpdate();
              setAssistantText(liveAssistantText);
              updateConversationTurn(turnId, { assistantReply: liveAssistantText });
            } else {
              scheduleStreamUiUpdate(liveAssistantText, turnId);
            }
          }
          return;
        }

        if (eventType === 'assistant_replace') {
          liveAssistantText = hardClean(String(event?.text || liveAssistantText));
          streamReachedFirstAsteriskRef.current = false;
          if (liveAssistantText) {
            cancelStreamUiUpdate();
            setAssistantText(liveAssistantText);
            if (turnId) {
              updateConversationTurn(turnId, { assistantReply: liveAssistantText });
            }
          }
          return;
        }

        if (eventType === 'assistant_tts_reset') {
          resetStreamAudioPlayback();
          return;
        }

        if (eventType === 'assistant_tts_trim') {
          trimStreamAudioPlayback(event?.max_sequence);
          return;
        }

        if (eventType === 'assistant_sentence') {
          // Sentence text is emitted for visual streaming. Audio is handled by assistant_sentence_tts.
          return;
        }

        if (eventType === 'assistant_sentence_tts') {
          enqueueStreamAudioSegment(event?.audio_base64, event?.sequence);
          return;
        }

        if (eventType === 'error') {
          const friendly = toFriendlyNotices([event?.message]).filter(Boolean);
          if (friendly.length > 0) {
            friendly.forEach((message) => pushNotice(message));
          } else if (event?.message) {
            pushNotice(String(event.message));
          }
          return;
        }

        if (eventType === 'final') {
          const finalReply = hardClean(event?.llm_response || liveAssistantText || '');
          streamReachedFirstAsteriskRef.current = false;
          const finalText = String(event?.transcription || textFallback || '').trim();
          setDominantEmotion(String(event?.dominant_emotion || 'Neutral'));
          setSpeechEmotion(String(event?.speech_emotion || 'Neutral'));
          setFaceEmotion(String(event?.face_emotion || 'Neutral'));
          setTranscription(finalText);
          cancelStreamUiUpdate();
          setAssistantText(finalReply);

          if (!turnId) {
            turnId = appendConversationTurn({
              userText: finalText,
              assistantReply: finalReply,
              emotion: String(event?.dominant_emotion || 'Neutral'),
              source: 'voice',
            });
          } else {
            updateConversationTurn(turnId, {
              userText: finalText,
              assistantReply: finalReply,
              emotion: String(event?.dominant_emotion || 'Neutral'),
            });
          }

          if (event?.tts_audio_base64) {
            enqueueStreamAudioSegment(event.tts_audio_base64, Number.MAX_SAFE_INTEGER);
          }
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
    streamAudioQueueRef.current = [];
    streamAudioActiveRef.current = false;
    if (streamAudioElementRef.current) {
      streamAudioElementRef.current.pause();
      streamAudioElementRef.current = null;
    }
    setStatusText('Session is active. Press Speak to start a conversation.');
  };

  const closeSessionNow = () => {
    endSessionAfterUnitRef.current = false;
    setSessionActive(false);
    setIsRecording(false);
    setIsProcessing(false);
    streamReachedFirstAsteriskRef.current = false;
    cancelStreamUiUpdate();
    streamAudioQueueRef.current = [];
    streamAudioActiveRef.current = false;
    if (streamAudioElementRef.current) {
      streamAudioElementRef.current.pause();
      streamAudioElementRef.current = null;
    }
    stopStreams();
    setStatusText('Session closed. Start Session to begin a new chat.');
  };

  const handleSessionToggle = () => {
    if (!sessionActive) {
      startSession();
      return;
    }

    if (isRecording) {
      endSessionAfterUnitRef.current = true;
      setStatusText('Finishing this conversation, then closing session...');
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }
      return;
    }

    closeSessionNow();
  };

  const startVoiceUnit = async () => {
    if (!sessionActive) {
      pushNotice('Please press Start Session first.');
      return;
    }

    if (isRecording || isProcessing) {
      return;
    }

    setNotices([]);
    setStatusText('Recording your voice... Press Stop to finish this conversation.');

    try {
      const stream = await ensureAudioStream();
      let captureWithCamera = false;

      if (cameraEnabled) {
        try {
          await ensureVideoStream();
          captureWithCamera = true;
        } catch (videoError) {
          captureWithCamera = false;
          setIsCameraOn(false);
          pushNotice(`Camera unavailable. Continuing in voice-only mode: ${videoError.message || videoError}`);
        }
      }

      const recorder = buildRecorder(stream);
      mediaRecorderRef.current = recorder;
      chunksRef.current = [];

      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      recorder.onerror = (event) => {
        setIsRecording(false);
        pushNotice(event.error?.message || 'Microphone recording failed.');
      };

      recorder.onstop = async () => {
        setIsRecording(false);
        setIsProcessing(true);
        setStatusText('Reflecting on this conversation...');

        const audioBlob = new Blob(chunksRef.current, { type: chunksRef.current[0]?.type || 'audio/webm' });
        let streamedTurnId = null;
        chunksRef.current = [];

        try {
          if (!audioBlob || audioBlob.size === 0) {
            pushNotice('No speech was recorded. Please speak and try again.');
          } else {
            const imageBase64 = captureWithCamera ? captureSnapshot() : '';
            setStatusText('Listening and understanding your feelings...');
            streamedTurnId = await submitInteractionStream({
              imageBase64,
              audioBlob,
              textFallback: promptText.trim(),
            });
            setStatusText('Conversation complete. Press Speak to continue.');
          }
        } catch (error) {
          // Fall back to legacy endpoint if stream path is unavailable.
          try {
            const imageBase64 = captureWithCamera ? captureSnapshot() : '';
            await submitInteraction({
              imageBase64,
              audioBlob,
              textFallback: promptText.trim(),
              existingTurnId: streamedTurnId || null,
            });
            setStatusText('Conversation complete. Press Speak to continue.');
          } catch (fallbackError) {
            const backendMessage =
              fallbackError.response?.data?.detail ||
              fallbackError.response?.data?.error ||
              (fallbackError.request ? 'Backend unreachable. Ensure API is running on http://127.0.0.1:5000' : null) ||
              error.message ||
              'Could not complete this conversation. Please try again.';
            pushNotice(backendMessage);
          }
        } finally {
          setIsProcessing(false);

          if (endSessionAfterUnitRef.current) {
            closeSessionNow();
          } else if (sessionActiveRef.current) {
            setStatusText('Conversation complete. Press Speak to continue.');
          }
        }
      };

      recorder.start();
      setIsRecording(true);
    } catch (error) {
      setIsMicOn(false);
      pushNotice(`Microphone unavailable: ${error.message || error}`);
      setStatusText('Microphone is required. Please allow mic access and try again.');
    }
  };

  const stopVoiceUnit = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const handleSpeakToggle = async () => {
    if (isRecording) {
      stopVoiceUnit();
      return;
    }

    await startVoiceUnit();
  };

  const toggleVision = async () => {
    if (isRecording || isProcessing) {
      pushNotice('Cannot change vision mode while a conversation is active.');
      return;
    }

    if (cameraEnabled) {
      setCameraEnabled(false);
      if (videoStreamRef.current) {
        videoStreamRef.current.getTracks().forEach((track) => track.stop());
        videoStreamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }
      setIsCameraOn(false);
      setStatusText('Vision disabled. Voice-only mode active.');
      return;
    }

    setCameraEnabled(true);

    try {
      await ensureVideoStream();
      setStatusText('Vision enabled. Facial + speech fusion is active.');
    } catch (error) {
      setCameraEnabled(false);
      setIsCameraOn(false);
      pushNotice(`Unable to enable vision: ${error.message || error}`);
      setStatusText('Vision unavailable. Remaining in voice-only mode.');
    }
  };

  const sendTextPrompt = async () => {
    const message = promptText.trim();
    let turnId = null;

    if (!sessionActive) {
      pushNotice('Start a session first to send a message.');
      return;
    }

    if (!message || isRecording || isProcessing) {
      return;
    }

    setNotices([]);
    setIsProcessing(true);
    setStatusText('Sending your message to SERENITY...');

    try {
      turnId = appendConversationTurn({
        userText: message,
        assistantReply: '',
        emotion: dominantEmotion,
        source: 'text',
      });
      let liveAssistantText = '';

      resetStreamAudioPlayback();
      streamReachedFirstAsteriskRef.current = false;

      await streamNdjson({
        url: `${API_BASE_URL}/api/chat/stream`,
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          username: user || 'anonymous',
          message,
        }),
        onEvent: async (event) => {
          const eventType = String(event?.type || '').toLowerCase();

          if (eventType === 'emotion') {
            const dominant = String(event?.dominant_emotion || 'Neutral');
            setDominantEmotion(dominant);
            setSpeechEmotion(String(event?.speech_emotion || 'Neutral'));
            setFaceEmotion(String(event?.face_emotion || 'Neutral'));
            addEmotionLog(dominant);
            updateConversationTurn(turnId, { emotion: dominant });
            return;
          }

          if (eventType === 'assistant_delta') {
            let rawToken = String(event?.delta || '');
            if (!rawToken) {
              const candidateText = String(event?.text || '');
              if (candidateText) {
                rawToken = candidateText.startsWith(liveAssistantText)
                  ? candidateText.slice(liveAssistantText.length)
                  : candidateText;
              }
            }

            const { cleanToken, reachedFirstAsterisk } = cleanStreamToken(rawToken, streamReachedFirstAsteriskRef.current);
            streamReachedFirstAsteriskRef.current = reachedFirstAsterisk;
            if (!cleanToken) {
              return;
            }

            const isFirstVisibleToken = !liveAssistantText;
            liveAssistantText += cleanToken;
            if (SHOW_PROVISIONAL_ASSISTANT_TEXT) {
              if (isFirstVisibleToken) {
                cancelStreamUiUpdate();
                setAssistantText(liveAssistantText);
                updateConversationTurn(turnId, { assistantReply: liveAssistantText });
              } else {
                scheduleStreamUiUpdate(liveAssistantText, turnId);
              }
            }
            return;
          }

          if (eventType === 'assistant_replace') {
            liveAssistantText = hardClean(String(event?.text || liveAssistantText));
            streamReachedFirstAsteriskRef.current = false;
            if (liveAssistantText) {
              cancelStreamUiUpdate();
              setAssistantText(liveAssistantText);
              updateConversationTurn(turnId, { assistantReply: liveAssistantText });
            }
            return;
          }

          if (eventType === 'assistant_tts_reset') {
            resetStreamAudioPlayback();
            return;
          }

          if (eventType === 'assistant_tts_trim') {
            trimStreamAudioPlayback(event?.max_sequence);
            return;
          }

          if (eventType === 'assistant_sentence') {
            return;
          }

          if (eventType === 'assistant_sentence_tts') {
            enqueueStreamAudioSegment(event?.audio_base64, event?.sequence);
            return;
          }

          if (eventType === 'error') {
            const friendly = toFriendlyNotices([event?.message]).filter(Boolean);
            if (friendly.length > 0) {
              friendly.forEach((notice) => pushNotice(notice));
            } else if (event?.message) {
              pushNotice(String(event.message));
            }
            return;
          }

          if (eventType === 'final') {
            const finalReply = hardClean(event?.llm_response || liveAssistantText || '');
            streamReachedFirstAsteriskRef.current = false;
            cancelStreamUiUpdate();
            setTranscription(message);
            setAssistantText(finalReply);
            updateConversationTurn(turnId, {
              userText: message,
              assistantReply: finalReply,
              emotion: String(event?.dominant_emotion || 'Neutral'),
            });
            if (event?.tts_audio_base64) {
              enqueueStreamAudioSegment(event.tts_audio_base64, Number.MAX_SAFE_INTEGER);
            }
            setStatusText('Message complete. Press Speak or send another message.');
          }
        },
      });

      setPromptText('');
    } catch (error) {
      // Fall back to legacy endpoint when stream endpoint is unavailable.
      try {
        const response = await axios.post(
          `${API_BASE_URL}/api/chat`,
          {
            username: user || 'anonymous',
            message,
          },
          { timeout: 120000 }
        );

        await updateFromPayload(response.data, message, 'text', turnId);
        setPromptText('');
        setStatusText('Message complete. Press Speak or send another message.');
      } catch (fallbackError) {
        const backendMessage =
          fallbackError.response?.data?.detail ||
          fallbackError.response?.data?.error ||
          (fallbackError.request ? 'Backend unreachable. Ensure API is running on http://127.0.0.1:5000' : null) ||
          error.message ||
          'Message sending failed. Please try again.';
        pushNotice(backendMessage);
        setStatusText('Message failed.');
      }
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col">
      <nav className="border-b border-cyan-900/50 px-8 py-4 flex justify-between items-center sticky top-0 z-50 bg-slate-950/95 backdrop-blur">
        <div className="flex items-center gap-4">
          <Link
            to="/dashboard"
            className="inline-flex items-center gap-2 text-slate-300 hover:text-white font-medium"
          >
            <ArrowLeft size={18} /> Back
          </Link>
          <h1 className="text-2xl font-bold text-cyan-300 flex items-center gap-2">
            <Activity className="text-cyan-300" /> SERENITY Support Session
          </h1>
        </div>

        <div className="flex items-center gap-4">
          <span className="text-slate-400">{greetingLine}</span>
          <button
            onClick={onLogout}
            className="text-rose-400 font-medium hover:text-rose-300 flex items-center gap-1"
          >
            <LogOut size={18} /> Logout
          </button>
        </div>
      </nav>

      <main className="flex-1 p-6">
        <div className="max-w-7xl mx-auto grid grid-cols-1 xl:grid-cols-12 gap-4">
          <section className="xl:col-span-2 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Client</h3>
            <p className="text-sm">Name: <b>{user}</b></p>
            <p className="text-sm text-emerald-300">Status: {sessionActive ? 'Session Active' : 'Session Closed'}</p>
            <p className="text-sm text-sky-300">Mic: {isMicOn ? 'Online' : 'Offline'}</p>
            <p className="text-sm text-amber-300">Session Style: {cameraEnabled ? 'Voice + Camera Support' : 'Voice Support (Mic required)'}</p>
            <p className="text-sm text-violet-300">Conversation: {isRecording ? 'Listening' : isProcessing ? 'Reflecting' : 'Ready'}</p>

            <div className="mt-4 border-t border-cyan-900/40 pt-3">
              <h4 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Emotion Check-In Log</h4>
              <div className="space-y-2 max-h-[260px] overflow-auto">
                {logs.length === 0 && <p className="text-xs text-slate-500">No conversation entries recorded in this session.</p>}
                {logs.map((entry, index) => (
                  <div key={`${entry.time}-${index}`} className="text-xs text-slate-300 flex justify-between">
                    <span>[{entry.time}]</span>
                    <span className="text-cyan-300 font-semibold">{entry.emotion}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="mt-4 border-t border-cyan-900/40 pt-3">
              <h4 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Emotional Pattern Summary</h4>
              <div className="space-y-1 text-xs">
                {Object.keys(emotionCounts).length === 0 && <p className="text-slate-500">No emotion counts yet.</p>}
                {Object.entries(emotionCounts)
                  .sort((a, b) => b[1] - a[1])
                  .map(([emotion, count]) => (
                    <div key={emotion} className="flex justify-between text-slate-300">
                      <span>{emotion}</span>
                      <span className="text-cyan-300 font-semibold">x{count}</span>
                    </div>
                  ))}
              </div>
            </div>
          </section>

          <section className="xl:col-span-7 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Emotional Signals</h3>
            <div className="relative rounded-xl overflow-hidden border border-cyan-950 bg-black aspect-video">
              <video ref={videoRef} autoPlay muted playsInline className="w-full h-full object-cover" />
              <canvas ref={canvasRef} className="hidden" />
              {!isCameraOn && (
                <div className="absolute inset-0 flex items-center justify-center text-slate-300">Camera Offline</div>
              )}
              <div className="absolute top-3 left-3 bg-slate-900/80 px-3 py-2 rounded-lg border border-cyan-800">
                <p className="text-[10px] uppercase tracking-widest text-slate-300">Emotion</p>
                <p className="text-2xl font-bold text-cyan-300">{dominantEmotion.toLowerCase()}</p>
                <p className="text-xs text-slate-300">speech: {speechEmotion.toLowerCase()} | face: {faceEmotion.toLowerCase()}</p>
              </div>
            </div>

            <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-3">
              <button
                onClick={toggleVision}
                disabled={isRecording || isProcessing}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-700 hover:bg-cyan-600 px-4 py-2 font-semibold disabled:opacity-60"
              >
                <Camera size={16} /> {cameraEnabled ? 'Disable Vision' : 'Enable Vision'}
              </button>

              <button
                onClick={handleSpeakToggle}
                disabled={!sessionActive || isProcessing}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-700 hover:bg-indigo-600 px-4 py-2 font-semibold disabled:opacity-60"
              >
                {isRecording ? <Square size={16} /> : <Mic size={16} />}
                {isRecording ? 'Stop' : 'Speak'}
              </button>

              <button
                onClick={handleSessionToggle}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-rose-700 hover:bg-rose-600 px-4 py-2 font-semibold"
              >
                <Square size={16} /> {sessionActive ? 'Stop Session' : 'Start Session'}
              </button>
            </div>

            <div className="mt-3 rounded-lg border border-cyan-800/60 bg-slate-900 p-3 space-y-2">
              <p className="text-cyan-300 font-semibold">Share Your Thoughts</p>
              <textarea
                value={promptText}
                onChange={(e) => setPromptText(e.target.value)}
                placeholder="Type how you are feeling or what you want to talk about"
                className="w-full rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
                rows={3}
              />
              <button
                onClick={sendTextPrompt}
                disabled={!sessionActive || isRecording || isProcessing || !promptText.trim()}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 px-4 py-2 font-semibold disabled:opacity-60"
              >
                <Send size={14} /> Send Message to SERENITY
              </button>
            </div>

            {notices.length > 0 && (
              <div className="mt-3 rounded-lg border border-amber-600/60 bg-amber-950/30 p-3 text-amber-200 text-sm">
                {notices.map((notice, index) => (
                  <p key={`${notice}-${index}`}>{notice}</p>
                ))}
              </div>
            )}
          </section>

          <section className="xl:col-span-3 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Support Dialogue</h3>

            <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3 mb-3">
              <p className="text-cyan-300 font-semibold">Session Guide</p>
              <p className="text-slate-300">{statusText}</p>
            </div>

            <div className="max-h-[560px] overflow-auto space-y-3 pr-1">
              {conversationTurns.length === 0 && (
                <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3">
                  <p className="text-slate-500">No conversation yet. Start a session and press Speak.</p>
                </div>
              )}

              {conversationTurns.map((turn) => (
                <div key={turn.id} className="space-y-2">
                  <div className="rounded-lg bg-slate-950 border border-cyan-900/60 p-2">
                    <p className="text-xs text-slate-400">You • {turn.time} • {turn.source}</p>
                    <p className="text-sm text-white mt-1">{turn.userText}</p>
                  </div>

                  <div className="rounded-lg bg-slate-950 border border-emerald-900/60 p-2">
                    <p className="text-xs text-emerald-400 inline-flex items-center gap-1">
                      <MessageCircle size={12} /> Serenity • emotion: {turn.emotion}
                    </p>
                    <p className="text-sm text-white mt-1">{turn.assistantReply}</p>
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-4 rounded-lg border border-cyan-800/60 bg-slate-900 p-3">
              <p className="text-slate-300">Latest message from you:</p>
              <p className="text-white">{transcription || '(awaiting speech input...)'}</p>
              <p className="text-cyan-300 mt-2 font-semibold">Latest reply from SERENITY:</p>
              <p className="text-white">{assistantText}</p>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default UnifiedEmotionPage;

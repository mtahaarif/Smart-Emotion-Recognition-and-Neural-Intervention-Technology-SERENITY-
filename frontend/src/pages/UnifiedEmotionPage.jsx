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
  RefreshCw,
  Send,
} from 'lucide-react';
import { Link } from 'react-router-dom';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
const CAPTURE_DURATION_MS = 5000;

const formatTime = (date = new Date()) => date.toLocaleTimeString();

const UnifiedEmotionPage = ({ user, onLogout }) => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const videoStreamRef = useRef(null);
  const audioStreamRef = useRef(null);
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const timerRef = useRef(null);

  const [isCameraOn, setIsCameraOn] = useState(false);
  const [isMicOn, setIsMicOn] = useState(false);
  const [isInteracting, setIsInteracting] = useState(false);
  const [isChatting, setIsChatting] = useState(false);
  const [statusText, setStatusText] = useState('Ready. Press the button below to speak.');
  const [userPrompt, setUserPrompt] = useState('');
  const [chatMessage, setChatMessage] = useState('');
  const [dominantEmotion, setDominantEmotion] = useState('Neutral');
  const [speechEmotion, setSpeechEmotion] = useState('Neutral');
  const [faceEmotion, setFaceEmotion] = useState('Neutral');
  const [transcription, setTranscription] = useState('');
  const [assistantText, setAssistantText] = useState('I am here with you.');
  const [errors, setErrors] = useState([]);
  const [logs, setLogs] = useState([]);
  const [videoDevices, setVideoDevices] = useState([]);
  const [audioDevices, setAudioDevices] = useState([]);
  const [selectedVideoDeviceId, setSelectedVideoDeviceId] = useState('');
  const [selectedAudioDeviceId, setSelectedAudioDeviceId] = useState('');

  const greetingLine = useMemo(() => `Patient: ${user}`, [user]);

  const addLog = (emotion) => {
    setLogs((prev) => [
      { time: formatTime(), emotion: emotion || 'Neutral' },
      ...prev,
    ].slice(0, 10));
  };

  const updateFromPayload = (payload) => {
    setDominantEmotion(payload.dominant_emotion || 'Neutral');
    setSpeechEmotion(payload.speech_emotion || 'Neutral');
    setFaceEmotion(payload.face_emotion || 'Neutral');
    setTranscription(payload.transcription || '');
    setAssistantText(payload.llm_response || 'I am here with you.');
    setErrors(payload.errors || []);
    addLog(payload.dominant_emotion || 'Neutral');
  };

  const playTtsIfAvailable = async (payload) => {
    if (!payload?.tts_audio_base64) return;
    try {
      const audio = new Audio(`data:audio/mpeg;base64,${payload.tts_audio_base64}`);
      await audio.play();
    } catch {
      setErrors((prev) => [...prev, 'Audio playback was blocked by browser policy.']);
    }
  };

  const refreshDevices = async () => {
    if (!navigator.mediaDevices?.enumerateDevices) {
      setErrors((prev) => [...prev, 'Device enumeration is not supported in this browser.']);
      return;
    }

    const devices = await navigator.mediaDevices.enumerateDevices();
    const cams = devices.filter((d) => d.kind === 'videoinput');
    const mics = devices.filter((d) => d.kind === 'audioinput');
    setVideoDevices(cams);
    setAudioDevices(mics);

    if (!selectedVideoDeviceId && cams.length > 0) {
      setSelectedVideoDeviceId(cams[0].deviceId);
    }
    if (!selectedAudioDeviceId && mics.length > 0) {
      setSelectedAudioDeviceId(mics[0].deviceId);
    }
  };

  useEffect(() => {
    refreshDevices().catch(() => {
      setErrors((prev) => [...prev, 'Unable to read media devices.']);
    });

    const handleDeviceChange = () => {
      refreshDevices().catch(() => {});
    };

    if (navigator.mediaDevices?.addEventListener) {
      navigator.mediaDevices.addEventListener('devicechange', handleDeviceChange);
    }

    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }

      if (videoStreamRef.current) {
        videoStreamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (audioStreamRef.current) {
        audioStreamRef.current.getTracks().forEach((track) => track.stop());
      }

      if (navigator.mediaDevices?.removeEventListener) {
        navigator.mediaDevices.removeEventListener('devicechange', handleDeviceChange);
      }
    };
  }, []);

  const ensureVideoStream = async () => {
    if (videoStreamRef.current) {
      const existingTrack = videoStreamRef.current.getVideoTracks()[0];
      const existingDeviceId = existingTrack?.getSettings?.().deviceId;
      if (!selectedVideoDeviceId || !existingDeviceId || existingDeviceId === selectedVideoDeviceId) {
        return videoStreamRef.current;
      }

      videoStreamRef.current.getTracks().forEach((track) => track.stop());
      videoStreamRef.current = null;
    }

    const candidateConstraints = [
      {
        video: selectedVideoDeviceId
          ? { deviceId: { exact: selectedVideoDeviceId } }
          : { facingMode: 'user' },
        audio: false,
      },
      { video: true, audio: false },
    ];

    let lastError = null;
    for (const constraints of candidateConstraints) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        videoStreamRef.current = stream;
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
        setIsCameraOn(true);
        return stream;
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error('Unable to access camera.');
  };

  const ensureAudioStream = async () => {
    if (audioStreamRef.current) {
      const existingTrack = audioStreamRef.current.getAudioTracks()[0];
      const existingDeviceId = existingTrack?.getSettings?.().deviceId;
      if (!selectedAudioDeviceId || !existingDeviceId || existingDeviceId === selectedAudioDeviceId) {
        return audioStreamRef.current;
      }

      audioStreamRef.current.getTracks().forEach((track) => track.stop());
      audioStreamRef.current = null;
    }

    const candidateConstraints = [
      {
        audio: selectedAudioDeviceId
          ? { deviceId: { exact: selectedAudioDeviceId } }
          : true,
        video: false,
      },
      { audio: true, video: false },
    ];

    let lastError = null;
    for (const constraints of candidateConstraints) {
      try {
        const stream = await navigator.mediaDevices.getUserMedia(constraints);
        audioStreamRef.current = stream;
        setIsMicOn(true);
        return stream;
      } catch (error) {
        lastError = error;
      }
    }

    throw lastError || new Error('Unable to access microphone.');
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

  const submitInteraction = async ({ imageBase64 = '', audioBlob = null, textFallback = '' }) => {
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

    const response = await axios.post(`${API_BASE_URL}/api/interact`, formData, {
      timeout: 120000,
    });
    const payload = response.data;
    updateFromPayload(payload);
    await playTtsIfAvailable(payload);
  };

  const recordAudioChunk = async (stream) => {
    return new Promise((resolve, reject) => {
      try {
        const mimeType = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
          ? 'audio/webm;codecs=opus'
          : 'audio/webm';
        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType });
      } catch (error) {
        reject(error);
        return;
      }

      chunksRef.current = [];
      mediaRecorderRef.current.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data);
        }
      };

      mediaRecorderRef.current.onerror = (event) => {
        reject(event.error || new Error('MediaRecorder failed.'));
      };

      mediaRecorderRef.current.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
        resolve(blob.size > 0 ? blob : null);
      };

      mediaRecorderRef.current.start();
      timerRef.current = setTimeout(() => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
          mediaRecorderRef.current.stop();
        }
      }, CAPTURE_DURATION_MS);
    });
  };

  const startInteraction = async () => {
    if (isInteracting) {
      return;
    }

    setIsInteracting(true);
    setErrors([]);
    setStatusText('Listening... capturing audio + face frame.');

    try {
      let imageBase64 = '';
      try {
        await ensureVideoStream();
        imageBase64 = captureSnapshot();
      } catch (videoError) {
        setErrors((prev) => [...prev, `Camera unavailable: ${videoError.message || videoError}`]);
      }

      let audioBlob = null;
      try {
        const audioStream = await ensureAudioStream();
        audioBlob = await recordAudioChunk(audioStream);
      } catch (audioError) {
        setErrors((prev) => [...prev, `Microphone unavailable: ${audioError.message || audioError}`]);
      }

      if (!audioBlob && !userPrompt.trim()) {
        throw new Error('No microphone audio captured. Enter text fallback or connect a microphone.');
      }

      await submitInteraction({
        imageBase64,
        audioBlob,
        textFallback: userPrompt.trim(),
      });
    } catch (error) {
      setStatusText('Media access failed. Check camera and microphone permissions.');
      setErrors([String(error.message || error)]);
    } finally {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
      chunksRef.current = [];
      setIsInteracting(false);
      setStatusText('Ready. Press the button below to speak.');
    }
  };

  const sendChat = async () => {
    const message = chatMessage.trim() || userPrompt.trim();
    if (!message || isChatting) {
      return;
    }

    setIsChatting(true);
    setErrors([]);
    setStatusText('Thinking with Qwen RAG...');
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/chat`,
        {
          username: user || 'anonymous',
          message,
        },
        { timeout: 120000 }
      );
      const payload = response.data;
      updateFromPayload(payload);
      await playTtsIfAvailable(payload);
      setChatMessage('');
    } catch (error) {
      const backendMessage =
        error.response?.data?.detail ||
        error.response?.data?.error ||
        (error.request ? 'Backend unreachable. Ensure API is running on http://127.0.0.1:5000' : null) ||
        'Chat request failed. Please try again.';
      setErrors([backendMessage]);
      setStatusText(backendMessage);
    } finally {
      setIsChatting(false);
      setStatusText('Ready. Press the button below to speak.');
    }
  };

  const stopCamera = () => {
    if (timerRef.current) {
      clearTimeout(timerRef.current);
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
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
    setIsInteracting(false);
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
            <Activity className="text-cyan-300" /> SERENITY Multimodal Loop
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
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Subject</h3>
            <p className="text-sm">Name: <b>{user}</b></p>
            <p className="text-sm text-emerald-300">Status: {isCameraOn ? 'Online' : 'Offline'}</p>
            <p className="text-sm text-sky-300">Mic: {isMicOn ? 'Online' : 'Offline'}</p>
            <div className="mt-4 border-t border-cyan-900/40 pt-3">
              <h4 className="text-xs uppercase tracking-wider text-cyan-300 mb-2">Logs</h4>
              <div className="space-y-2 max-h-[420px] overflow-auto">
                {logs.map((entry, index) => (
                  <div key={`${entry.time}-${index}`} className="text-xs text-slate-300 flex justify-between">
                    <span>[{entry.time}]</span>
                    <span className="text-cyan-300 font-semibold">{entry.emotion.toLowerCase()}</span>
                  </div>
                ))}
              </div>
            </div>
          </section>

          <section className="xl:col-span-7 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Visual Cortex</h3>
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
                onClick={ensureVideoStream}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-cyan-700 hover:bg-cyan-600 px-4 py-2 font-semibold"
              >
                <Camera size={16} /> Start Vision
              </button>

              <button
                onClick={startInteraction}
                disabled={isInteracting}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-indigo-700 hover:bg-indigo-600 px-4 py-2 font-semibold disabled:opacity-60"
              >
                {isInteracting ? <Square size={16} /> : <Mic size={16} />}
                {isInteracting ? 'Capturing...' : 'Press to Talk (5s)'}
              </button>

              <button
                onClick={stopCamera}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-rose-700 hover:bg-rose-600 px-4 py-2 font-semibold"
              >
                <Square size={16} /> Stop
              </button>
            </div>

            <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
              <select
                value={selectedVideoDeviceId}
                onChange={(e) => setSelectedVideoDeviceId(e.target.value)}
                className="rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
              >
                <option value="">Default Camera</option>
                {videoDevices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Camera ${device.deviceId.slice(0, 6)}`}
                  </option>
                ))}
              </select>

              <select
                value={selectedAudioDeviceId}
                onChange={(e) => setSelectedAudioDeviceId(e.target.value)}
                className="rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
              >
                <option value="">Default Microphone</option>
                {audioDevices.map((device) => (
                  <option key={device.deviceId} value={device.deviceId}>
                    {device.label || `Mic ${device.deviceId.slice(0, 6)}`}
                  </option>
                ))}
              </select>

              <button
                onClick={() => refreshDevices().catch(() => setErrors((prev) => [...prev, 'Unable to refresh devices.']))}
                className="inline-flex items-center justify-center gap-2 rounded-lg bg-slate-700 hover:bg-slate-600 px-4 py-2 font-semibold"
              >
                <RefreshCw size={16} /> Refresh Devices
              </button>
            </div>

            <div className="mt-3">
              <textarea
                value={userPrompt}
                onChange={(e) => setUserPrompt(e.target.value)}
                placeholder="Optional fallback text prompt if speech transcription fails"
                className="w-full rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
                rows={2}
              />
            </div>

            {errors.length > 0 && (
              <div className="mt-3 rounded-lg border border-amber-600/60 bg-amber-950/30 p-3 text-amber-200 text-sm">
                {errors.map((error, index) => (
                  <p key={`${error}-${index}`}>{error}</p>
                ))}
              </div>
            )}
          </section>

          <section className="xl:col-span-3 rounded-2xl border border-cyan-900/50 bg-slate-900/60 p-4">
            <h3 className="text-xs uppercase tracking-wider text-cyan-300 mb-3">Neural Dialogue</h3>
            <div className="space-y-3 text-sm">
              <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3">
                <p className="text-cyan-300 font-semibold">System</p>
                <p className="text-slate-300">{statusText}</p>
              </div>
              <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3">
                <p className="text-slate-300">You</p>
                <p className="text-white">{transcription || '(awaiting speech input...)'}</p>
              </div>
              <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3">
                <p className="text-cyan-300 font-semibold inline-flex items-center gap-1"><MessageCircle size={14} /> Serenity</p>
                <p className="text-white mt-1">{assistantText}</p>
              </div>

              <div className="rounded-lg border border-cyan-800/60 bg-slate-900 p-3 space-y-2">
                <p className="text-cyan-300 font-semibold">Chat with Qwen RAG</p>
                <textarea
                  value={chatMessage}
                  onChange={(e) => setChatMessage(e.target.value)}
                  placeholder="Type your message if you prefer text chat"
                  className="w-full rounded-lg bg-slate-950 border border-cyan-900/60 px-3 py-2 text-sm text-slate-200"
                  rows={3}
                />
                <button
                  onClick={sendChat}
                  disabled={isChatting}
                  className="inline-flex items-center justify-center gap-2 rounded-lg bg-emerald-700 hover:bg-emerald-600 px-4 py-2 font-semibold disabled:opacity-60"
                >
                  <Send size={14} /> {isChatting ? 'Sending...' : 'Send to Serenity'}
                </button>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
};

export default UnifiedEmotionPage;

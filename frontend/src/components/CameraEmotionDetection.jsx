import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';
const ANALYSIS_INTERVAL_MS = 2000;

const CameraEmotionDetection = () => {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const rafIdRef = useRef(null);
  const lastSentAtRef = useRef(0);
  const isMountedRef = useRef(true);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [emotion, setEmotion] = useState(null);
  const [confidence, setConfidence] = useState(null);
  const [aiMessage, setAiMessage] = useState('');
  const [inferenceError, setInferenceError] = useState('');

  useEffect(() => {
    return () => {
      isMountedRef.current = false;
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
    };
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      streamRef.current = stream;
      videoRef.current.srcObject = stream;
      setInferenceError('');
      setIsStreaming(true);
    } catch (err) {
      console.error(err);
      alert("Camera access denied. Please allow camera permissions.");
    }
  };

  const stopCamera = () => {
    if (rafIdRef.current) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
    setIsAnalyzing(false);
  };

  const captureAndAnalyze = async () => {
    if (!videoRef.current || !canvasRef.current) return;
    if (videoRef.current.videoWidth === 0 || videoRef.current.videoHeight === 0) return;

    const context = canvasRef.current.getContext('2d');
    context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);
    const imageBase64 = canvasRef.current.toDataURL('image/jpeg');

    try {
      setIsAnalyzing(true);
      const res = await axios.post(`${API_BASE_URL}/detect_emotion`, {
        image: imageBase64,
      });

      if (!isMountedRef.current) return;

      setEmotion(res.data.emotion ?? 'Neutral');
      setConfidence(Number(res.data.confidence ?? 0));
      setAiMessage(res.data.ai_message || '');
      setInferenceError(res.data.error || '');
    } catch (error) {
      console.error("Backend Error:", error);
      if (!isMountedRef.current) return;
      const backendMessage =
        error.response?.data?.error ||
        error.response?.data?.detail ||
        (error.request ? 'Backend unreachable. Ensure API is running on http://127.0.0.1:5000' : null) ||
        'Failed to analyze frame. Please try again.';
      setInferenceError(backendMessage);
    } finally {
      if (isMountedRef.current) {
        setIsAnalyzing(false);
      }
    }
  };

  useEffect(() => {
    if (!isStreaming) return;

    const runLoop = (timestamp) => {
      if (!isMountedRef.current || !isStreaming) return;

      const elapsed = timestamp - lastSentAtRef.current;
      if (elapsed >= ANALYSIS_INTERVAL_MS) {
        lastSentAtRef.current = timestamp;
        captureAndAnalyze();
      }

      rafIdRef.current = requestAnimationFrame(runLoop);
    };

    rafIdRef.current = requestAnimationFrame(runLoop);

    return () => {
      if (rafIdRef.current) {
        cancelAnimationFrame(rafIdRef.current);
        rafIdRef.current = null;
      }
    };
  }, [isStreaming]);

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col gap-6 p-4">
      {/* 1. Camera Container */}
      <div className="relative w-full aspect-video bg-black rounded-2xl overflow-hidden shadow-2xl border-4 border-white">
        <video ref={videoRef} autoPlay muted className="w-full h-full object-cover" />
        <canvas ref={canvasRef} width="640" height="480" className="hidden" />
        {!isStreaming && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900 text-white">
            <p className="text-lg font-medium">Camera Offline</p>
          </div>
        )}
      </div>

      {/* 2. Controls & Results - Separated cleanly */}
      <div className="flex flex-col items-center gap-4">
        {!isStreaming && (
          <button 
            onClick={startCamera} 
            className="px-8 py-3 bg-blue-600 text-white font-bold rounded-full shadow-lg hover:bg-blue-700 transition-transform hover:scale-105 active:scale-95"
          >
            📸 Start Camera
          </button>
        )}

        {isStreaming && (
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-4 py-2 bg-red-100 text-red-700 rounded-full animate-pulse">
              <div className="w-3 h-3 bg-red-600 rounded-full"></div>
              {isAnalyzing ? 'Analyzing Live Feed...' : 'Camera Live'}
            </div>
            <button
              onClick={stopCamera}
              className="px-5 py-2 bg-gray-700 text-white font-semibold rounded-full shadow hover:bg-gray-800"
            >
              Stop
            </button>
          </div>
        )}

        {inferenceError && (
          <div className="w-full bg-amber-50 border border-amber-200 text-amber-800 p-3 rounded-lg text-sm">
            Model fallback active: {inferenceError}
          </div>
        )}

        {emotion && (
          <div className="w-full bg-white p-6 rounded-xl shadow-lg border border-gray-100 animate-fade-in">
            <div className="text-center mb-4">
               <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">Detected Emotion</p>
               <h2 className="text-4xl font-extrabold text-blue-900 uppercase">{emotion}</h2>
               <p className="text-sm text-gray-400 mt-1">Confidence: {confidence}%</p>
            </div>
            
            {aiMessage && (
              <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500">
                <p className="text-sm font-bold text-blue-800 mb-1">Serenity AI says:</p>
                <p className="text-gray-700 italic">"{aiMessage}"</p>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default CameraEmotionDetection;
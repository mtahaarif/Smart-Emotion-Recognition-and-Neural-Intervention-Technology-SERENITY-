import React, { useEffect, useRef, useState } from 'react';
import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://127.0.0.1:5000';

const SpeechEmotionDetection = () => {
  const [isRecording, setIsRecording] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState('');
  const [result, setResult] = useState(null);
  const mediaRecorderRef = useRef(null);
  const mediaStreamRef = useRef(null);
  const chunksRef = useRef([]);

  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop();
      }

      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop());
        mediaStreamRef.current = null;
      }
    };
  }, []);

  const cleanupMedia = () => {
    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((track) => track.stop());
      mediaStreamRef.current = null;
    }
  };

  const startRecording = async () => {
    try {
      setError('');
      setResult(null);

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'audio/webm' });
      chunksRef.current = [];

      mediaRecorderRef.current.ondataavailable = (e) => {
        if (e.data.size > 0) chunksRef.current.push(e.data);
      };

      mediaRecorderRef.current.onerror = (event) => {
        console.error('MediaRecorder error:', event.error);
        setError('Recording failed. Please try again.');
        setIsRecording(false);
        cleanupMedia();
      };

      mediaRecorderRef.current.onstop = async () => {
        setIsRecording(false);
        await uploadAudio();
        cleanupMedia();
      };

      mediaRecorderRef.current.start();
      setIsRecording(true);
    } catch (err) {
      console.error(err);
      alert("Microphone access denied.");
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      mediaRecorderRef.current.stop();
    }
  };

  const uploadAudio = async () => {
    if (chunksRef.current.length === 0) {
      setError('No audio captured. Please record again.');
      return;
    }

    setIsUploading(true);
    const blob = new Blob(chunksRef.current, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('file', blob, 'recording.webm');

    try {
      const res = await axios.post(`${API_BASE_URL}/analyze_audio`, formData);
      setResult(res.data);
      setError('');
    } catch (err) {
      console.error(err);
      const backendMessage =
        err.response?.data?.error ||
        err.response?.data?.detail ||
        (err.request ? 'Backend unreachable. Ensure API is running on http://127.0.0.1:5000' : null) ||
        'Audio analysis failed. Please try again.';
      setError(backendMessage);
    } finally {
      setIsUploading(false);
      chunksRef.current = [];
    }
  };

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-col items-center gap-8 p-6">
      
      {/* 1. Header */}
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-800">Voice Analysis</h2>
        <p className="text-gray-500">Press the button and speak clearly</p>
      </div>

      {/* 2. Main Button - Big and Clear */}
      <button
        onClick={isRecording ? stopRecording : startRecording}
        disabled={isUploading}
        className={`
          w-24 h-24 rounded-full flex items-center justify-center shadow-xl transition-all duration-300
          ${isUploading ? 'opacity-60 cursor-not-allowed' : ''}
          ${isRecording 
            ? 'bg-red-500 hover:bg-red-600 scale-110 shadow-red-200' 
            : 'bg-blue-600 hover:bg-blue-700 hover:scale-105 shadow-blue-200'}
        `}
      >
        {isRecording ? (
          <div className="w-8 h-8 bg-white rounded-sm animate-pulse" /> // Stop Icon
        ) : (
          <svg className="w-10 h-10 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg> // Mic Icon
        )}
      </button>

      {/* 3. Status Indicator */}
      <div className="h-8">
        {isRecording && <p className="text-red-500 font-semibold animate-pulse">● Recording in progress...</p>}
        {isUploading && (
          <p className="text-blue-600 font-semibold animate-pulse">Analyzing audio...</p>
        )}
      </div>

      {error && (
        <div className="w-full bg-red-50 border border-red-200 text-red-700 rounded-lg px-4 py-3 text-sm">
          {error}
        </div>
      )}

      {/* 4. Results Card */}
      {result && (
        <div className="w-full bg-white p-6 rounded-xl shadow-lg border border-gray-100 animate-fade-in text-center">
          <p className="text-gray-500 text-xs uppercase tracking-widest mb-1">Detected Emotion</p>
          <h2 className="text-3xl font-extrabold text-blue-900 uppercase mb-4">{result.emotion}</h2>

          {typeof result.confidence !== 'undefined' && (
            <p className="text-sm text-gray-500 mb-3">Confidence: {Number(result.confidence).toFixed(2)}%</p>
          )}

          {result.error && (
            <div className="mb-4 bg-amber-50 border border-amber-200 text-amber-800 rounded-lg px-3 py-2 text-sm text-left">
              Model fallback active: {result.error}
            </div>
          )}
          
          <div className="bg-blue-50 p-4 rounded-lg border-l-4 border-blue-500 text-left">
             <p className="text-sm font-bold text-blue-800 mb-1">Serenity AI says:</p>
             <p className="text-gray-700 italic">"{result.ai_message}"</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default SpeechEmotionDetection;
"use client";

import { useState, useRef, useEffect } from "react";

type Message = {
  role: "user" | "ai";
  text: string;
  timestamp: Date;
};

export default function VoiceAgentPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isSessionActive, setIsSessionActive] = useState(false);
  const [status, setStatus] = useState("Ready to start a session");
  
  // UI Sub-states
  const [isListening, setIsListening] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isSpeechDetected, setIsSpeechDetected] = useState(false);
  const [volume, setVolume] = useState(0);

  // PDF Upload States
  const [isUploading, setIsUploading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<{message: string, type: 'success' | 'error' | 'none'}>({message: '', type: 'none'});
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const silenceStartRef = useRef<number | null>(null);
  const speechDetectedRef = useRef(false);
  const isSessionActiveRef = useRef(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // VAD Config
  const VOLUME_THRESHOLD = 0.005; 
  const SILENCE_DURATION = 1500; // ms

  // Auto-scroll to bottom
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // Clean up on unmount
  useEffect(() => {
    return () => {
      stopSession();
    };
  }, []);

  const startSession = async () => {
    setIsSessionActive(true);
    isSessionActiveRef.current = true;
    setStatus("Session Active");
    await startListeningTurn();
  };

  const stopSession = () => {
    setIsSessionActive(false);
    isSessionActiveRef.current = false;
    setIsListening(false);
    setIsThinking(false);
    setIsSpeaking(false);
    setIsSpeechDetected(false);
    setVolume(0);
    setStatus("Session Ended");

    // 1. Stop Recording
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
    }
    
    // 2. Stop AI Playback
    if (audioRef.current) {
      audioRef.current.pause();
      audioRef.current.currentTime = 0;
    }

    // 3. Cleanup Audio Nodes
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }

    if (audioContextRef.current) {
      try {
        if (audioContextRef.current.state !== 'closed') {
          audioContextRef.current.close();
        }
      } catch (e) {
        console.warn("Session AudioContext cleanup issue", e);
      }
      audioContextRef.current = null;
    }
  };

  const startListeningTurn = async () => {
    if (!isSessionActiveRef.current) return;
    if (isThinking || isSpeaking) return;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ 
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        } 
      });
      
      const audioContext = new AudioContext();
      audioContextRef.current = audioContext;
      const source = audioContext.createMediaStreamSource(stream);
      const analyser = audioContext.createAnalyser();
      analyser.fftSize = 256;
      source.connect(analyser);
      analyserRef.current = analyser;

      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/wav" });
        await processTurn(audioBlob);
        
        // Cleanup VAD for this turn
        if (animationFrameRef.current) {
          cancelAnimationFrame(animationFrameRef.current);
          animationFrameRef.current = null;
        }

        try {
          if (audioContext.state !== 'closed') {
            await audioContext.close();
          }
        } catch (e) {
          console.warn("AudioContext already closing/closed", e);
        }
        
        stream.getTracks().forEach(track => track.stop());
      };

      mediaRecorder.start();
      setIsListening(true);
      setStatus("Listening...");
      
      speechDetectedRef.current = false;
      setIsSpeechDetected(false);
      silenceStartRef.current = null;
      
      monitorAudio();
    } catch (err) {
      console.error("Error accessing microphone:", err);
      setStatus("Mic error - check permissions");
      setIsSessionActive(false);
      isSessionActiveRef.current = false;
    }
  };

  const monitorAudio = () => {
    if (!analyserRef.current || !audioContextRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Float32Array(bufferLength);
    
    const checkLevel = async () => {
      // Use the Ref here to avoid stale closure issues
      if (!isSessionActiveRef.current || !analyserRef.current || !audioContextRef.current) return;
      
      if (audioContextRef.current.state === 'suspended') {
        await audioContextRef.current.resume();
      }
      
      analyserRef.current.getFloatTimeDomainData(dataArray);
      let sum = 0;
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i] * dataArray[i];
      }
      const rms = Math.sqrt(sum / bufferLength);
      
      // Update volume state for UI visualization
      setVolume(rms);

      if (rms > VOLUME_THRESHOLD) {
        if (!speechDetectedRef.current) {
          speechDetectedRef.current = true;
          setIsSpeechDetected(true);
        }
        silenceStartRef.current = null;
      } else if (speechDetectedRef.current) {
        if (!silenceStartRef.current) {
          silenceStartRef.current = Date.now();
        } else if (Date.now() - silenceStartRef.current > SILENCE_DURATION) {
          handleEndSpeech();
          return;
        }
      }

      animationFrameRef.current = requestAnimationFrame(checkLevel);
    };

    animationFrameRef.current = requestAnimationFrame(checkLevel);
  };

  const handleEndSpeech = () => {
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === "recording") {
      mediaRecorderRef.current.stop();
      setIsListening(false);
      setIsSpeechDetected(false);
      setStatus("Thinking...");
    }
  };

  const processTurn = async (audioBlob: Blob) => {
    if (!isSessionActiveRef.current) return;
    
    setIsThinking(true);
    setStatus("Agent is thinking...");

    const formData = new FormData();
    formData.append("audio", audioBlob, "user_voice.wav");
    formData.append("thread_id", "session_user_vapi");

    try {
      const response = await fetch("http://localhost:8000/api/chat/audio", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Failed to process audio");

      const data = await response.json();
      
      setMessages(prev => [
        ...prev,
        { role: "user", text: data.user_text, timestamp: new Date() },
        { role: "ai", text: data.ai_response, timestamp: new Date() }
      ]);

      setIsThinking(false);
      await playResponse(data.audio_base64);
    } catch (err) {
      console.error("API error:", err);
      setStatus("Processing failed");
      setIsThinking(false);
      // Restart listening if session still active
      if (isSessionActiveRef.current) setTimeout(startListeningTurn, 2000);
    }
  };

  const playResponse = (base64String: string) => {
    return new Promise((resolve) => {
      if (!isSessionActiveRef.current) {
        resolve(null);
        return;
      }

      const audioUrl = `data:audio/wav;base64,${base64String}`;
      const audio = new Audio(audioUrl);
      audioRef.current = audio;
      
      setIsSpeaking(true);
      setStatus("Aura is speaking...");
      audio.play();

      audio.onended = () => {
        setIsSpeaking(false);
        setStatus("Waiting for you...");
        resolve(null);
        
        if (isSessionActiveRef.current) {
          setTimeout(startListeningTurn, 200); 
        }
      };

      audio.onerror = () => {
        setIsSpeaking(false);
        resolve(null);
        if (isSessionActiveRef.current) startListeningTurn();
      }
    });
  };

  const triggerFileUpload = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    if (file.type !== "application/pdf") {
      setUploadStatus({message: "Please select a valid PDF file", type: 'error'});
      return;
    }

    setIsUploading(true);
    setUploadStatus({message: "Indexing knowledge...", type: 'none'});

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/api/pdf/upload", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data = await response.json();
      setUploadStatus({message: "Knowledge Updated Successfully!", type: 'success'});
      
      // Auto-clear success message
      setTimeout(() => setUploadStatus({message: '', type: 'none'}), 3000);
    } catch (err) {
      console.error("Upload error:", err);
      setUploadStatus({message: "Failed to upload knowledge", type: 'error'});
    } finally {
      setIsUploading(false);
      // Reset input value to allow re-uploading the same file
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  return (
    <main className="relative flex-1 flex flex-col items-center justify-center p-8 overflow-hidden">
      {/* Background Decor */}
      <div className="bg-gradient"></div>
      <div className="bg-blur" style={{ top: '10%', left: '15%' }}></div>
      <div className="bg-blur" style={{ bottom: '10%', right: '15%', background: 'var(--accent)' }}></div>

      {/* Fixed Header - Centered at the top */}
      <header className="fixed top-12 left-0 right-0 flex flex-col items-center z-50 pointer-events-none">
        <div className="flex flex-col items-center pointer-events-auto px-6 py-4">
          <h1 className="text-7xl font-black tracking-tighter mb-4 bg-gradient-to-b from-white to-white/30 bg-clip-text text-transparent drop-shadow-[0_0_30px_rgba(255,255,255,0.2)]">
            AURA
          </h1>
          <div className="flex items-center gap-3 px-6 py-2 bg-white/5 backdrop-blur-2xl rounded-full border border-white/10 shadow-[0_8px_32px_rgba(0,0,0,0.4)]">
            <div className={`w-2.5 h-2.5 rounded-full ${isSessionActive ? "bg-cyan-400 animate-pulse shadow-[0_0_15px_#22d3ee]" : "bg-white/20"}`}></div>
            <p className="text-white/90 uppercase tracking-[0.3em] text-[11px] font-black">
              {status}
            </p>
          </div>
        </div>
      </header>

      <div className="flex-1 w-full max-w-4xl flex flex-col items-center justify-center gap-12 mt-32">
        {/* Chat History */}
        <div className="chat-container w-full">
          <div className="messages-list pr-2">
            {messages.length === 0 ? (
              <div className="flex flex-col items-center justify-center h-full text-slate-500 italic opacity-50">
                <p>Start a session to begin our conversation...</p>
              </div>
            ) : (
              messages.map((msg, index) => (
                <div 
                  key={index} 
                  className={`message-bubble ${msg.role === "user" ? "user-message" : "ai-message"}`}
                >
                  {msg.text}
                </div>
              ))
            )}
            <div ref={messagesEndRef} />
          </div>
        </div>

        {/* Controls Container */}
        <div className="controls flex flex-col items-center gap-6">
          
          {/* Status Area */}
          <div className="flex flex-col items-center gap-2">
            {isListening && isSpeechDetected && (
              <span className="bg-cyan-500/20 text-cyan-400 text-[10px] px-2 py-0.5 rounded-full font-black uppercase ring-1 ring-cyan-500/50 animate-pulse">
                Speech Detected
              </span>
            )}
            
            {/* Volume Indicator */}
            {isListening && (
              <div className="w-32 h-1 bg-white/10 rounded-full overflow-hidden">
                <div 
                  className="h-full bg-cyan-400 transition-all duration-75"
                  style={{ width: `${Math.min(volume * 1000, 100)}%` }}
                />
              </div>
            )}
          </div>

          {/* The Interaction Orb */}
          <div 
            className={`pulse-orb 
              ${isSessionActive ? "active" : ""} 
              ${isThinking ? "animate-spin-slow" : ""} 
              ${isSpeaking ? "ring-4 ring-cyan-400 ring-offset-4 ring-offset-black" : ""}
            `}
            onClick={() => isSessionActive ? stopSession() : startSession()}
          >
            {isSessionActive ? (
              <div className="bg-white w-8 h-8 rounded-sm animate-pulse shadow-[0_0_15px_rgba(255,255,255,0.5)]"></div>
            ) : (
              <svg className="w-16 h-16 text-white drop-shadow-lg" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3z" />
                <path d="M17 11c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z" />
              </svg>
            )}
          </div>

          {/* Contextual Action Button & Upload */}
          <div className="flex flex-col items-center gap-4">
            {!isSessionActive ? (
              <div className="flex flex-col items-center gap-6">
                <button 
                  onClick={startSession}
                  className="px-12 py-5 bg-indigo-600 hover:bg-indigo-500 text-white font-black rounded-full transition-all shadow-[0_0_30px_rgba(79,70,229,0.5)] hover:shadow-indigo-500/70 transform hover:-translate-y-1 active:scale-95 text-lg"
                >
                  Connect to Aura
                </button>
                
                {/* PDF Knowledge Upload */}
                <div className="flex flex-col items-center">
                  <input 
                    type="file" 
                    ref={fileInputRef} 
                    onChange={handleFileChange} 
                    accept=".pdf" 
                    className="hidden" 
                  />
                  <button 
                    onClick={triggerFileUpload}
                    disabled={isUploading}
                    className={`flex items-center gap-2 px-6 py-2.5 rounded-full border border-white/10 text-sm font-bold transition-all backdrop-blur-sm
                      ${isUploading ? "bg-white/5 cursor-wait" : "bg-white/5 hover:bg-white/10 active:scale-95"}
                    `}
                  >
                    <svg className={`w-4 h-4 ${isUploading ? "animate-bounce" : ""}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                    </svg>
                    {isUploading ? "Indexing..." : "Upload Knowledge (PDF)"}
                  </button>
                  
                  {uploadStatus.message && (
                    <p className={`mt-3 text-[11px] font-black uppercase tracking-widest animate-pulse
                      ${uploadStatus.type === 'success' ? "text-cyan-400" : uploadStatus.type === 'error' ? "text-red-400" : "text-white/40"}
                    `}>
                      {uploadStatus.message}
                    </p>
                  )}
                </div>
              </div>
            ) : (
              <div className="flex gap-4">
                  <button 
                    onClick={stopSession}
                    className="px-8 py-3 bg-red-600/20 hover:bg-red-600/40 text-red-400 border border-red-500/50 rounded-full text-base font-black transition-all backdrop-blur-sm shadow-[0_0_20px_rgba(239,68,68,0.2)]"
                  >
                    End Session
                  </button>
                  {isSpeaking && (
                    <button 
                      onClick={() => { if(audioRef.current) audioRef.current.pause(); setIsSpeaking(false); startListeningTurn(); }}
                      className="px-8 py-3 bg-white/10 hover:bg-white/20 text-white rounded-full text-base font-black border border-white/10 transition-all backdrop-blur-sm"
                    >
                      Stop Agent
                    </button>
                  )}
              </div>
            )}
          </div>
        </div>
      </div>
    </main>
  );
}

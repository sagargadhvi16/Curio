import React, { useRef, useState, useEffect } from 'react';
import { Mic, AudioLines } from 'lucide-react';

interface AudioVisualizerProps {
  onAudioData: (level: number) => void;
}

const AudioVisualizer: React.FC<AudioVisualizerProps> = ({ onAudioData }) => {
  const [isListening, setIsListening] = useState(false);
  const [permissionGranted, setPermissionGranted] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [frequencyData, setFrequencyData] = useState<number[]>([]);
  
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const microphoneStreamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  
  const API_URL = "http://127.0.0.1:8000/assistant"; // Replace with your actual base URL if needed

  // Health check on mount
  useEffect(() => {
    fetch(`${API_URL}/status`)
      .then(res => {
        if (!res.ok) throw new Error("API health check failed");
      })
      .catch(() => setErrorMessage("API is not available. Please try again later."));
  }, []);

  const startAPI = async () => {
    try {
      await fetch(`${API_URL}/start`, { method: "POST" });
      setIsListening(true);
    } catch {
      setErrorMessage("Failed to start API session.");
    }
  };

  const stopAPI = async () => {
    try {
      await fetch(`${API_URL}/stop`, { method: "POST" });
    } catch {
      setErrorMessage("Failed to stop API session.");
    }
  };

  const requestMicrophoneAccess = async () => {
    try {
      // Call API start endpoint
      await startAPI();

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
      
      if (!audioContextRef.current) {
        audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      }
      
      analyserRef.current = audioContextRef.current.createAnalyser();
      analyserRef.current.fftSize = 256;
      
      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);
      
      microphoneStreamRef.current = stream;
      setPermissionGranted(true);
      setIsListening(true);
      setErrorMessage(null);
      
      startVisualization();
    } catch (error) {
      console.error('Error accessing microphone:', error);
      setErrorMessage('Microphone access denied. Please allow microphone access to enable the audio visualization.');
      setPermissionGranted(false);
      setIsListening(false);
    }
  };
  
  const startVisualization = () => {
    if (!analyserRef.current) return;
    
    const bufferLength = analyserRef.current.frequencyBinCount;
    const dataArray = new Uint8Array(bufferLength);
    
    const updateVisualization = () => {
      if (!analyserRef.current) return;
      
      analyserRef.current.getByteFrequencyData(dataArray);
      
      // Calculate average volume level
      let sum = 0;
      const frequencyValues = [];
      
      for (let i = 0; i < bufferLength; i++) {
        sum += dataArray[i];
        if (i % 4 === 0) { // Sample every 4th value for visualization
          frequencyValues.push(dataArray[i] / 255);
        }
      }
      
      const average = sum / bufferLength / 255; // Normalize to 0-1
      setFrequencyData(frequencyValues);
      onAudioData(average);
      
      // Draw visualization on canvas
      drawVisualization(dataArray, bufferLength);
      
      animationFrameRef.current = requestAnimationFrame(updateVisualization);
    };
    
    animationFrameRef.current = requestAnimationFrame(updateVisualization);
  };
  
  const drawVisualization = (dataArray: Uint8Array, bufferLength: number) => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    
    // Clear canvas
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Set dimensions
    const WIDTH = canvas.width;
    const HEIGHT = canvas.height;
    
    // Set bar width based on available bars to draw
    const barWidth = (WIDTH / (bufferLength / 4)) * 2.5;
    let barHeight;
    let x = 0;
    
    // Draw circular visualization
    const centerX = WIDTH / 2;
    const centerY = HEIGHT / 2;
    const radius = Math.min(WIDTH, HEIGHT) / 4;
    
    for (let i = 0; i < bufferLength; i += 4) {
      const normalized = dataArray[i] / 255;
      const angle = (i / bufferLength) * Math.PI * 2;
      
      // Outer circle (frequency bars)
      if (i % 8 === 0) {
        ctx.beginPath();
        const innerRadius = radius * 0.8;
        const outerRadius = radius * (1 + normalized * 0.5);
        
        // Calculate points
        const innerX = centerX + innerRadius * Math.cos(angle);
        const innerY = centerY + innerRadius * Math.sin(angle);
        const outerX = centerX + outerRadius * Math.cos(angle);
        const outerY = centerY + outerRadius * Math.sin(angle);
        
        // Draw lines
        ctx.moveTo(innerX, innerY);
        ctx.lineTo(outerX, outerY);
        
        // Style based on frequency
        const hue = (i / bufferLength) * 360;
        ctx.strokeStyle = `hsla(${hue}, 80%, 60%, ${0.7 + normalized * 0.3})`;
        ctx.lineWidth = 3;
        ctx.stroke();
      }
      
      // Inner circle (pulsing)
      if (i === 0) {
        const pulseRadius = radius * 0.6 * (0.8 + normalized * 0.4);
        ctx.beginPath();
        ctx.arc(centerX, centerY, pulseRadius, 0, Math.PI * 2);
        ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.fill();
      }
    }
  };
  
  const toggleListening = () => {
    if (isListening) {
      stopListening();
    } else {
      requestMicrophoneAccess();
    }
  };
  
  const stopListening = async () => {
    // Call API stop endpoint
    await stopAPI();

    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
    
    if (microphoneStreamRef.current) {
      microphoneStreamRef.current.getTracks().forEach(track => track.stop());
      microphoneStreamRef.current = null;
    }
    
    setIsListening(false);
    setFrequencyData([]);
    onAudioData(0);
    
    // Clear canvas
    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      if (ctx) {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
      }
    }
  };
  
  useEffect(() => {
    return () => {
      if (animationFrameRef.current) {
        cancelAnimationFrame(animationFrameRef.current);
      }
      
      if (microphoneStreamRef.current) {
        microphoneStreamRef.current.getTracks().forEach(track => track.stop());
      }
    };
  }, []);
  
  return (
    <div className="flex flex-col items-center">
      <div className="relative h-[80px] w-[180px] mb-4">
        <canvas ref={canvasRef} width={180} height={80} className="w-full h-full" />
        
        {/* Audio bars visualization when not listening */}
        {!isListening && (
          <div className="absolute inset-0 flex items-center justify-center">
            <AudioLines size={32} className="text-gray-400" />
          </div>
        )}
      </div>
      
      <button 
        className={`relative p-4 rounded-full ${isListening ? 'bg-red-500/30 shadow-lg shadow-red-500/20' : 'bg-gray-300/30'} transition-all`}
        onClick={toggleListening}
        aria-label={isListening ? 'Stop listening' : 'Start listening'}
      >
        <Mic 
          size={32} 
          className={`${isListening ? 'text-red-500 animate-pulse-mic' : 'text-gray-700'}`} 
        />
        
        {/* Ripple effect when listening */}
        {isListening && (
          <>
            <span className="absolute inset-0 rounded-full animate-ping bg-red-400 opacity-25"></span>
            <span className="absolute inset-0 rounded-full animate-pulse bg-red-500 opacity-10"></span>
          </>
        )}
      </button>
      
      {errorMessage && (
        <p className="text-xs text-red-500 mt-2 max-w-[200px] text-center">{errorMessage}</p>
      )}
    </div>
  );
};

export default AudioVisualizer;

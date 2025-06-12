import React, { useState, useEffect } from 'react';
import AudioVisualizer from './AudioVisualizer';
import { PlayCircle, StopCircle, BarChart3, Clock, Lightbulb } from 'lucide-react';

const KidWatchApp = () => {
  const [isRunning, setIsRunning] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [audioLevel, setAudioLevel] = useState(0);
  const [error, setError] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [showAnalysis, setShowAnalysis] = useState(false);
  const [activeSpeechAnimation, setActiveSpeechAnimation] = useState(false);
  
  // API endpoint base URL
  const API_URL = 'https://127.0.0.1:8000/assistant';
  
  // Check if KidWatch is already running on component mount
  useEffect(() => {
    checkKidWatchStatus();
  }, []);
  
  // Simulate response animation when audio level is high enough
  useEffect(() => {
    if (audioLevel > 0.15 && isRunning) {
      setActiveSpeechAnimation(true);
      // Reset after a delay to simulate processing time
      const timeout = setTimeout(() => {
        setActiveSpeechAnimation(false);
      }, 2000);
      
      return () => clearTimeout(timeout);
    }
  }, [audioLevel, isRunning]);
  
  // Check if KidWatch is running
  const checkKidWatchStatus = async () => {
    try {
      const response = await fetch(`${API_URL}/status`);
      const data = await response.json();
      setIsRunning(data.kidwatch_running);
    } catch (err) {
      console.error('Error checking KidWatch status:', err);
      setError('Could not connect to KidWatch API');
    }
  };
  
  // Start KidWatch assistant
  const startKidWatch = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/start`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to start KidWatch: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('KidWatch started:', data);
      setIsRunning(true);
    } catch (err) {
      console.error('Error starting KidWatch:', err);
      setError(err.message || 'Failed to start KidWatch');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Stop KidWatch assistant
  const stopKidWatch = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_URL}/stop`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
      });
      
      if (!response.ok) {
        throw new Error(`Failed to stop KidWatch: ${response.statusText}`);
      }
      
      const data = await response.json();
      console.log('KidWatch stopped:', data);
      setIsRunning(false);
    } catch (err) {
      console.error('Error stopping KidWatch:', err);
      setError(err.message || 'Failed to stop KidWatch');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Analyze chat history
  const analyzeChats = async (timeRange = null) => {
    setIsLoading(true);
    setError(null);
    
    try {
      let url = `${API_URL}/analyze`;
      if (timeRange) {
        url += `?time_range=${timeRange}`;
      }
      
      const response = await fetch(url);
      
      if (!response.ok) {
        throw new Error(`Failed to analyze chats: ${response.statusText}`);
      }
      
      const data = await response.json();
      setAnalysisResults(data.analysis);
      setShowAnalysis(true);
    } catch (err) {
      console.error('Error analyzing chats:', err);
      setError(err.message || 'Failed to analyze chat history');
    } finally {
      setIsLoading(false);
    }
  };
  
  // Handle audio level changes from visualizer
  const handleAudioData = (level) => {
    setAudioLevel(level);
  };
  
  return (
    <div className="flex flex-col items-center bg-gradient-to-b from-blue-50 to-purple-50 min-h-screen p-6">
      <div className="w-full max-w-md bg-white rounded-xl shadow-lg overflow-hidden">
        {/* Header */}
        <div className="bg-blue-600 text-white p-4 flex items-center justify-center">
          <h1 className="text-xl font-bold">KidWatch Assistant</h1>
        </div>
        
        {/* Main Content */}
        <div className="p-6">
          {/* Status Indicator */}
          <div className="flex justify-center mb-6">
            <div className={`inline-flex items-center ${isRunning ? 'text-green-500' : 'text-gray-400'} text-sm font-medium`}>
              <span className={`inline-block w-3 h-3 rounded-full mr-2 ${isRunning ? 'bg-green-500 animate-pulse' : 'bg-gray-300'}`}></span>
              {isRunning ? 'KidWatch is listening' : 'KidWatch is inactive'}
            </div>
          </div>
          
          {/* Audio Visualizer */}
          <div className="flex justify-center mb-6">
            <AudioVisualizer onAudioData={handleAudioData} />
          </div>
          
          {/* Speech Activity Indicator */}
          {isRunning && (
            <div className="text-center mb-6">
              <div className={`text-sm ${audioLevel > 0.1 ? 'text-blue-500' : 'text-gray-400'} transition-colors duration-300`}>
                {audioLevel > 0.1 ? 'I hear you talking!' : 'Waiting for voice...'}
              </div>
              
              {/* Assistant Response Animation */}
              {activeSpeechAnimation && (
                <div className="mt-2">
                  <div className="flex justify-center space-x-1">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }}></div>
                  </div>
                  <p className="text-xs text-blue-500 mt-1">KidWatch is responding...</p>
                </div>
              )}
            </div>
          )}
          
          {/* Control Buttons */}
          <div className="flex justify-center space-x-4 mb-6">
            <button 
              onClick={isRunning ? stopKidWatch : startKidWatch}
              disabled={isLoading}
              className={`flex items-center px-4 py-2 rounded-full text-white font-medium shadow transition-colors ${
                isRunning 
                  ? 'bg-red-500 hover:bg-red-600' 
                  : 'bg-green-500 hover:bg-green-600'
              } ${isLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {isRunning ? (
                <>
                  <StopCircle size={18} className="mr-2" />
                  Stop KidWatch
                </>
              ) : (
                <>
                  <PlayCircle size={18} className="mr-2" />
                  Start KidWatch
                </>
              )}
            </button>
          </div>
          
          {/* Analysis Section Toggle */}
          <div className="border-t pt-4 text-center">
            <button 
              onClick={() => setShowAnalysis(!showAnalysis)}
              className="text-blue-600 hover:text-blue-800 text-sm font-medium flex items-center justify-center mx-auto"
            >
              <BarChart3 size={16} className="mr-1" />
              {showAnalysis ? 'Hide Analysis' : 'Show Analysis Tools'}
            </button>
          </div>
          
          {/* Analysis Tools */}
          {showAnalysis && (
            <div className="mt-4 border-t pt-4">
              <h2 className="text-lg font-medium mb-3 text-center">Conversation Analysis</h2>
              
              <div className="grid grid-cols-2 gap-2 mb-4">
                <button
                  onClick={() => analyzeChats('today')}
                  disabled={isLoading}
                  className="flex items-center justify-center bg-blue-100 hover:bg-blue-200 text-blue-700 py-2 px-3 rounded-md text-sm transition-colors"
                >
                  <Clock size={14} className="mr-2" />
                  Today's Chats
                </button>
                <button
                  onClick={() => analyzeChats()}
                  disabled={isLoading}
                  className="flex items-center justify-center bg-purple-100 hover:bg-purple-200 text-purple-700 py-2 px-3 rounded-md text-sm transition-colors"
                >
                  <Lightbulb size={14} className="mr-2" />
                  All Chat History
                </button>
              </div>
              
              {/* Analysis Results */}
              {analysisResults && (
                <div className="bg-gray-50 border rounded-lg p-4 mt-2">
                  <h3 className="font-medium mb-2 text-gray-700">Analysis Results</h3>
                  <pre className="text-xs overflow-auto max-h-40 bg-gray-100 p-2 rounded">
                    {JSON.stringify(analysisResults, null, 2)}
                  </pre>
                </div>
              )}
            </div>
          )}
          
          {/* Error Display */}
          {error && (
            <div className="mt-4 bg-red-50 border border-red-200 text-red-600 p-3 rounded-md text-sm">
              {error}
            </div>
          )}
        </div>
      </div>
      
      {/* Footer */}
      <div className="mt-4 text-gray-500 text-xs">
        KidWatch Assistant © 2025
      </div>
    </div>
  );
};

export default KidWatchApp;
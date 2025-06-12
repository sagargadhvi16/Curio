
import React, { useState } from 'react';
import WatchFrame from './WatchFrame';
import TimeDisplay from './TimeDisplay';
import WaveBackground from './WaveBackground';
import AudioVisualizer from './AudioVisualizer';

const SmartWatch: React.FC = () => {
  const [audioLevel, setAudioLevel] = useState(0);

  const handleAudioData = (level: number) => {
    setAudioLevel(level);
  };

  return (
    <WatchFrame>
      <div className="relative flex flex-col h-full">
        <WaveBackground audioLevel={audioLevel} />
        
        <div className="relative z-10 flex flex-col h-full">
          <TimeDisplay />
          
          <div className="flex-grow flex flex-col items-center justify-center px-4 text-center">
            <h2 className="text-2xl font-bold mb-6 text-black">
              Hey Nexxy, <br />
              where do dreams <br />
              come from?
            </h2>
            
            <AudioVisualizer onAudioData={handleAudioData} />
          </div>
        </div>
      </div>
    </WatchFrame>
  );
};

export default SmartWatch;


import React from 'react';

interface WaveBackgroundProps {
  audioLevel: number;
}

const WaveBackground: React.FC<WaveBackgroundProps> = ({ audioLevel = 0 }) => {
  const amplitudeFactor = Math.min(audioLevel * 50, 30);
  
  return (
    <div className="absolute inset-0 overflow-hidden bg-gradient-to-b from-[#2C3E50] to-[#4C5C68]">
      <div 
        className={`absolute w-full h-[50%] bg-gradient-to-r from-watch-peach to-[#FFD1DC] transform-gpu transition-transform`} 
        style={{
          transform: `translateY(${-10 + amplitudeFactor * -0.5}px)`,
          transition: 'transform 0.1s ease',
          opacity: 0.8
        }}
      ></div>
      <div 
        className={`absolute w-full h-[55%] top-[10%] bg-gradient-to-r from-watch-mint to-[#B5EAD7] transform-gpu transition-transform`}
        style={{
          transform: `translateY(${amplitudeFactor * -0.7}px)`,
          transition: 'transform 0.15s ease',
          opacity: 0.7
        }}
      ></div>
      <div 
        className={`absolute w-full h-[60%] top-[20%] bg-gradient-to-r from-watch-blue to-[#C7CEEA] transform-gpu transition-transform`}
        style={{
          transform: `translateY(${amplitudeFactor * -0.9}px)`,
          transition: 'transform 0.2s ease',
          opacity: 0.6
        }}
      ></div>
      <div 
        className={`absolute w-full h-[65%] top-[30%] bg-gradient-to-r from-watch-pink to-[#FFDFD3] transform-gpu transition-transform`}
        style={{
          transform: `translateY(${amplitudeFactor * -1.1}px)`,
          transition: 'transform 0.25s ease',
          opacity: 0.5
        }}
      ></div>
    </div>
  );
};

export default WaveBackground;

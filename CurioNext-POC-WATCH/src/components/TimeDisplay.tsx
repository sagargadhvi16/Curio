
import React, { useState, useEffect } from 'react';
import { Wifi } from 'lucide-react';

const TimeDisplay: React.FC = () => {
  const [currentTime, setCurrentTime] = useState('');

  useEffect(() => {
    const updateTime = () => {
      const now = new Date();
      const hours = now.getHours();
      const minutes = now.getMinutes();
      const ampm = hours >= 12 ? 'PM' : 'AM';
      const formattedHours = hours % 12 || 12;
      const formattedMinutes = minutes < 10 ? `0${minutes}` : minutes;
      setCurrentTime(`${formattedHours}:${formattedMinutes} ${ampm}`);
    };

    updateTime();
    const interval = setInterval(updateTime, 1000);
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="flex justify-between items-center w-full px-4 py-2 bg-watch-peach/90 z-10">
      <h1 className="text-2xl font-bold text-black">{currentTime}</h1>
      <Wifi size={24} color="black" />
    </div>
  );
};

export default TimeDisplay;

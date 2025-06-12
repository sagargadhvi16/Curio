import React from 'react';
interface WatchFrameProps {
  children: React.ReactNode;
}
const WatchFrame: React.FC<WatchFrameProps> = ({
  children
}) => {
  return <div className="flex flex-col items-center justify-center w-full min-h-screen bg-gray-100 p-4">
      <div className="relative">
        {/* Watch Strap Top - Increased width by 3x */}
        <div className="w-[150px] h-[120px] bg-watch-strap mx-auto rounded-t-lg shadow-lg"></div>
        
        {/* Watch Frame */}
        <div className="relative w-[320px] h-[380px] bg-watch-frame rounded-[40px] shadow-xl flex flex-col items-center justify-center">
          {/* Watch Screen */}
          <div className="relative w-[280px] h-[340px] rounded-watch overflow-hidden">
            {children}
          </div>
          
          {/* Watch Crown */}
          
          <div className="absolute -right-3 top-2/3 w-3 h-10 bg-watch-strap rounded-r-lg"></div>
        </div>
        
        {/* Watch Strap Bottom - Increased width by 3x */}
        <div className="w-[150px] h-[120px] bg-watch-strap mx-auto rounded-b-lg shadow-lg"></div>
      </div>
    </div>;
};
export default WatchFrame;
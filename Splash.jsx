import React, { useEffect, useState } from 'react';
import './Splash.css';

export default function Splash() {
  const [isVisible, setIsVisible] = useState(true);

  useEffect(() => {
    // Unmount from DOM completely after 4.2 seconds
    const timer = setTimeout(() => setIsVisible(false), 4200);
    return () => clearTimeout(timer);
  }, []);

  if (!isVisible) return null;

  return (
    <div className="splash-overlay" id="splash-screen">
      <div className="splash-content">
        <div className="text-wrapper">
          <span className="letter-n">N</span>
          <span className="letter-euro">euro</span>
        </div>
        
        <div className="scope-wrapper">
          <svg className="scope-circle" viewBox="0 0 100 100">
            <circle cx="50" cy="50" r="45" fill="none" stroke="#F97316" strokeWidth="6" />
            <line x1="50" y1="0" x2="50" y2="25" stroke="#F97316" strokeWidth="6" className="crosshair ch-top" />
            <line x1="50" y1="100" x2="50" y2="75" stroke="#F97316" strokeWidth="6" className="crosshair ch-bottom" />
            <line x1="0" y1="50" x2="25" y2="50" stroke="#F97316" strokeWidth="6" className="crosshair ch-left" />
            <line x1="100" y1="50" x2="75" y2="50" stroke="#F97316" strokeWidth="6" className="crosshair ch-right" />
          </svg>
        </div>

        <div className="xai-wrapper">
          <span className="letter-xai">Scope-XAI</span>
        </div>
      </div>
    </div>
  );
}

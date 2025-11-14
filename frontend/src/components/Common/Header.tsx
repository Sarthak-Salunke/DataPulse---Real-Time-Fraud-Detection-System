import { useState, useEffect } from 'react';
import { ICONS } from '../../utils/constants';

const Header = () => {
  const [lastUpdated, setLastUpdated] = useState(new Date());

  useEffect(() => {
    const timer = setInterval(() => setLastUpdated(new Date()), 5000);
    return () => clearInterval(timer);
  }, []);

  return (
    <div>
      <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center pb-4 border-b border-[var(--border-color-dark)]">
        <div className="flex flex-col">
          <h2 className="text-3xl font-bold text-inherit">Live Fraud Detection Dashboard</h2>
          <p className="text-sm text-[var(--text-secondary-dark)] mt-1">Monitoring customer transactions in real-time</p>
        </div>
        <div className="flex items-center space-x-4 mt-4 sm:mt-0">
          <div className="flex items-center space-x-2">
            <div className="relative flex items-center justify-center w-3 h-3">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-green-400 opacity-75"></span>
              <span className="relative inline-flex h-2 w-2 rounded-full bg-green-500"></span>
            </div>
            <span className="text-sm text-[var(--text-secondary-dark)]">System Status: <span className="text-green-400 font-semibold">Active</span></span>
          </div>
          <div className="hidden sm:flex items-center space-x-2">
            <ICONS.clock className="h-4 w-4 text-[var(--text-secondary-dark)]"/>
            <span className="text-sm text-[var(--text-secondary-dark)]">Last Updated: {lastUpdated.toLocaleTimeString()}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Header;
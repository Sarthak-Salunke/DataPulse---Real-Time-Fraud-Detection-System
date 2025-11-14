import { useState, useEffect } from 'react';
import { ICONS } from '../utils/constants';

const handleSmoothScroll = (e: React.MouseEvent<HTMLAnchorElement>) => {
  e.preventDefault();
  const href = e.currentTarget.getAttribute('href');
  if (!href || !href.startsWith('#')) return;
  
  const targetId = href.substring(1);
  const targetElement = document.getElementById(targetId);

  if (targetElement) {
    targetElement.scrollIntoView({
      behavior: 'smooth',
      block: 'start'
    });
  }
};

export const LandingHeader = () => {
  return (
    <header className="sticky top-0 left-0 right-0 z-50 bg-[var(--sidebar-bg-dark)]/80 backdrop-blur-lg border-b border-[var(--border-color-dark)]">
      <nav className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-20 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <ICONS.shieldCheck className="h-8 w-8 text-[var(--accent-text)]" />
          <span className="text-xl font-bold text-inherit">DataPulse AI</span>
        </div>
        <div className="flex items-center space-x-4">
          <a href="#solution" onClick={handleSmoothScroll} className="hidden sm:block text-sm font-medium text-[var(--text-secondary-dark)] hover:text-white transition-colors">Solution</a>
          <a href="#dashboard" onClick={handleSmoothScroll} className="hidden sm:block text-sm font-medium text-[var(--text-secondary-dark)] hover:text-white transition-colors">Dashboard</a>
          <a href="#how-it-works" onClick={handleSmoothScroll} className="hidden sm:block text-sm font-medium text-[var(--text-secondary-dark)] hover:text-white transition-colors">How it Works</a>
        </div>
      </nav>
    </header>
  );
};

const AiAssistant = () => {
  const messages = [
    " Greetings! I am Pulse, your Assistant. Welcome to DataPulse AI.",
    " This dashboard provides real-time fraud intelligence. Ready to see it in action?",
    " Click the 'Launch Dashboard' button below to get started!",
  ];

  const [currentMessageIndex, setCurrentMessageIndex] = useState(0);
  const [displayedText, setDisplayedText] = useState('');
  const [isComponentVisible, setIsComponentVisible] = useState(false);
  const [isTypingComplete, setIsTypingComplete] = useState(false);
  
  useEffect(() => {
    const visibilityTimer = setTimeout(() => {
      setIsComponentVisible(true);
    }, 1500);
    return () => clearTimeout(visibilityTimer);
  }, []);

  useEffect(() => {
    if (!isComponentVisible) return;

    setDisplayedText('');
    setIsTypingComplete(false);
    let i = 0;
    const currentMessage = messages[currentMessageIndex];
    const typingInterval = setInterval(() => {
      if (i < currentMessage.length) {
        setDisplayedText(prev => prev + currentMessage.charAt(i));
        i++;
      } else {
        clearInterval(typingInterval);
        setIsTypingComplete(true);
      }
    }, 40);

    return () => clearInterval(typingInterval);
  }, [currentMessageIndex, isComponentVisible, messages]);
  
  useEffect(() => {
    if (isTypingComplete) {
      const messageCycleTimer = setTimeout(() => {
        setCurrentMessageIndex(prevIndex => (prevIndex + 1) % messages.length);
      }, 4000);
      return () => clearTimeout(messageCycleTimer);
    }
  }, [isTypingComplete, messages.length]);

  return (
    <div id="ai-assistant-container" className={isComponentVisible ? 'visible' : ''}>
      <div className="speech-bubble">
        <p className={isTypingComplete ? 'typing-done' : ''}>{displayedText}</p>
      </div>
      <div id="ai-assistant-orb" className="pulse">
        <ICONS.shieldCheck className="w-6 h-6 text-white" />
      </div>
    </div>
  );
};

export const Hero = () => {
  const mainTitle = "DataPulse AI";
  const subTitle = "Real-Time Fraud Intelligence & Visualization";

  return (
    <section className="relative min-h-screen flex flex-col justify-center text-center overflow-hidden bg-[var(--bg-color-dark)]">
      <div id="hero-video-container">
        <video id="hero-video" autoPlay loop muted playsInline>
          <source src="/video.mp4" type="video/mp4" />
        </video>
      </div>
      <div className="video-overlay"></div>
      <div className="absolute inset-0 bg-gradient-to-b from-transparent via-transparent to-[var(--bg-color-dark)] z-[2]"></div>
      <div className="relative z-10 max-w-4xl mx-auto px-4 sm:px-6 lg:px-8">
        <h1
          className="hero-glitch text-5xl sm:text-6xl md:text-7xl font-black uppercase will-animate animate-now delay-100"
          data-text={mainTitle}
        >
          {mainTitle}
        </h1>
        <p className="hero-subtitle mt-4 text-lg sm:text-xl will-animate animate-now delay-200">
          {subTitle}
        </p>

        <p className="mt-8 max-w-2xl mx-auto text-lg text-[var(--text-secondary-dark)] will-animate animate-now delay-300">
          DataPulse AI provides a high-fidelity dashboard for monitoring your entire credit card transaction pipeline, from data ingestion to ML-powered fraud analysis.
        </p>
        <div className="mt-10 flex justify-center will-animate animate-now delay-500">
          <a 
            href="#dashboard"
            onClick={handleSmoothScroll}
            className="super-button"
          >
            <span>Launch Dashboard</span>
            <ICONS.arrowRight className="arrow" />
          </a>
        </div>
      </div>
      <AiAssistant />
    </section>
  );
};

export const Footer = () => (
  <footer className="bg-slate-900/50 border-t border-[var(--border-color-dark)] py-8">
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center text-sm text-[var(--text-secondary-dark)]">
      <p>&copy; {new Date().getFullYear()} DataPulse AI. All rights reserved.</p>
      <p className="mt-2">A demonstration of real-time data visualization and fraud detection.</p>
    </div>
  </footer>
);
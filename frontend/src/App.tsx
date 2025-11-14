import { useState, useEffect, createContext, useContext, PropsWithChildren, useRef } from 'react';
import Header from './components/Common/Header';
import Dashboard from './components/Dashboard/Dashboard';
import { LandingHeader, Hero, Footer } from './components/LandingPage';
import ArchitectureDiagram from './components/Pipeline/ArchitectureDiagram';
import ProblemSection from './components/ProblemSection';
import SolutionSection from './components/SolutionSection';
import VerifiableComputeSection from './components/VerifiableComputeSection';

// Theme Management
type Theme = 'light' | 'dark';
const ThemeContext = createContext<{ theme: Theme; toggleTheme: () => void } | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (!context) throw new Error('useTheme must be used within a ThemeProvider');
  return context;
};

const ThemeProvider = ({ children }: PropsWithChildren) => {
  const [theme, setTheme] = useState<Theme>('dark');

  useEffect(() => {
    const root = window.document.documentElement;
    root.classList.remove(theme === 'light' ? 'dark' : 'light');
    root.classList.add(theme);
  }, [theme]);

  const toggleTheme = () => {
    setTheme(prevTheme => (prevTheme === 'light' ? 'dark' : 'light'));
  };

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// AnimateOnScroll Wrapper
const AnimateOnScroll = ({ 
  children, 
  className, 
  threshold = 0.1 
}: { 
  children: React.ReactNode; 
  className?: string; 
  threshold?: number;
}) => {
  const ref = useRef<HTMLDivElement>(null);
  const [isVisible, setIsVisible] = useState(false);

  useEffect(() => {
    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsVisible(true);
          observer.unobserve(entry.target);
        }
      },
      { threshold }
    );

    const currentRef = ref.current;
    if (currentRef) {
      observer.observe(currentRef);
    }

    return () => {
      if (currentRef) {
        observer.unobserve(currentRef);
      }
    };
  }, [threshold]);

  return (
    <div ref={ref} className={`${className || ''} ${isVisible ? 'is-visible' : ''}`}>
      {children}
    </div>
  );
};

// Main App Component
function App() {
  return (
    <ThemeProvider>
      <div className="bg-[var(--bg-color-dark)] text-[var(--text-primary-dark)] font-sans min-h-screen">
        <LandingHeader />
        <main>
          <Hero />
          
          <AnimateOnScroll>
            <section id="dashboard" className="py-16 sm:py-24">
              <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="will-animate">
                  <Header />
                </div>
                <div className="mt-8 will-animate delay-200">
                  <Dashboard />
                </div>
              </div>
            </section>
          </AnimateOnScroll>

          <AnimateOnScroll><ProblemSection /></AnimateOnScroll>
          <AnimateOnScroll><SolutionSection /></AnimateOnScroll>
          <AnimateOnScroll><VerifiableComputeSection /></AnimateOnScroll>
          <AnimateOnScroll><ArchitectureDiagram /></AnimateOnScroll>
        </main>
        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default App;
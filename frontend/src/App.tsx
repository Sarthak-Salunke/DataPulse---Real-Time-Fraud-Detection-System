import { useState, useEffect, createContext, useContext, type PropsWithChildren } from 'react';
import { LandingHeader, Hero, StatsTicker, CtaSection, Footer } from './components/LandingPage';
import ProblemSection from './components/ProblemSection';
import HowItWorks from './components/SolutionSection';
import FeaturesGrid from './components/VerifiableComputeSection';
import ArchitectureDiagram from './components/Pipeline/ArchitectureDiagram';
import Header from './components/Common/Header';
import Dashboard from './components/Dashboard/Dashboard';

// ── Theme context ──────────────────────────────────────────────────────────
type Theme = 'dark' | 'light';

interface ThemeCtx {
  theme: Theme;
  toggleTheme: () => void;
}

const ThemeContext = createContext<ThemeCtx | undefined>(undefined);

export const useTheme = (): ThemeCtx => {
  const ctx = useContext(ThemeContext);
  if (!ctx) throw new Error('useTheme must be used inside ThemeProvider');
  return ctx;
};

const ThemeProvider = ({ children }: PropsWithChildren) => {
  const [theme, setTheme] = useState<Theme>('dark');

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'light') {
      root.classList.add('light');
    } else {
      root.classList.remove('light');
    }
  }, [theme]);

  const toggleTheme = () => setTheme(t => (t === 'dark' ? 'light' : 'dark'));

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
};

// ── App ────────────────────────────────────────────────────────────────────
function App() {
  return (
    <ThemeProvider>
      <div style={{ background: 'var(--bg-void)', color: 'var(--text-primary)', minHeight: '100vh' }}>

        {/* ── Landing nav ── */}
        <LandingHeader />

        <main>
          {/* ── Hero ── */}
          <Hero />

          {/* ── Stats ticker ── */}
          <StatsTicker />

          {/* ── The Problem ── */}
          <ProblemSection />

          {/* ── How It Works (pinned scroll steps) ── */}
          <HowItWorks />

          {/* ── Features grid ── */}
          <FeaturesGrid />

          {/* ── Architecture pipeline ── */}
          <ArchitectureDiagram />

          {/* ── CTA ── */}
          <CtaSection />

          {/* ── Live Dashboard ── */}
          <section
            id="dashboard"
            style={{
              padding: '80px 32px 100px',
              background: 'var(--bg-void)',
              borderTop: '1px solid var(--border)',
            }}
          >
            <div style={{ maxWidth: '1400px', margin: '0 auto' }}>
              <Header />
              <Dashboard />
            </div>
          </section>
        </main>

        {/* ── Footer ── */}
        <Footer />
      </div>
    </ThemeProvider>
  );
}

export default App;

const stats = [
  { label: 'Global Fraud Losses', value: '$32.39 Billion' },
  { label: 'Growth by 2027', value: 'Projected to $40B' },
  { label: 'Detection Challenge', value: 'Millisecond Response Time' },
];

const ProblemSection = () => {
  return (
    <section id="problem" className="relative py-20 sm:py-32 overflow-hidden">
      <div className="absolute top-0 left-0 -translate-x-1/4 -translate-y-1/2 w-full h-full bg-[radial-gradient(ellipse_at_center,_var(--accent-primary)_0%,transparent_40%)] opacity-30"></div>
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-2 gap-12 items-center">
          <div className="relative z-10 will-animate">
            <h2 className="text-3xl sm:text-4xl font-extrabold text-inherit tracking-tight">
              The Hidden Cost of Delayed Fraud Detection
            </h2>
            <p className="mt-6 text-lg text-[var(--text-secondary-dark)]">
              In the world of digital transactions, every second counts. Traditional fraud detection systems are often too slow, relying on batch processing that leaves a critical window of opportunity for fraudsters. By the time an issue is flagged, the damage is already done, leading to financial loss and eroded customer trust.
            </p>
            <div className="mt-8 flex flex-wrap gap-x-8 gap-y-4">
              {stats.map((stat) => (
                <div key={stat.label}>
                  <p className="text-3xl font-bold text-[var(--accent-text)]">{stat.value}</p>
                  <p className="text-sm text-[var(--text-secondary-dark)] uppercase tracking-wider">{stat.label}</p>
                </div>
              ))}
            </div>
          </div>
          <div className="relative h-64 md:h-auto will-animate delay-200">
            <div className="absolute inset-0 bg-gradient-to-tr from-slate-900 to-slate-800 border border-[var(--panel-border-dark)] rounded-2xl -rotate-2"></div>
            <div className="relative inset-0 bg-[var(--panel-bg-dark)] border border-[var(--panel-border-dark)] rounded-2xl p-6 rotate-1 shadow-2xl flex flex-col justify-center">
              <p className="font-mono text-red-400 text-sm">ALERT: High-risk transaction detected.</p>
              <p className="font-mono text-gray-400 text-sm mt-2">&gt; Transaction ID: TX45329871</p>
              <p className="font-mono text-gray-400 text-sm">&gt; Amount: $749.99</p>
              <p className="font-mono text-gray-400 text-sm">&gt; Merchant: ShadyElectronics.com</p>
              <p className="font-mono text-yellow-400 text-sm mt-2 animate-pulse">&gt; ACTION: System holds transaction pending review...</p>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
};

export default ProblemSection;

import { ICONS } from '../utils/constants';

const features = [
  {
    icon: ICONS.activity,
    name: 'Real-time Analytics',
    description: 'Monitor transactions, fraud rates, and model performance as they happen with sub-second latency.',
  },
  {
    icon: ICONS.alert,
    name: 'Instant Fraud Alerts',
    description: 'Get immediate notifications in a dedicated feed for every transaction flagged by the ML model.',
  },
  {
    icon: ICONS.shieldCheck,
    name: 'Model Performance',
    description: 'Track key metrics like accuracy and recall to ensure the ML model is performing optimally.',
  },
  {
    icon: ICONS.server,
    name: 'System Health Monitor',
    description: 'Keep an eye on all backend components, from the database to the streaming pipeline, in one place.',
  },
  {
    icon: ICONS.trendingUp,
    name: 'Interactive Charts',
    description: 'Dive deep into data with interactive charts for transaction trends, amounts, and categories.',
  },
  {
    icon: ICONS.code,
    name: 'Developer Friendly',
    description: 'Built with a modern stack for high performance and easy integration into your existing workflows.',
  },
];

const VerifiableComputeSection = () => {
    return (
        <section className="py-20 sm:py-32 bg-[var(--bg-color-dark)]">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                 <div className="text-center will-animate">
                    <h2 className="text-3xl sm:text-4xl font-extrabold text-inherit tracking-tight">
                        A Powerful, All-in-One Dashboard
                    </h2>
                    <p className="mt-4 text-lg max-w-3xl mx-auto text-[var(--text-secondary-dark)]">
                        DataPulse AI provides every tool you need to effectively monitor and combat financial fraud, all from a single, intuitive interface.
                    </p>
                </div>
                <div className="mt-16 grid gap-8 md:grid-cols-2 lg:grid-cols-3">
                    {features.map((feature) => (
                        <div key={feature.name} className="bg-[var(--panel-bg-dark)] backdrop-blur-md border border-[var(--panel-border-dark)] rounded-xl p-6 shadow-lg glow-border transition-all duration-300 will-animate stagger">
                             <div className="w-12 h-12 rounded-lg bg-indigo-500/10 flex items-center justify-center border border-indigo-500/20">
                                <feature.icon className="w-6 h-6 text-[var(--accent-text)]" />
                            </div>
                            <h3 className="mt-4 text-lg font-bold text-inherit">{feature.name}</h3>
                            <p className="mt-2 text-sm text-[var(--text-secondary-dark)]">{feature.description}</p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default VerifiableComputeSection;
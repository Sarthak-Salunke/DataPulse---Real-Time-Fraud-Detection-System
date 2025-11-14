
import { ICONS } from '../utils/constants';

const benefits = [
  {
    icon: ICONS.alert,
    title: 'Act Instantly',
    description: 'Our real-time feed and alert system mean you see fraudulent activity the moment it happens, not hours later.',
  },
  {
    icon: ICONS.trendingUp,
    title: 'Visualize Trends',
    description: 'Go beyond raw numbers. Our interactive charts help you spot patterns in fraudulent behavior to stay ahead.',
  },
  {
    icon: ICONS.server,
    title: 'Monitor System Health',
    description: 'Ensure your entire pipeline is running smoothly with a comprehensive health dashboard for all your services.',
  },
    {
    icon: ICONS.shieldCheck,
    title: 'Trust Your Data',
    description: 'With a clear view of model accuracy and performance, you can be confident in your system\'s decisions.',
  },
];

const SolutionSection = () => {
    return (
        <section id="solution" className="py-20 sm:py-32 bg-[var(--panel-bg-dark)]/40">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center will-animate">
                     <p className="text-base font-semibold text-[var(--accent-text)] tracking-wider uppercase">The Solution</p>
                    <h2 className="mt-3 text-3xl sm:text-4xl font-extrabold text-inherit tracking-tight">
                        From Reactive to Proactive Fraud Management
                    </h2>
                    <p className="mt-4 text-lg max-w-2xl mx-auto text-[var(--text-secondary-dark)]">
                        DataPulse AI closes the gap. By processing and visualizing transaction data in real-time, we empower your team to detect and respond to threats instantly, transforming your security posture.
                    </p>
                </div>
                <div className="mt-16 grid gap-8 md:grid-cols-2 lg:grid-cols-4">
                    {benefits.map((item) => (
                        <div key={item.title} className="text-center p-6 bg-slate-800/20 rounded-xl border border-transparent hover:border-[var(--panel-border-dark)] transition-colors duration-300 will-animate stagger">
                            <div className="w-16 h-16 rounded-full bg-indigo-500/10 flex items-center justify-center mx-auto border-2 border-indigo-500/20">
                                <item.icon className="w-8 h-8 text-[var(--accent-text)]" />
                            </div>
                            <h3 className="mt-5 text-lg font-bold text-inherit">{item.title}</h3>
                            <p className="mt-2 text-sm text-[var(--text-secondary-dark)]">
                                {item.description}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </section>
    );
};

export default SolutionSection;
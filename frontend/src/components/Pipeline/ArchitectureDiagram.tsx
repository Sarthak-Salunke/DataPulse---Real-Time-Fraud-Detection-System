import React from 'react';
import type { Theme } from '../../types';
import { ICONS } from '../../utils/constants';

const Node: React.FC<{ icon: React.ElementType, title: string, description: string, isEnd?: boolean }> = ({ icon: Icon, title, description, isEnd }) => (
    <div className="group flex flex-col items-center text-center w-32 mx-2 cursor-pointer">
        <div className={`w-16 h-16 rounded-full border-2 flex items-center justify-center bg-slate-800/50 transition-all duration-300 ${isEnd ? 'border-[var(--accent-text)] animate-subtle-glow' : 'border-[var(--border-color-dark)] group-hover:border-[var(--accent-text)] group-hover:shadow-[0_0_20px_var(--accent-text)]'}`}>
            <Icon className={`w-8 h-8 transition-all duration-300 ${isEnd ? 'text-[var(--accent-text)]' : 'text-gray-400 group-hover:text-[var(--accent-text)] group-hover:drop-shadow-[0_0_5px_var(--accent-text)]'}`} />
        </div>
        <h4 className="font-bold mt-2 text-sm text-inherit transition-colors group-hover:text-white">{title}</h4>
        <p className="text-xs text-[var(--text-secondary-dark)] transition-colors group-hover:text-gray-300">{description}</p>
    </div>
);

const Arrow: React.FC = () => (
    <div className="flex-1 h-px bg-gradient-to-r from-[var(--border-color-dark)] to-transparent"></div>
);

const Packet: React.FC<{ delay: string; duration: string; travelDistance: string }> = ({ delay, duration, travelDistance }) => (
    <div
        className="absolute top-1/2 -translate-y-1/2 w-2 h-2 rounded-full bg-[var(--accent-text)]"
        style={{
            animation: `movePacket ${duration} linear ${delay} infinite`,
            boxShadow: '0 0 8px var(--accent-text)',
            // @ts-ignore
            '--packet-travel-distance': travelDistance,
        }}
    />
);

const ArchitectureDiagram: React.FC = () => {
    return (
        <section id="how-it-works" className="py-20 sm:py-32">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="text-center will-animate">
                    <h2 className="text-3xl sm:text-4xl font-extrabold text-inherit tracking-tight">
                        How The Backend Pipeline Works
                    </h2>
                    <p className="mt-4 text-lg max-w-2xl mx-auto text-[var(--text-secondary-dark)]">
                        Follow the journey of a transaction from the point of sale to your screen, powered by a robust, real-time data processing architecture.
                    </p>
                </div>
                <div className="mt-16 bg-[var(--panel-bg-dark)]/70 backdrop-blur-md border border-[var(--panel-border-dark)] rounded-xl p-8 sm:p-12 shadow-2xl glow-border will-animate delay-200">
                    <div className="relative flex items-center justify-center">
                        <Node icon={ICONS.database} title="Data Source" description="Live Transactions" />
                        <Arrow />
                        <Node icon={ICONS.server} title="Kafka" description="Real-time Stream" />
                        <Arrow />
                        <Node icon={ICONS.cpu} title="Spark ML" description="Fraud Detection" />
                        <Arrow />
                        <Node icon={ICONS.database} title="PostgreSQL" description="Stores Results" />
                        <Arrow />
                        <Node icon={ICONS.trendingUp} title="Dashboard" description="Live Visualization" isEnd />
                        
                        <div className="absolute top-0 left-0 w-full h-full pointer-events-none">
                           <div className="absolute top-1/2 left-[calc(8%)] w-[18%]" style={{transform: 'translateY(-1rem)'}}>
                                <Packet delay="0s" duration="2s" travelDistance="100%" />
                           </div>
                           <div className="absolute top-1/2 left-[calc(29%)] w-[18%]" style={{transform: 'translateY(-1rem)'}}>
                                <Packet delay="2s" duration="2s" travelDistance="100%" />
                           </div>
                           <div className="absolute top-1/2 left-[calc(50%)] w-[18%]" style={{transform: 'translateY(-1rem)'}}>
                                <Packet delay="4s" duration="2s" travelDistance="100%" />
                           </div>
                            <div className="absolute top-1/2 left-[calc(71%)] w-[18%]" style={{transform: 'translateY(-1rem)'}}>
                                <Packet delay="6s" duration="2s" travelDistance="100%" />
                           </div>
                        </div>
                    </div>
                </div>
            </div>
        </section>
    );
};

export default ArchitectureDiagram;
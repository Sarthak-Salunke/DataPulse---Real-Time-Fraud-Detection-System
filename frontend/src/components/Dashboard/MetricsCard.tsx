import React, { useState, useEffect, useRef } from 'react';
import type { Theme } from '../../types';
import type { KPI } from '../../types';
import { ICONS } from '../../utils/constants';

interface KPICardProps extends KPI {}

const Panel: React.FC<React.PropsWithChildren<{ className?: string }>> = ({ children, className }) => (
    <div className={`bg-[var(--panel-bg-dark)] backdrop-blur-md border border-[var(--panel-border-dark)] rounded-xl p-4 shadow-lg transition-all duration-300 glow-border h-full ${className}`}>
        {children}
    </div>
);

const easeOutExpo = (t: number): number => (t === 1 ? 1 : 1 - Math.pow(2, -10 * t));

const AnimatedCounter = ({ value }: { value: string }) => {
    const [displayValue, setDisplayValue] = useState(0);
    // Fix: Provide an initial value to useRef to fix "Expected 1 arguments, but got 0" error.
    const frameRef = useRef(0);
    const startTimeRef = useRef(0);
    const startValueRef = useRef(0);

    const endValue = parseFloat(value.replace(/[,%]/g, ''));
    const suffix = value.match(/[%]/g)?.join('') || '';
    const hasDecimal = value.includes('.');
    const decimalPlaces = hasDecimal ? value.split('.')[1].replace('%','').length : 0;
    
    useEffect(() => {
        const initialValue = parseFloat(value.replace(/[,%]/g, ''));
        setDisplayValue(initialValue);
    }, []);

    useEffect(() => {
        startValueRef.current = displayValue;
        startTimeRef.current = performance.now();
        const duration = 1500;

        const animate = (timestamp: number) => {
            const progress = timestamp - startTimeRef.current;
            const percentage = Math.min(progress / duration, 1);
            const easedPercentage = easeOutExpo(percentage);

            const currentCount = startValueRef.current + (endValue - startValueRef.current) * easedPercentage;
            setDisplayValue(currentCount);

            if (progress < duration) {
                frameRef.current = requestAnimationFrame(animate);
            } else {
                setDisplayValue(endValue);
            }
        };

        frameRef.current = requestAnimationFrame(animate);

        return () => {
            if (frameRef.current) {
                cancelAnimationFrame(frameRef.current);
            }
        };
    }, [endValue]);

    const formatValue = (num: number) => {
         if (isNaN(num)) return '...';
         if (hasDecimal) {
            return num.toFixed(decimalPlaces);
         }
         return Math.floor(num).toLocaleString();
    };

    return <>{formatValue(displayValue)}{suffix}</>;
};


const KPICard: React.FC<KPICardProps> = ({ title, value, details, change, changeType }) => {
    const isIncrease = changeType === 'increase';
    const colorClass = isIncrease ? 'text-green-400' : 'text-red-400';
    const Icon = isIncrease ? ICONS.trendingUp : ICONS.trendingDown;

    return (
        <Panel>
            <p className="text-sm text-[var(--text-secondary-dark)] font-semibold uppercase truncate">{title}</p>
            <div className="mt-2 flex items-baseline justify-between">
                <p className="text-3xl font-extrabold text-inherit">
                    <AnimatedCounter value={value} />
                </p>
                 <div className={`flex items-center text-sm font-semibold ${colorClass}`}>
                    <Icon className="h-4 w-4 mr-1" />
                    <span>{change}</span>
                </div>
            </div>
            <p className="text-xs text-[var(--text-secondary-dark)] mt-1 truncate">{details}</p>
        </Panel>
    );
};

export default KPICard;
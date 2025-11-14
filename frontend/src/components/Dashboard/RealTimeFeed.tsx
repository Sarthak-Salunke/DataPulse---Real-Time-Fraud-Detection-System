import React from 'react';
import type { Theme } from '../../types';
import { ICONS } from '../../utils/constants';
import type { Transaction } from '../../types';

interface AlertCardProps {
    alert: Transaction;
}

const AlertCard = ({ alert }: AlertCardProps) => (
    <div className="bg-white/5 p-3 rounded-lg border border-white/10">
        <div className="flex justify-between items-start">
            <div className="flex items-center gap-2">
                <ICONS.alert className="h-5 w-5 text-red-400 flex-shrink-0" />
                <div>
                    <p className="font-semibold text-sm text-inherit">{alert.merchant}</p>
                    <p className="text-xs text-[var(--text-secondary-dark)]">{alert.category}</p>
                </div>
            </div>
            <p className="text-lg font-bold text-red-400">${alert.amount.toFixed(2)}</p>
        </div>
        <div className="text-xs text-[var(--text-secondary-dark)] mt-2 space-y-1">
            <p><strong>Customer:</strong> {alert.customer}</p>
            <p><strong>Distance:</strong> {alert.distance} km</p>
            <p><strong>Confidence:</strong> {alert.confidence}%</p>
        </div>
    </div>
);


const AlertsFeed: React.FC<{ alerts: Transaction[] }> = ({ alerts }) => {
    return (
        <div className="bg-[var(--panel-bg-dark)] backdrop-blur-md border border-[var(--panel-border-dark)] rounded-xl p-4 sm:p-6 shadow-lg glow-border h-[26rem] flex flex-col">
            <div className="flex justify-between items-center mb-4">
                <h3 className="font-bold text-lg text-inherit">Fraud Alerts Feed</h3>
                <button className="text-gray-400 hover:text-white">
                    <ICONS.moreHorizontal className="h-5 w-5" />
                </button>
            </div>
            <div className="space-y-3 flex-1 overflow-y-auto pr-2 -mr-3">
                {alerts.map((alert, index) => (
                    <div key={alert.id} className={index === 0 ? 'animate-slide-down' : ''}>
                        <AlertCard alert={alert} />
                    </div>
                ))}
            </div>
        </div>
    );
};

export default AlertsFeed;

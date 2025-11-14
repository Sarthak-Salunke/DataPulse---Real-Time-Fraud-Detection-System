import React, { useEffect, useState } from 'react';
import MetricsCard from './MetricsCard';
import RealTimeFeed from './RealTimeFeed';
import TransactionChart from '../Charts/TransactionChart';
import CreditDebitChart from '../Charts/CreditDebitChart';
import SystemAdvicesChart from '../Charts/SystemAdvicesChart';
import type { Transaction, KPI, TransactionStatus, Toast as ToastType } from '../../types';
import { ICONS } from '../../utils/constants';
// ✨ NEW IMPORTS - API Integration
import { useDashboardMetrics, useRecentAlerts, useWebSocket } from '../../hooks/useApiData';
import type { DashboardMetrics, FraudAlert } from '@services/api';

const Panel: React.FC<{ children: React.ReactNode; className?: string }> = ({ children, className }) => (
    <div className={`bg-[var(--panel-bg-dark)] backdrop-blur-md border border-[var(--panel-border-dark)] rounded-xl p-4 sm:p-6 shadow-lg glow-border ${className}`}>
        {children}
    </div>
);

// --- TOAST COMPONENT ---
const Toast: React.FC<{ toast: ToastType; onRemove: (id: number) => void; }> = ({ toast, onRemove }) => {
    const [isFadingOut, setIsFadingOut] = useState(false);

    useEffect(() => {
        const timer = setTimeout(() => { setIsFadingOut(true); }, 5000);
        return () => clearTimeout(timer);
    }, []);

    useEffect(() => {
        if (isFadingOut) {
            const timer = setTimeout(() => { onRemove(toast.id); }, 500);
            return () => clearTimeout(timer);
        }
    }, [isFadingOut, onRemove, toast.id]);

    const Icon = toast.type === 'error' ? ICONS.alert : ICONS.check;
    const color = toast.type === 'error' ? 'red' : 'green';

    return (
        <div 
            className={`w-full max-w-sm bg-[var(--panel-bg-dark)] backdrop-blur-lg border border-${color}-500/50 rounded-lg shadow-2xl p-4 flex items-start space-x-4 mb-4`}
            style={{ animation: isFadingOut ? 'toast-out 0.5s ease-out forwards' : 'toast-in 0.5s ease-out forwards' }}
        >
            <Icon className={`w-6 h-6 text-${color}-400 mt-1 flex-shrink-0`} />
            <div className="flex-1">
                <h3 className={`font-bold text-${color}-400`}>{toast.title}</h3>
                <p className="text-sm text-gray-300">{toast.message}</p>
            </div>
            <button onClick={() => setIsFadingOut(true)} className="p-1 rounded-full text-gray-400 hover:bg-white/10">
                <ICONS.x className="w-4 h-4" />
            </button>
        </div>
    );
};

// --- MODAL COMPONENT ---
const DetailRow = ({ label, value }: { label: string; value: string | number | undefined }) => (
  <div className="flex justify-between items-center py-2 border-b border-white/10">
    <span className="text-sm text-[var(--text-secondary-dark)]">{label}</span>
    <span className="text-sm font-semibold text-right">{value}</span>
  </div>
);

const TransactionDetailModal: React.FC<{ transaction: Transaction | null; onClose: () => void; }> = ({ transaction, onClose }) => {
  if (!transaction) return null;
  const isFraud = transaction.status === 'Fraud';
  return (
    <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-50 flex items-center justify-center animate-fade-in-up" onClick={onClose}>
      <div 
        className="bg-[var(--panel-bg-dark)] border border-[var(--panel-border-dark)] rounded-xl shadow-2xl w-full max-w-md m-4 glow-border"
        onClick={e => e.stopPropagation()}
        style={{ animation: 'slideDown 0.3s ease-out forwards' }}
      >
        <div className="p-6 border-b border-white/10 flex justify-between items-center">
          <div>
            <h2 className="text-lg font-bold">Transaction Details</h2>
            <p className="text-xs text-[var(--text-secondary-dark)] font-mono">{transaction.id}</p>
          </div>
          <button onClick={onClose} className="p-2 rounded-full text-gray-400 hover:bg-white/10">
            <ICONS.x className="w-5 h-5" />
          </button>
        </div>
        <div className="p-6">
            <div className={`p-4 rounded-lg mb-4 text-center ${isFraud ? 'bg-red-500/10 border border-red-500/30' : 'bg-green-500/10 border border-green-500/30'}`}>
                <p className={`text-2xl font-bold ${isFraud ? 'text-red-400' : 'text-green-400'}`}>${transaction.amount.toFixed(2)}</p>
                <p className={`text-sm font-semibold uppercase tracking-wider ${isFraud ? 'text-red-400' : 'text-green-400'}`}>{transaction.status}</p>
            </div>
            <div className="space-y-1">
                <DetailRow label="Time" value={transaction.time} />
                <DetailRow label="Customer" value={transaction.customer} />
                <DetailRow label="Merchant" value={transaction.merchant} />
                <DetailRow label="Category" value={transaction.category} />
                <DetailRow label="Distance from Home" value={`${transaction.distance} km`} />
                {isFraud && <DetailRow label="Fraud Confidence" value={`${transaction.confidence}%`} />}
            </div>
        </div>
      </div>
    </div>
  );
};

// --- MOCK DATA GENERATION (Fallback) ---
const merchants = ['Amazon', 'Walmart', 'Apple', 'Starbucks', 'Netflix', 'ExxonMobil', 'Costco', 'Best Buy'];
const categories = ['Shopping', 'Grocery', 'Electronics', 'Coffee', 'Streaming', 'Gas', 'Wholesale', 'Retail'];
let txIdCounter = 1000;

const generateTransaction = (): Transaction => {
    const merchantIndex = Math.floor(Math.random() * merchants.length);
    const isFraud = Math.random() < 0.05;
    const amount = isFraud ? Math.random() * 400 + 100 : Math.random() * 150 + 5;
    
    return {
        id: `TX${++txIdCounter}`,
        time: new Date().toLocaleTimeString(),
        customer: `**** **** **** ${String(Math.floor(Math.random() * 9000) + 1000)}`,
        merchant: merchants[merchantIndex],
        category: categories[merchantIndex],
        amount: parseFloat(amount.toFixed(2)),
        distance: Math.floor(Math.random() * 2000),
        status: isFraud ? 'Fraud' : 'Normal',
        confidence: isFraud ? parseFloat((Math.random() * 15 + 85).toFixed(2)) : undefined,
    };
};

// ✨ NEW: Convert FraudAlert to Transaction format
const convertAlertToTransaction = (alert: FraudAlert): Transaction => ({
    id: alert.transaction_id || `API-${Date.now()}`,
    time: new Date(alert.timestamp).toLocaleTimeString(),
    customer: alert.customer_name || alert.ccNum || 'Unknown',
    merchant: alert.merchant_name || alert.merchant || 'Unknown',
    category: alert.category || 'Unknown',
    amount: alert.amount,
    distance: alert.distance || 0,
    status: 'Fraud' as TransactionStatus,
    confidence: alert.confidence,
    type: alert.transaction_type || 'debit'
});

// --- TRANSACTION FEED ---
const TransactionFeed: React.FC<{ transactions: Transaction[], onRowClick: (transaction: Transaction) => void }> = ({ transactions, onRowClick }) => (
    <Panel className="h-96 flex flex-col">
        <h2 className="text-lg font-bold text-inherit mb-4">Real-Time Transaction Feed</h2>
        <div className="flex-grow overflow-x-auto">
            <table className="w-full text-sm text-left">
                <thead className="text-xs text-[var(--text-secondary-dark)] uppercase">
                    <tr>
                        <th scope="col" className="px-4 py-2">Time</th>
                        <th scope="col" className="px-4 py-2">Transaction ID</th>
                        <th scope="col" className="px-4 py-2">Merchant</th>
                        <th scope="col" className="px-4 py-2">Amount</th>
                        <th scope="col" className="px-4 py-2">Status</th>
                    </tr>
                </thead>
                <tbody>
                    {transactions.map((tx, index) => (
                        <tr key={tx.id} className={`border-b border-[var(--border-color-dark)] transition-colors hover:bg-white/5 cursor-pointer ${index === 0 ? 'animate-slide-down' : ''}`} onClick={() => onRowClick(tx)}>
                            <td className="px-4 py-2">{tx.time}</td>
                            <td className="px-4 py-2 font-mono">{tx.id}</td>
                            <td className="px-4 py-2">{tx.merchant}</td>
                            <td className="px-4 py-2 font-semibold">${tx.amount.toFixed(2)}</td>
                            <td className={`px-4 py-2 font-bold ${tx.status === 'Fraud' ? 'text-red-400' : 'text-green-400'}`}>{tx.status}</td>
                        </tr>
                    ))}
                </tbody>
            </table>
        </div>
    </Panel>
);

const SystemHealth: React.FC = () => {
    const healthData = [
        { name: 'PostgreSQL', status: 'Connected', latency: '12ms', icon: ICONS.database },
        { name: 'Kafka', status: 'Active', latency: '23k msg/s', icon: ICONS.server },
        { name: 'Spark Streaming', status: 'Running', latency: '2s batch', icon: ICONS.cpu },
        { name: 'ML Model v1.2', status: 'Loaded', latency: '94.3% Acc', icon: ICONS.shieldCheck },
    ];
    return (
         <Panel>
            <h3 className="text-lg font-bold text-inherit mb-4">System Health Monitor</h3>
            <ul className="space-y-3">
                {healthData.map(item => (
                    <li key={item.name} className="flex items-center justify-between text-sm">
                        <div className="flex items-center gap-3">
                            <item.icon className="w-5 h-5 text-[var(--accent-text)]"/>
                            <span>{item.name}</span>
                        </div>
                        <div className="text-right">
                           <span className="font-semibold text-green-400">{item.status}</span>
                           <span className="text-xs text-[var(--text-secondary-dark)] ml-2">{item.latency}</span>
                        </div>
                    </li>
                ))}
            </ul>
        </Panel>
    );
};

const ActivityLog: React.FC<{ logs: { time: string, message: string }[] }> = ({ logs }) => (
    <Panel>
        <h3 className="text-lg font-bold text-inherit mb-4">Recent Activity Log</h3>
        <ul className="space-y-2 text-sm">
            {logs.map((log, index) => (
                <li key={index} className="flex items-start gap-3">
                    <span className="text-[var(--text-secondary-dark)]">{log.time}</span>
                    <span className="text-gray-300">{log.message}</span>
                </li>
            ))}
        </ul>
    </Panel>
);

const Dashboard: React.FC = () => {
    // ✨ NEW: Use API hooks
    const { metrics: apiMetrics, loading: metricsLoading, error: metricsError } = useDashboardMetrics(5000);
    const { alerts: apiAlerts, loading: alertsLoading, error: alertsError } = useRecentAlerts(10, 2000);

    // Existing state
    const [transactions, setTransactions] = useState<Transaction[]>([]);
    const [kpis, setKpis] = useState<KPI[]>([]);
    const [alerts, setAlerts] = useState<Transaction[]>([]);
    const [activityLog, setActivityLog] = useState<{ time: string, message: string }[]>([]);
    const [toasts, setToasts] = useState<ToastType[]>([]);
    const [selectedTransaction, setSelectedTransaction] = useState<Transaction | null>(null);

    const addToast = (toast: Omit<ToastType, 'id'>) => {
        setToasts(prev => [...prev, { ...toast, id: Date.now() }]);
    };

    const removeToast = (id: number) => {
        setToasts(prev => prev.filter(t => t.id !== id));
    };

    // ✨ NEW: WebSocket for real-time fraud alerts
    const { lastMessage } = useWebSocket('ws://localhost:8000/ws/fraud-alerts');

    useEffect(() => {
        if (lastMessage) {
            try {
                const message = JSON.parse(lastMessage.data);
                
                if (message.type === 'fraud_alert' && message.data) {
                    const alert = convertAlertToTransaction(message.data);
                    
                    // Add to alerts (keep last 10)
                    setAlerts(prev => [alert, ...prev].slice(0, 10));
                    
                    // Add to transactions (keep last 20)
                    setTransactions(prev => [alert, ...prev].slice(0, 20));
                    
                    // Show toast notification
                    addToast({
                        title: '🚨 Real-time Fraud Alert!',
                        message: `${alert.customer} - $${alert.amount.toFixed(2)} at ${alert.merchant}`,
                        type: 'error'
                    });
                    
                    // Update activity log
                    setActivityLog(prev => [
                        { 
                            time: alert.time, 
                            message: `Real-time fraud detected: ${alert.id}` 
                        },
                        ...prev
                    ].slice(0, 5));
                }
            } catch (error) {
                console.error('Error parsing WebSocket message:', error);
            }
        }
    }, [lastMessage]);

    // ✨ NEW: Update KPIs from API metrics
    useEffect(() => {
        if (apiMetrics) {
            const newKpis: KPI[] = [
                { 
                    title: 'Total Transactions Today', 
                    value: apiMetrics.totalTransactions.toLocaleString(), 
                    details: 'All transactions processed', 
                    change: '+2.1%', 
                    changeType: 'increase' 
                },
                { 
                    title: 'Fraud Detected Today', 
                    value: `${apiMetrics.fraudDetected}`, 
                    details: `${apiMetrics.fraudRate.toFixed(2)}% of total`, 
                    change: '+5.4%', 
                    changeType: 'increase' 
                },
                { 
                    title: 'Normal Transactions', 
                    value: (apiMetrics.totalTransactions - apiMetrics.fraudDetected).toLocaleString(), 
                    details: `${(100 - apiMetrics.fraudRate).toFixed(2)}% of total`, 
                    change: '+1.8%', 
                    changeType: 'increase' 
                },
                { 
                    title: 'Model Accuracy', 
                    value: `${apiMetrics.accuracy.toFixed(2)}%`, 
                    details: 'Recall: 92.0%', 
                    change: '-0.2%', 
                    changeType: 'decrease' 
                },
            ];
            setKpis(newKpis);
        }
    }, [apiMetrics]);

    // ✨ NEW: Convert API alerts to transaction format
    useEffect(() => {
        if (apiAlerts && apiAlerts.length > 0) {
            const convertedAlerts = apiAlerts.map(convertAlertToTransaction);
            setAlerts(convertedAlerts);
            
            // Add toast for new alerts
            const latestAlert = convertedAlerts[0];
            if (latestAlert) {
                addToast({
                    title: 'High-Risk Transaction Detected',
                    message: `${latestAlert.customer} for $${latestAlert.amount.toFixed(2)} at ${latestAlert.merchant}.`,
                    type: 'error'
                });
            }
        }
    }, [apiAlerts]);

    // Fallback: Generate mock transactions if API is not available
    useEffect(() => {
        // Only use mock data if API is not returning data
        if (!metricsLoading && !apiMetrics) {
            const initialTxs = Array.from({ length: 10 }, generateTransaction);
            setTransactions(initialTxs);

            const interval = setInterval(() => {
                const newTx = generateTransaction();
                
                setTransactions(prev => [newTx, ...prev].slice(0, 20));
                
                if (newTx.status === 'Fraud') {
                    setAlerts(prev => [newTx, ...prev].slice(0, 10));
                    setActivityLog(prev => [{ time: newTx.time, message: `Fraud detected: ${newTx.id}` }, ...prev].slice(0, 5));
                    addToast({
                        title: 'High-Risk Transaction Detected',
                        message: `Tx ${newTx.id} for $${newTx.amount.toFixed(2)} at ${newTx.merchant}.`,
                        type: 'error'
                    });
                } else if (Math.random() > 0.5) {
                     setActivityLog(prev => [{ time: newTx.time, message: `Processed batch: ${newTx.id}` }, ...prev].slice(0, 5));
                }
            }, 2000);

            return () => clearInterval(interval);
        }
    }, [metricsLoading, apiMetrics]);

    // Update KPIs from mock transactions (fallback)
    useEffect(() => {
        if (!apiMetrics && transactions.length > 0) {
            const total = transactions.length;
            const fraud = transactions.filter(t => t.status === 'Fraud').length;
            const normal = total - fraud;
            
            const newKpis: KPI[] = [
                { title: 'Total Transactions Today', value: (1247 + total).toLocaleString(), details: 'All transactions processed', change: '+2.1%', changeType: 'increase' },
                { title: 'Fraud Detected Today', value: `${47 + fraud}`, details: `${((47 + fraud) / (1247 + total) * 100).toFixed(2)}% of total`, change: '+5.4%', changeType: 'increase' },
                { title: 'Normal Transactions', value: (1200 + normal).toLocaleString(), details: `${((1200 + normal) / (1247 + total) * 100).toFixed(2)}% of total`, change: '+1.8%', changeType: 'increase' },
                { title: 'Model Accuracy', value: '94.35%', details: 'Recall: 92.0%', change: '-0.2%', changeType: 'decrease' },
            ];
            setKpis(newKpis);
        }
    }, [transactions, apiMetrics]);

    const handleRowClick = (transaction: Transaction) => {
        setSelectedTransaction(transaction);
    };

    const handleCloseModal = () => {
        setSelectedTransaction(null);
    };

    // ✨ NEW: Show loading state
    if (metricsLoading && kpis.length === 0) {
        return (
            <div className="flex items-center justify-center h-screen">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-500 mx-auto mb-4"></div>
                    <p className="text-gray-400">Loading dashboard...</p>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="fixed top-24 right-4 z-[60] w-full max-w-sm">
                {toasts.map(toast => (
                    <Toast key={toast.id} toast={toast} onRemove={removeToast} />
                ))}
            </div>
            
            {/* ✨ NEW: Show API connection status */}
            {metricsError && (
                <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3 text-sm text-yellow-400">
                    ⚠️ Using mock data - Backend API not available
                </div>
            )}
            
             <TransactionDetailModal transaction={selectedTransaction} onClose={handleCloseModal} />
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {kpis.map(kpi => <MetricsCard key={kpi.title} {...kpi} />)}
            </div>

            <div className="grid grid-cols-12 gap-6">
                {/* Main Content */}
                <div className="col-span-12 lg:col-span-8 space-y-6">
                    <TransactionFeed transactions={transactions} onRowClick={handleRowClick} />
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <Panel><TransactionChart /></Panel>
                        <Panel><CreditDebitChart /></Panel>
                    </div>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                        <ActivityLog logs={activityLog} />
                        <Panel><SystemAdvicesChart /></Panel>
                    </div>
                </div>
                
                {/* Right Sidebar */}
                <div className="col-span-12 lg:col-span-4 space-y-6">
                    <RealTimeFeed alerts={alerts} />
                    <SystemHealth />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;

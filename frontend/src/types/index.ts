export type TransactionStatus = 'Normal' | 'Fraud';

export interface Transaction {
  id: string;
  time: string;
  customer: string;
  merchant: string;
  category: string;
  amount: number;
  distance: number;
  status: TransactionStatus;
  confidence?: number;
}

export interface KPI {
  title: string;
  value: string;
  details: string;
  change: string;
  changeType: 'increase' | 'decrease';
}

export interface Toast {
  id: number;
  title: string;
  message: string;
  type: 'error' | 'success';
}
"""
FastAPI Backend for Real-time Fraud Detection Dashboard
Connects to PostgreSQL and provides REST API + WebSocket for real-time updates
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import asyncio
import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import json

load_dotenv()

# ============================================================================
# Database Configuration
# ============================================================================

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'fraud_detection'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '12345')
}

def get_db_connection():
    """Get PostgreSQL connection"""
    try:
        return psycopg2.connect(**DB_CONFIG)
    except Exception as e:
        print(f"Database connection error: {e}")
        raise

# ============================================================================
# Pydantic Models
# ============================================================================

class DashboardMetrics(BaseModel):
    totalTransactions: int
    fraudDetected: int
    fraudRate: float
    accuracy: float

class FraudAlert(BaseModel):
    timestamp: str
    ccNum: str
    amount: float
    merchant: str
    confidence: float
    transNum: str

class Transaction(BaseModel):
    id: str
    time: str
    customer: str
    merchant: str
    category: str
    amount: float
    distance: float
    status: str
    confidence: Optional[float] = None

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Fraud Detection API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],  # React dev servers
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# WebSocket Connection Manager
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        print(f"✅ WebSocket connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
        print(f"❌ WebSocket disconnected. Total connections: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        """Send message to all connected clients"""
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                print(f"Error sending to client: {e}")
                disconnected.append(connection)
        
        # Remove disconnected clients
        for conn in disconnected:
            if conn in self.active_connections:
                self.active_connections.remove(conn)

manager = ConnectionManager()

# ============================================================================
# Database Query Functions
# ============================================================================

def get_dashboard_metrics():
    """Get dashboard metrics from database"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get total transactions today
        cursor.execute("""
            SELECT COUNT(*)::INT as total
            FROM (
                SELECT cc_num, trans_time FROM fraud_transaction
                WHERE trans_time >= CURRENT_DATE
                UNION ALL
                SELECT cc_num, trans_time FROM non_fraud_transaction
                WHERE trans_time >= CURRENT_DATE
            ) t
        """)
        total_result = cursor.fetchone()
        total_transactions = total_result['total'] if total_result else 0
        
        # Get fraud count today
        cursor.execute("""
            SELECT COUNT(*)::INT as fraud_count
            FROM fraud_transaction
            WHERE trans_time >= CURRENT_DATE
        """)
        fraud_result = cursor.fetchone()
        fraud_detected = fraud_result['fraud_count'] if fraud_result else 0
        
        # Calculate fraud rate
        fraud_rate = (fraud_detected / total_transactions * 100) if total_transactions > 0 else 0
        
        return {
            "totalTransactions": total_transactions,
            "fraudDetected": fraud_detected,
            "fraudRate": round(fraud_rate, 2),
            "accuracy": 94.35  # You can calculate this from your ML model
        }
    finally:
        cursor.close()
        conn.close()

def get_recent_fraud_alerts(limit: int = 10):
    """Get recent fraud transactions"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT 
                trans_time,
                trans_num,
                cc_num,
                amt,
                merchant,
                is_fraud,
                category,
                distance,
                created_at
            FROM fraud_transaction
            ORDER BY created_at DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        
        alerts = []
        for row in results:
            alerts.append({
                "timestamp": row['trans_time'].isoformat() if row['trans_time'] else datetime.now().isoformat(),
                "ccNum": f"**** **** **** {str(row['cc_num'])[-4:]}",
                "amount": float(row['amt']),
                "merchant": row['merchant'],
                "confidence": float(row['is_fraud']) * 100,
                "transNum": row['trans_num'],
                "category": row['category'],
                "distance": float(row['distance']) if row['distance'] else 0
            })
        
        return alerts
    finally:
        cursor.close()
        conn.close()

def get_all_transactions(limit: int = 50):
    """Get all recent transactions (fraud + non-fraud)"""
    conn = get_db_connection()
    cursor = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cursor.execute("""
            SELECT 
                trans_num as id,
                trans_time as time,
                cc_num as customer,
                merchant,
                category,
                amt as amount,
                distance,
                'Fraud' as status,
                is_fraud * 100 as confidence
            FROM fraud_transaction
            WHERE trans_time >= CURRENT_DATE - INTERVAL '1 day'
            
            UNION ALL
            
            SELECT 
                trans_num as id,
                trans_time as time,
                cc_num as customer,
                merchant,
                category,
                amt as amount,
                distance,
                'Normal' as status,
                NULL as confidence
            FROM non_fraud_transaction
            WHERE trans_time >= CURRENT_DATE - INTERVAL '1 day'
            
            ORDER BY time DESC
            LIMIT %s
        """, (limit,))
        
        results = cursor.fetchall()
        
        transactions = []
        for row in results:
            transactions.append({
                "id": row['id'],
                "time": row['time'].strftime('%H:%M:%S') if isinstance(row['time'], datetime) else str(row['time']),
                "customer": f"**** **** **** {str(row['customer'])[-4:]}",
                "merchant": row['merchant'],
                "category": row['category'],
                "amount": float(row['amount']),
                "distance": float(row['distance']) if row['distance'] else 0,
                "status": row['status'],
                "confidence": float(row['confidence']) if row['confidence'] else None
            })
        
        return transactions
    finally:
        cursor.close()
        conn.close()

# ============================================================================
# Background Task: Monitor for New Fraud Transactions
# ============================================================================

last_check_time = datetime.now()

async def monitor_fraud_transactions():
    """Background task to check for new fraud transactions and broadcast via WebSocket"""
    global last_check_time
    
    while True:
        try:
            await asyncio.sleep(5)  # Check every 5 seconds
            
            conn = get_db_connection()
            cursor = conn.cursor(cursor_factory=RealDictCursor)
            
            # Get new fraud transactions since last check
            cursor.execute("""
                SELECT 
                    trans_time,
                    trans_num,
                    cc_num,
                    amt,
                    merchant,
                    is_fraud,
                    category,
                    distance,
                    created_at
                FROM fraud_transaction
                WHERE created_at > %s
                ORDER BY created_at DESC
            """, (last_check_time,))
            
            new_frauds = cursor.fetchall()
            
            if new_frauds:
                print(f"🚨 Found {len(new_frauds)} new fraud transactions!")
                
                for fraud in new_frauds:
                    alert = {
                        "type": "fraud_alert",
                        "data": {
                            "timestamp": fraud['trans_time'].isoformat() if fraud['trans_time'] else datetime.now().isoformat(),
                            "ccNum": f"**** **** **** {str(fraud['cc_num'])[-4:]}",
                            "amount": float(fraud['amt']),
                            "merchant": fraud['merchant'],
                            "confidence": float(fraud['is_fraud']) * 100,
                            "transNum": fraud['trans_num'],
                            "category": fraud['category'],
                            "distance": float(fraud['distance']) if fraud['distance'] else 0
                        }
                    }
                    
                    # Broadcast to all connected WebSocket clients
                    await manager.broadcast(alert)
                
                # Update last check time
                last_check_time = max(fraud['created_at'] for fraud in new_frauds)
            
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"Error in monitor_fraud_transactions: {e}")
            await asyncio.sleep(10)

# ============================================================================
# REST API Endpoints
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "Fraud Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/api/health",
            "metrics": "/api/dashboard/metrics",
            "alerts": "/api/fraud/alerts",
            "transactions": "/api/transactions",
            "websocket": "/ws"
        }
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        conn = get_db_connection()
        conn.close()
        return {
            "status": "healthy",
            "database": "connected",
            "time": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

@app.get("/api/dashboard/metrics")
async def dashboard_metrics():
    """Get dashboard metrics"""
    try:
        metrics = get_dashboard_metrics()
        return metrics
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching metrics: {str(e)}")

@app.get("/api/fraud/alerts")
async def fraud_alerts(limit: int = 10):
    """Get recent fraud alerts"""
    try:
        alerts = get_recent_fraud_alerts(limit)
        return alerts
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching alerts: {str(e)}")

@app.get("/api/transactions")
async def transactions(limit: int = 50):
    """Get all recent transactions"""
    try:
        txs = get_all_transactions(limit)
        return txs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching transactions: {str(e)}")

# ============================================================================
# WebSocket Endpoint
# ============================================================================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time fraud alerts"""
    await manager.connect(websocket)
    
    try:
        # Send initial connection message
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to Fraud Detection API",
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive
        while True:
            # Wait for messages from client (ping/pong)
            data = await websocket.receive_text()
            
            # Echo back
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"WebSocket error: {e}")
        manager.disconnect(websocket)

# ============================================================================
# Startup Event
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Start background tasks on startup"""
    print("\n" + "="*60)
    print("🚀 Starting FastAPI Fraud Detection Server")
    print("="*60)
    print(f"📊 API Server: http://localhost:8000")
    print(f"📡 WebSocket: ws://localhost:8000/ws")
    print(f"📖 Docs: http://localhost:8000/docs")
    print(f"🔍 Health: http://localhost:8000/api/health")
    print("="*60 + "\n")
    
    # Start background task for monitoring fraud
    asyncio.create_task(monitor_fraud_transactions())

# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
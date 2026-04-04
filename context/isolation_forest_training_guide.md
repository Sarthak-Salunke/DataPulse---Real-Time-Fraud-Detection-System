# Isolation Forest Training Guide for Fraud Detection

## 📊 Dataset Summary
- **Total Transactions**: 11,871
- **Fraud Cases**: 501 (4.22%)
- **Normal Cases**: 11,370 (95.78%)
- **Customers**: 100
- **Files**: `customer.csv`, `transactions.csv`

---

## 🎯 Why Isolation Forest for Fraud Detection?

- ✅ **Unsupervised learning** - doesn't need fraud labels for training
- ✅ **Excellent for anomaly detection** - designed specifically for outliers
- ✅ **Handles high-dimensional data** efficiently
- ✅ **Fast training and prediction** - tree-based algorithm
- ✅ **Works well with imbalanced data** - focuses on rare events
- ✅ **No feature scaling required** - tree-based
- ⚠️ **Sensitive to contamination parameter** - needs tuning
- ⚠️ **Can struggle with complex patterns** - simpler than supervised models

**Expected Performance: ROC-AUC 0.70-0.78** (unsupervised!)

**Unique Advantage: Can find fraud patterns without labeled examples!**

---

## 🚀 Step-by-Step Training Process

### **STEP 1: Environment Setup and Imports**

```python
# fraud_detection_isolation_forest.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve, f1_score,
    recall_score, precision_score
)
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import time
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

print("=" * 80)
print("FRAUD DETECTION - ISOLATION FOREST (UNSUPERVISED)")
print("=" * 80)
print("\n⚡ Isolation Forest Characteristics:")
print("   • Unsupervised anomaly detection algorithm")
print("   • Isolates outliers by randomly partitioning data")
print("   • Based on the principle: anomalies are few and different")
print("   • Returns anomaly scores (not probabilities)")
print("   • No need for labeled training data")
print("   • Excellent for discovering unknown fraud patterns")
```

---

### **STEP 2: Load and Explore Data**

```python
# Load datasets
print("\n📁 Loading Data...")
customer_df = pd.read_csv('customer.csv')
transactions_df = pd.read_csv('transactions.csv')

print(f"   Customers: {len(customer_df)} records")
print(f"   Transactions: {len(transactions_df)} records")
print(f"   Known Fraud Rate: {transactions_df['is_fraud'].mean()*100:.2f}%")

print("\n💡 Isolation Forest Training Strategy:")
print("   • Can train on ALL data (supervised)")
print("   • Can train on NORMAL data only (semi-supervised)")
print("   • Can train on UNLABELED data (fully unsupervised)")
print("   We'll explore all three approaches!")

# Basic stats
print("\n📊 Transaction Amount Statistics:")
print(transactions_df.groupby('is_fraud')['amt'].describe())
```

---

### **STEP 3: Feature Engineering for Isolation Forest**

```python
def engineer_features_if(trans_df, cust_df):
    """
    Feature engineering for Isolation Forest
    
    Key considerations:
    - Focus on features that make fraud "different"
    - Include behavioral anomalies
    - Temporal patterns
    - Amount deviations
    - Location anomalies
    """
    print("\n🔧 Engineering Features for Isolation Forest...")
    
    # Merge datasets
    df = trans_df.merge(cust_df, on='cc_num', how='left', suffixes=('', '_cust'))
    
    # ------------------------
    # TEMPORAL FEATURES
    # ------------------------
    print("   ✓ Temporal features...")
    
    df['trans_datetime'] = pd.to_datetime(df['trans_date'])
    df['trans_time_obj'] = pd.to_datetime(df['trans_time'], format='%H:%M:%S')
    
    df['hour'] = df['trans_time_obj'].dt.hour
    df['day_of_week'] = df['trans_datetime'].dt.dayofweek
    df['day_of_month'] = df['trans_datetime'].dt.day
    df['month'] = df['trans_datetime'].dt.month
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Unusual time flags
    df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['is_early_morning'] = ((df['hour'] >= 1) & (df['hour'] <= 6)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # ------------------------
    # AMOUNT ANOMALIES
    # ------------------------
    print("   ✓ Amount-based anomaly features...")
    
    df['amt_original'] = df['amt']
    df['amt_log'] = np.log1p(df['amt'])
    df['amt_sqrt'] = np.sqrt(df['amt'])
    
    # Customer statistics
    customer_stats = df.groupby('cc_num')['amt'].agg([
        'mean', 'std', 'min', 'max', 'median', 'count'
    ]).reset_index()
    customer_stats.columns = ['cc_num', 'cust_amt_mean', 'cust_amt_std', 
                              'cust_amt_min', 'cust_amt_max', 'cust_amt_median', 'cust_trans_count']
    df = df.merge(customer_stats, on='cc_num', how='left')
    
    # Deviation features (key for anomaly detection!)
    df['amt_ratio_to_mean'] = df['amt'] / (df['cust_amt_mean'] + 1)
    df['amt_zscore'] = (df['amt'] - df['cust_amt_mean']) / (df['cust_amt_std'] + 1)
    df['amt_deviation_from_median'] = np.abs(df['amt'] - df['cust_amt_median'])
    df['amt_percentile_in_history'] = df['amt'] / (df['cust_amt_max'] + 1)
    
    # Extreme amount flags
    df['is_amount_extreme'] = (np.abs(df['amt_zscore']) > 3).astype(int)
    df['is_amount_high'] = (df['amt'] > df['cust_amt_mean'] + 2 * df['cust_amt_std']).astype(int)
    df['is_round_amount'] = (df['amt'] % 100 == 0).astype(int)
    df['is_very_round'] = (df['amt'] % 1000 == 0).astype(int)
    
    # ------------------------
    # GEOSPATIAL ANOMALIES
    # ------------------------
    print("   ✓ Geospatial anomaly features...")
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371  # Earth radius in km
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    # Distance from home
    df['distance_from_home'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    df['distance_log'] = np.log1p(df['distance_from_home'])
    
    # Customer location statistics
    customer_dist_stats = df.groupby('cc_num')['distance_from_home'].agg(['mean', 'std', 'max']).reset_index()
    customer_dist_stats.columns = ['cc_num', 'cust_dist_mean', 'cust_dist_std', 'cust_dist_max']
    df = df.merge(customer_dist_stats, on='cc_num', how='left')
    
    # Distance deviation
    df['distance_zscore'] = (df['distance_from_home'] - df['cust_dist_mean']) / (df['cust_dist_std'] + 1)
    df['is_distance_anomaly'] = (np.abs(df['distance_zscore']) > 2).astype(int)
    df['is_very_distant'] = (df['distance_from_home'] > 500).astype(int)
    
    # Geographic diversity
    df['merchant_lat_deviation'] = np.abs(df['merch_lat'] - df['lat'])
    df['merchant_long_deviation'] = np.abs(df['merch_long'] - df['long'])
    
    # ------------------------
    # BEHAVIORAL ANOMALIES
    # ------------------------
    print("   ✓ Behavioral anomaly features...")
    
    df = df.sort_values(['cc_num', 'unix_time']).reset_index(drop=True)
    
    # Transaction sequence
    df['trans_sequence'] = df.groupby('cc_num').cumcount() + 1
    df['is_first_trans'] = (df['trans_sequence'] == 1).astype(int)
    
    # Time gaps
    df['time_since_last'] = df.groupby('cc_num')['unix_time'].diff().fillna(0)
    df['time_since_last_hours'] = df['time_since_last'] / 3600
    df['time_since_last_log'] = np.log1p(df['time_since_last_hours'])
    
    # Time gap statistics per customer
    customer_time_stats = df[df['time_since_last_hours'] > 0].groupby('cc_num')['time_since_last_hours'].agg(['mean', 'std']).reset_index()
    customer_time_stats.columns = ['cc_num', 'cust_time_gap_mean', 'cust_time_gap_std']
    df = df.merge(customer_time_stats, on='cc_num', how='left')
    
    # Time gap deviation
    df['time_gap_zscore'] = (df['time_since_last_hours'] - df['cust_time_gap_mean']) / (df['cust_time_gap_std'] + 1)
    df['is_quick_succession'] = (df['time_since_last_hours'] < 0.5).astype(int)
    df['is_very_quick'] = (df['time_since_last_hours'] < 0.1).astype(int)
    
    # Transaction velocity
    df['prev_merch_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
    df['prev_merch_long'] = df.groupby('cc_num')['merch_long'].shift(1)
    df['distance_from_prev'] = haversine_distance(
        df['prev_merch_lat'].fillna(df['merch_lat']),
        df['prev_merch_long'].fillna(df['merch_long']),
        df['merch_lat'],
        df['merch_long']
    )
    
    df['velocity_kmh'] = df['distance_from_prev'] / (df['time_since_last_hours'] + 0.01)
    df['is_impossible_velocity'] = (df['velocity_kmh'] > 800).astype(int)  # Faster than plane
    df['is_very_fast_velocity'] = (df['velocity_kmh'] > 300).astype(int)
    
    # Transaction frequency
    df['trans_per_day'] = df['cust_trans_count'] / ((df['unix_time'].max() - df['unix_time'].min()) / 86400 + 1)
    
    # ------------------------
    # CATEGORY & MERCHANT ANOMALIES
    # ------------------------
    print("   ✓ Category and merchant anomaly features...")
    
    # Category usage
    category_counts = df.groupby(['cc_num', 'category']).size().reset_index(name='category_usage_count')
    df = df.merge(category_counts, on=['cc_num', 'category'], how='left')
    
    # Category diversity per customer
    category_diversity = df.groupby('cc_num')['category'].nunique().reset_index()
    category_diversity.columns = ['cc_num', 'category_diversity']
    df = df.merge(category_diversity, on='cc_num', how='left')
    
    df['is_new_category'] = (df['category_usage_count'] == 1).astype(int)
    df['category_concentration'] = df['category_usage_count'] / df['cust_trans_count']
    
    # Merchant usage
    merchant_counts = df.groupby(['cc_num', 'merchant']).size().reset_index(name='merchant_usage_count')
    df = df.merge(merchant_counts, on=['cc_num', 'merchant'], how='left')
    
    # Merchant diversity per customer
    merchant_diversity = df.groupby('cc_num')['merchant'].nunique().reset_index()
    merchant_diversity.columns = ['cc_num', 'merchant_diversity']
    df = df.merge(merchant_diversity, on='cc_num', how='left')
    
    df['is_new_merchant'] = (df['merchant_usage_count'] == 1).astype(int)
    df['merchant_concentration'] = df['merchant_usage_count'] / df['cust_trans_count']
    
    # Merchant novelty score
    df['novelty_score'] = df['is_new_category'].astype(int) + df['is_new_merchant'].astype(int)
    
    # ------------------------
    # COMPOSITE ANOMALY SCORES
    # ------------------------
    print("   ✓ Composite anomaly features...")
    
    # Combined z-scores (sum of absolute z-scores)
    df['combined_zscore'] = (
        np.abs(df['amt_zscore']) + 
        np.abs(df['distance_zscore']) + 
        np.abs(df['time_gap_zscore'].fillna(0))
    )
    
    # Anomaly flags count
    df['anomaly_flag_count'] = (
        df['is_amount_extreme'] + 
        df['is_distance_anomaly'] + 
        df['is_impossible_velocity'] +
        df['is_late_night'] +
        df['is_new_category'] +
        df['is_new_merchant']
    )
    
    # Risk indicators
    df['high_risk_combination'] = (
        (df['is_late_night'] == 1) & 
        (df['amt_ratio_to_mean'] > 2) & 
        (df['distance_from_home'] > 100)
    ).astype(int)
    
    print(f"\n   ✅ Feature Engineering Complete!")
    print(f"   Total Columns: {len(df.columns)}")
    
    return df

# Apply feature engineering
df_engineered = engineer_features_if(transactions_df, customer_df)

print(f"\n📊 Engineered Dataset Shape: {df_engineered.shape}")
```

---

### **STEP 4: Prepare Features for Isolation Forest**

```python
def prepare_features_for_if(df):
    """
    Select features optimized for Isolation Forest
    Focus on numerical features that capture anomalies
    """
    print("\n📋 Preparing Features for Isolation Forest...")
    
    # Select numerical features
    numerical_features = [
        # Amount features
        'amt_original', 'amt_log', 'amt_sqrt',
        'amt_ratio_to_mean', 'amt_zscore', 'amt_deviation_from_median',
        'amt_percentile_in_history',
        
        # Distance features
        'distance_from_home', 'distance_log', 'distance_zscore',
        'merchant_lat_deviation', 'merchant_long_deviation',
        
        # Temporal features
        'hour', 'day_of_week', 'day_of_month', 'month',
        'time_since_last_hours', 'time_since_last_log', 'time_gap_zscore',
        
        # Velocity features
        'distance_from_prev', 'velocity_kmh',
        
        # Behavioral features
        'trans_sequence', 'cust_trans_count', 'trans_per_day',
        'category_diversity', 'merchant_diversity',
        'category_usage_count', 'merchant_usage_count',
        'category_concentration', 'merchant_concentration',
        
        # Customer statistics
        'cust_amt_mean', 'cust_amt_std', 'cust_dist_mean',
        
        # Composite features
        'combined_zscore', 'anomaly_flag_count', 'novelty_score'
    ]
    
    # Binary features
    binary_features = [
        'is_weekend', 'is_late_night', 'is_early_morning', 'is_business_hours',
        'is_amount_extreme', 'is_amount_high', 'is_round_amount', 'is_very_round',
        'is_distance_anomaly', 'is_very_distant',
        'is_first_trans', 'is_quick_succession', 'is_very_quick',
        'is_impossible_velocity', 'is_very_fast_velocity',
        'is_new_category', 'is_new_merchant',
        'high_risk_combination'
    ]
    
    # Categorical features (one-hot encode)
    categorical_features = ['category', 'gender']
    
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Binary features: {len(binary_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    
    # Build feature matrix
    X = df[numerical_features + binary_features].copy()
    
    # One-hot encode categorical
    for cat_feat in categorical_features:
        if cat_feat in df.columns:
            dummies = pd.get_dummies(df[cat_feat], prefix=cat_feat, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Target (for evaluation purposes)
    y = df['is_fraud'].values
    
    print(f"\n   ✅ Feature Matrix Shape: {X.shape}")
    print(f"   Total Features: {X.shape[1]}")
    
    return X, y, X.columns.tolist()

X, y, feature_names = prepare_features_for_if(df_engineered)

print(f"\n📊 Dataset Summary:")
print(f"   Total samples: {len(X)}")
print(f"   Known fraud: {sum(y)} ({sum(y)/len(y)*100:.2f}%)")
print(f"   Features: {len(feature_names)}")
```

---

### **STEP 5: Train-Test Split**

```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✂️ Data Split:")
print(f"   Training: {len(X_train)} samples ({y_train.mean()*100:.2f}% fraud)")
print(f"   Testing: {len(X_test)} samples ({y_test.mean()*100:.2f}% fraud)")

print("\n💡 Training Strategy:")
print("   We have 3 options for Isolation Forest:")
print("   1. Train on ALL data (knows about fraud)")
print("   2. Train on NORMAL data only (semi-supervised)")
print("   3. Ignore labels completely (fully unsupervised)")
print("\n   We'll primarily use option 2 (normal data only) - most realistic!")
```

---

### **STEP 6: Train Baseline Isolation Forest**

```python
print("\n" + "="*80)
print("TRAINING BASELINE ISOLATION FOREST")
print("="*80)

print("\n🔧 Baseline Configuration:")
print("   n_estimators: 100 (number of trees)")
print("   contamination: 0.042 (expected fraud rate)")
print("   max_samples: 256 (subsample size)")
print("   random_state: 42")

# Train on normal data only (semi-supervised approach)
X_train_normal = X_train[y_train == 0]

print(f"\n📊 Training on NORMAL transactions only:")
print(f"   Normal samples: {len(X_train_normal)}")
print(f"   This is semi-supervised learning!")

print("\n⏳ Training Isolation Forest...")
start_time = time.time()

if_baseline = IsolationForest(
    n_estimators=100,
    contamination=0.042,  # Expected fraud rate
    max_samples=256,
    random_state=42,
    n_jobs=-1,
    verbose=0
)

if_baseline.fit(X_train_normal)

training_time = time.time() - start_time
print(f"✅ Training Complete in {training_time:.2f} seconds!")

print("\n📊 Model Info:")
print(f"   Number of trees: {if_baseline.n_estimators}")
print(f"   Max samples per tree: {if_baseline.max_samples}")
print(f"   Contamination: {if_baseline.contamination}")
```

---

### **STEP 7: Understand Isolation Forest Scores**

```python
print("\n" + "="*80)
print("UNDERSTANDING ISOLATION FOREST SCORES")
print("="*80)

# Get anomaly scores
train_scores = if_baseline.decision_function(X_train)
test_scores = if_baseline.decision_function(X_test)

# Get predictions (-1 for anomaly, 1 for normal)
train_pred = if_baseline.predict(X_train)
test_pred = if_baseline.predict(X_test)

print("\n📊 Score Distribution:")
print(f"   Training scores - Min: {train_scores.min():.4f}, Max: {train_scores.max():.4f}")
print(f"   Test scores - Min: {test_scores.min():.4f}, Max: {test_scores.max():.4f}")

print("\n💡 Understanding Scores:")
print("   • NEGATIVE scores = Anomalies (potential fraud)")
print("   • POSITIVE scores = Normal transactions")
print("   • More negative = More anomalous")

# Convert predictions (-1/1) to binary (1/0)
test_pred_binary = (test_pred == -1).astype(int)

print(f"\n📊 Predictions on Test Set:")
print(f"   Flagged as anomaly: {sum(test_pred_binary)}")
print(f"   Flagged as normal: {sum(test_pred_binary==0)}")
print(f"   Anomaly rate: {sum(test_pred_binary)/len(test_pred_binary)*100:.2f}%")

# Compare with actual fraud
print(f"\n📊 Actual vs Predicted:")
print(f"   Actual fraud: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.2f}%)")
print(f"   Predicted anomalies: {sum(test_pred_binary)} ({sum(test_pred_binary)/len(test_pred_binary)*100:.2f}%)")
```

---

### **STEP 8: Evaluate Baseline Model**

```python
def evaluate_isolation_forest(scores, y_true, model_name="Isolation Forest"):
    """
    Comprehensive evaluation for Isolation Forest
    Uses anomaly scores (not binary predictions) for better metrics
    """
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - EVALUATION")
    print(f"{'='*80}")
    
    # Convert scores to probabilities (invert and normalize)
    # More negative score = higher fraud probability
    scores_inverted = -scores
    scores_normalized = (scores_inverted - scores_inverted.min()) / (scores_inverted.max() - scores_inverted.min())
    
    # Use threshold at contamination level
    threshold = np.percentile(scores, 100 * (1 - 0.042))
    y_pred = (scores < threshold).astype(int)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Normal  Anomaly")
    print(f"Actual Normal   {tn:5d}  {fp:5d}")
    print(f"       Fraud    {fn:5d}  {tp:5d}")
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC scores (using normalized scores as probabilities)
    roc_auc = roc_auc_score(y_true, scores_normalized)
    precision_curve, recall_curve, _ = precision_recall_curve(y_true, scores_normalized)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy:     {accuracy:.4f}")
    print(f"   Precision:    {precision:.4f}  ({precision*100:.1f}% of anomalies are fraud)")
    print(f"   Recall:       {recall:.4f}  (Caught {recall*100:.1f}% of fraud)")
    print(f"   F1-Score:     {f1:.4f}")
    print(f"   Specificity:  {specificity:.4f}")
    print(f"   ROC-AUC:      {roc_auc:.4f} ⭐ (unsupervised!)")
    print(f"   PR-AUC:       {pr_auc:.4f}")
    
    print(f"\n💰 Business Metrics:")
    fpr = fp / (fp + tn) * 100
    fnr = fn / (fn + tp) * 100
    print(f"   False Positive Rate: {fpr:.2f}%")
    print(f"   False Negative Rate: {fnr:.2f}%")
    print(f"   Fraud Detection Rate: {recall*100:.1f}%")
    print(f"   Alert Rate: {(tp+fp)/len(y_true)*100:.2f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'scores': scores,
        'scores_normalized': scores_normalized,
        'predictions': y_pred
    }

# Evaluate baseline
baseline_results = evaluate_isolation_forest(test_scores, y_test, "Baseline Isolation Forest")
```

---

### **STEP 9: Hyperparameter Tuning**

```python
print("\n" + "="*80)
print("HYPERPARAMETER TUNING - ISOLATION FOREST")
print("="*80)

print("\n🔍 Key Hyperparameters to Tune:")
print("   • n_estimators: Number of isolation trees")
print("   • max_samples: Subsample size for each tree")
print("   • contamination: Expected proportion of outliers")
print("   • max_features: Features to consider for splitting")

# Define parameter grid
param_grid = {
    'n_estimators': [50, 100, 150, 200],
    'max_samples': [128, 256, 512, 'auto'],
    'contamination': [0.03, 0.042, 0.05, 0.06, 0.08],
    'max_features': [0.5, 0.75, 1.0]
}

print(f"\n📊 Testing {len(param_grid['n_estimators']) * len(param_grid['max_samples']) * len(param_grid['contamination']) * len(param_grid['max_features'])} combinations")

# Custom scorer for Isolation Forest
def if_scorer(estimator, X, y):
    """Custom scorer using anomaly scores"""
    scores = estimator.decision_function(X)
    scores_inverted = -scores
    scores_normalized = (scores_inverted - scores_inverted.min()) / (scores_inverted.max() - scores_inverted.min())
    return roc_auc_score(y, scores_normalized)

# Grid search
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

print("\n⏳ Running Grid Search (this will take 3-5 minutes)...")
start_time = time.time()

grid_search = GridSearchCV(
    IsolationForest(random_state=42, n_jobs=-1),
    param_grid,
    cv=cv_strategy,
    scoring=if_scorer,
    n_jobs=1,  # IF uses multiple cores internally
    verbose=1
)

# Train on normal data only
grid_search.fit(X_train_normal, y_train[y_train == 0])

search_time = time.time() - start_time
print(f"\n✅ Grid Search Complete in {search_time/60:.1f} minutes!")

print(f"\n🏆 Best Parameters:")
for param, value in grid_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Get best model
best_if = grid_search.best_estimator_

print(f"\n📊 Best Model Configuration:")
print(f"   Trees: {best_if.n_estimators}")
print(f"   Max samples: {best_if.max_samples}")
print(f"   Contamination: {best_if.contamination}")
print(f"   Max features: {best_if.max_features}")
```

---

### **STEP 10: Evaluate Optimized Model**

```python
print("\n" + "="*80)
print("OPTIMIZED ISOLATION FOREST - EVALUATION")
print("="*80)

# Get scores from optimized model
test_scores_optimized = best_if.decision_function(X_test)

# Evaluate
optimized_results = evaluate_isolation_forest(
    test_scores_optimized, y_test, "Optimized Isolation Forest"
)

# Compare with baseline
print(f"\n📊 IMPROVEMENT OVER BASELINE:")
print(f"   ROC-AUC:  {baseline_results['roc_auc']:.4f} → {optimized_results['roc_auc']:.4f} "
      f"({(optimized_results['roc_auc']-baseline_results['roc_auc'])*100:+.2f}%)")
print(f"   PR-AUC:   {baseline_results['pr_auc']:.4f} → {optimized_results['pr_auc']:.4f} "
      f"({(optimized_results['pr_auc']-baseline_results['pr_auc'])*100:+.2f}%)")
print(f"   F1-Score: {baseline_results['f1']:.4f} → {optimized_results['f1']:.4f} "
      f"({(optimized_results['f1']-baseline_results['f1'])*100:+.2f}%)")
print(f"   Recall:   {baseline_results['recall']:.4f} → {optimized_results['recall']:.4f} "
      f"({(optimized_results['recall']-baseline_results['recall'])*100:+.2f}%)")
```

---

### **STEP 11: Alternative Training Strategy - Fully Unsupervised**

```python
print("\n" + "="*80)
print("ALTERNATIVE: FULLY UNSUPERVISED TRAINING")
print("="*80)

print("\n💡 Training on ALL data (ignoring labels):")
print("   This simulates real-world scenario where labels are unavailable")

# Train on all data
print("\n⏳ Training on full dataset...")
if_unsupervised = IsolationForest(
    n_estimators=best_if.n_estimators,
    max_samples=best_if.max_samples,
    contamination=0.042,
    max_features=best_if.max_features,
    random_state=42,
    n_jobs=-1
)

if_unsupervised.fit(X_train)  # ALL training data
print("✅ Unsupervised model trained!")

# Evaluate
test_scores_unsupervised = if_unsupervised.decision_function(X_test)
unsupervised_results = evaluate_isolation_forest(
    test_scores_unsupervised, y_test, "Fully Unsupervised IF"
)

# Compare all three approaches
print(f"\n📊 COMPARISON OF TRAINING STRATEGIES:")
print(f"\n{'Strategy':<25} {'ROC-AUC':<10} {'PR-AUC':<10} {'Recall':<10}")
print("-" * 55)
print(f"{'Semi-supervised (normal)':<25} {baseline_results['roc_auc']:<10.4f} {baseline_results['pr_auc']:<10.4f} {baseline_results['recall']:<10.4f}")
print(f"{'Optimized (normal)':<25} {optimized_results['roc_auc']:<10.4f} {optimized_results['pr_auc']:<10.4f} {optimized_results['recall']:<10.4f}")
print(f"{'Fully unsupervised':<25} {unsupervised_results['roc_auc']:<10.4f} {unsupervised_results['pr_auc']:<10.4f} {unsupervised_results['recall']:<10.4f}")
```

---

### **STEP 12: Visualization - ROC and PR Curves**

```python
def plot_if_performance_curves(y_test, baseline_scores, optimized_scores, unsupervised_scores):
    """
    Plot ROC and PR curves for all Isolation Forest variants
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Normalize scores
    def normalize_scores(scores):
        scores_inv = -scores
        return (scores_inv - scores_inv.min()) / (scores_inv.max() - scores_inv.min())
    
    models = [
        ('Baseline IF', normalize_scores(baseline_scores), 'blue'),
        ('Optimized IF', normalize_scores(optimized_scores), 'green'),
        ('Unsupervised IF', normalize_scores(unsupervised_scores), 'red')
    ]
    
    # ROC Curve
    ax = axes[0]
    for name, scores_norm, color in models:
        fpr, tpr, _ = roc_curve(y_test, scores_norm)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
               label=f'{name} (AUC = {roc_auc:.4f})')
    
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves - Isolation Forest', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    
    # PR Curve
    ax = axes[1]
    for name, scores_norm, color in models:
        precision, recall, _ = precision_recall_curve(y_test, scores_norm)
        pr_auc = auc(recall, precision)
        ax.plot(recall, precision, color=color, lw=2,
               label=f'{name} (AUC = {pr_auc:.4f})')
    
    ax.axhline(y=y_test.mean(), color='k', linestyle='--', lw=2,
              label=f'Baseline ({y_test.mean():.4f})')
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('isolation_forest_performance_curves.png', dpi=300, bbox_inches='tight')
    print("\n📊 Performance curves saved to 'isolation_forest_performance_curves.png'")
    plt.close()

# Generate plots
plot_if_performance_curves(
    y_test, 
    baseline_results['scores'],
    optimized_results['scores'],
    unsupervised_results['scores']
)
```

---

### **STEP 13: Anomaly Score Distribution Analysis**

```python
def plot_anomaly_score_distribution(scores, y_true, model_name):
    """
    Visualize how anomaly scores separate fraud from normal
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Separate scores by class
    normal_scores = scores[y_true == 0]
    fraud_scores = scores[y_true == 1]
    
    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(normal_scores, bins=50, alpha=0.6, label='Normal', color='blue', density=True)
    ax.hist(fraud_scores, bins=50, alpha=0.6, label='Fraud', color='red', density=True)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Threshold (0)')
    ax.set_xlabel('Anomaly Score', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Score Distribution by Class', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Box Plot
    ax = axes[0, 1]
    bp = ax.boxplot([normal_scores, fraud_scores], labels=['Normal', 'Fraud'],
                     patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2)
    ax.set_ylabel('Anomaly Score', fontsize=11)
    ax.set_title('Score Distribution Comparison', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    
    # 3. Cumulative Distribution
    ax = axes[1, 0]
    sorted_normal = np.sort(normal_scores)
    sorted_fraud = np.sort(fraud_scores)
    ax.plot(sorted_normal, np.linspace(0, 1, len(sorted_normal)), 
           label='Normal', color='blue', linewidth=2)
    ax.plot(sorted_fraud, np.linspace(0, 1, len(sorted_fraud)), 
           label='Fraud', color='red', linewidth=2)
    ax.axvline(x=0, color='black', linestyle='--', linewidth=2, label='Threshold')
    ax.set_xlabel('Anomaly Score', fontsize=11)
    ax.set_ylabel('Cumulative Probability', fontsize=11)
    ax.set_title('Cumulative Distribution Functions', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Statistics Table
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_data = [
        ['Metric', 'Normal', 'Fraud'],
        ['Count', f'{len(normal_scores)}', f'{len(fraud_scores)}'],
        ['Mean', f'{normal_scores.mean():.4f}', f'{fraud_scores.mean():.4f}'],
        ['Std', f'{normal_scores.std():.4f}', f'{fraud_scores.std():.4f}'],
        ['Min', f'{normal_scores.min():.4f}', f'{fraud_scores.min():.4f}'],
        ['25%', f'{np.percentile(normal_scores, 25):.4f}', f'{np.percentile(fraud_scores, 25):.4f}'],
        ['Median', f'{np.median(normal_scores):.4f}', f'{np.median(fraud_scores):.4f}'],
        ['75%', f'{np.percentile(normal_scores, 75):.4f}', f'{np.percentile(fraud_scores, 75):.4f}'],
        ['Max', f'{normal_scores.max():.4f}', f'{fraud_scores.max():.4f}']
    ]
    
    table = ax.table(cellText=stats_data, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    for i in range(3):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax.set_title('Score Statistics Summary', fontsize=12, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig(f'isolation_forest_score_analysis.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Score analysis saved to 'isolation_forest_score_analysis.png'")
    plt.close()

# Analyze optimized model scores
print("\n📊 Analyzing Anomaly Score Distribution...")
plot_anomaly_score_distribution(
    optimized_results['scores'], 
    y_test, 
    "Optimized Isolation Forest"
)
```

---

### **STEP 14: Feature Importance (Approximation)**

```python
print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

print("\n💡 Note: Isolation Forest doesn't provide direct feature importance")
print("   We'll approximate by measuring impact on anomaly scores\n")

def approximate_feature_importance(model, X, feature_names, n_iterations=5):
    """
    Approximate feature importance by permutation
    """
    print("⏳ Computing feature importance (this may take 1-2 minutes)...")
    
    baseline_scores = model.decision_function(X)
    baseline_auc = roc_auc_score(y_test, -baseline_scores)
    
    importances = []
    
    for feat_idx, feat_name in enumerate(feature_names):
        if feat_idx % 10 == 0:
            print(f"   Processing feature {feat_idx+1}/{len(feature_names)}...")
        
        score_changes = []
        for _ in range(n_iterations):
            X_permuted = X.copy()
            X_permuted[:, feat_idx] = np.random.permutation(X_permuted[:, feat_idx])
            
            permuted_scores = model.decision_function(X_permuted)
            permuted_auc = roc_auc_score(y_test, -permuted_scores)
            
            score_changes.append(baseline_auc - permuted_auc)
        
        importance = np.mean(score_changes)
        importances.append(importance)
    
    return np.array(importances)

# Calculate importance
feature_importance = approximate_feature_importance(best_if, X_test.values, feature_names)

# Create DataFrame
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print("\n🔝 Top 20 Most Important Features:")
print("\n" + "-"*60)
print(f"{'Rank':<6} {'Feature':<40} {'Importance':<12}")
print("-"*60)
for idx, (_, row) in enumerate(importance_df.head(20).iterrows(), 1):
    print(f"{idx:<6} {row['feature']:<40} {row['importance']:>10.6f}")

# Save
importance_df.to_csv('isolation_forest_feature_importance.csv', index=False)
print("\n✅ Feature importance saved to 'isolation_forest_feature_importance.csv'")
```

---

### **STEP 15: Threshold Optimization**

```python
def optimize_if_threshold(scores, y_true):
    """
    Find optimal threshold for anomaly classification
    """
    print("\n🎯 Optimizing Classification Threshold...")
    
    # Test different percentiles as thresholds
    percentiles = np.arange(90, 99, 0.5)
    results = []
    
    for percentile in percentiles:
        threshold = np.percentile(scores, percentile)
        y_pred = (scores < threshold).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        results.append({
            'percentile': percentile,
            'threshold': threshold,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
    
    results_df = pd.DataFrame(results)
    
    # Find optimal for F1
    optimal_idx = results_df['f1'].idxmax()
    optimal_row = results_df.loc[optimal_idx]
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    ax.plot(results_df['percentile'], results_df['precision'], 'b-o', label='Precision', linewidth=2, markersize=4)
    ax.plot(results_df['percentile'], results_df['recall'], 'g-s', label='Recall', linewidth=2, markersize=4)
    ax.plot(results_df['percentile'], results_df['f1'], 'r-^', label='F1-Score', linewidth=2, markersize=4)
    
    ax.axvline(optimal_row['percentile'], color='purple', linestyle='--', linewidth=2,
              label=f"Optimal = {optimal_row['percentile']:.1f}th percentile")
    
    ax.set_xlabel('Score Percentile Threshold', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Threshold Optimization - Isolation Forest', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig('isolation_forest_threshold_optimization.png', dpi=300, bbox_inches='tight')
    print("   ✅ Threshold plot saved to 'isolation_forest_threshold_optimization.png'")
    plt.close()
    
    print(f"\n   Optimal Percentile: {optimal_row['percentile']:.1f}th")
    print(f"   Threshold Value: {optimal_row['threshold']:.4f}")
    print(f"   F1-Score: {optimal_row['f1']:.4f}")
    print(f"   Precision: {optimal_row['precision']:.4f}")
    print(f"   Recall: {optimal_row['recall']:.4f}")
    
    return optimal_row['threshold'], results_df

# Optimize threshold
optimal_threshold, threshold_results = optimize_if_threshold(
    optimized_results['scores'], y_test
)
```

---

### **STEP 16: Save Models and Artifacts**

```python
print("\n" + "="*80)
print("SAVING MODELS AND ARTIFACTS")
print("="*80)

# Save all models
models_to_save = {
    'baseline': if_baseline,
    'optimized': best_if,
    'unsupervised': if_unsupervised
}

for name, model in models_to_save.items():
    filename = f'isolation_forest_{name}_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ {name.capitalize()} IF saved to '{filename}'")

# Save feature names
with open('isolation_forest_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved to 'isolation_forest_feature_names.pkl'")

# Save optimal threshold
with open('isolation_forest_optimal_threshold.pkl', 'wb') as f:
    pickle.dump(optimal_threshold, f)
print("✅ Optimal threshold saved to 'isolation_forest_optimal_threshold.pkl'")

# Save comprehensive results
results_summary = {
    'model_type': 'Isolation Forest (Unsupervised)',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_strategy': 'Semi-supervised (normal data only)',
    'best_parameters': grid_search.best_params_,
    'dataset_info': {
        'total_samples': len(df_engineered),
        'train_samples': len(X_train_normal),
        'test_samples': len(X_test),
        'n_features': len(feature_names),
        'contamination': best_if.contamination
    },
    'performance': {
        'baseline': {
            'roc_auc': baseline_results['roc_auc'],
            'pr_auc': baseline_results['pr_auc'],
            'f1': baseline_results['f1'],
            'recall': baseline_results['recall']
        },
        'optimized': {
            'roc_auc': optimized_results['roc_auc'],
            'pr_auc': optimized_results['pr_auc'],
            'f1': optimized_results['f1'],
            'recall': optimized_results['recall']
        },
        'unsupervised': {
            'roc_auc': unsupervised_results['roc_auc'],
            'pr_auc': unsupervised_results['pr_auc'],
            'f1': unsupervised_results['f1'],
            'recall': unsupervised_results['recall']
        }
    },
    'optimal_threshold': optimal_threshold
}

with open('isolation_forest_results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print("✅ Results summary saved to 'isolation_forest_results_summary.pkl'")

print("\n" + "="*80)
print("ISOLATION FOREST TRAINING COMPLETE!")
print("="*80)

print(f"\n📦 Generated Files:")
print(f"   1. isolation_forest_baseline_model.pkl")
print(f"   2. isolation_forest_optimized_model.pkl (recommended)")
print(f"   3. isolation_forest_unsupervised_model.pkl")
print(f"   4. isolation_forest_feature_names.pkl")
print(f"   5. isolation_forest_optimal_threshold.pkl")
print(f"   6. isolation_forest_results_summary.pkl")
print(f"   7. isolation_forest_feature_importance.csv")
print(f"   8. isolation_forest_performance_curves.png")
print(f"   9. isolation_forest_score_analysis.png")
print(f"   10. isolation_forest_threshold_optimization.png")

print(f"\n🎯 Best Model Performance (Optimized IF):")
print(f"   ROC-AUC: {optimized_results['roc_auc']:.4f} ⭐ (Unsupervised!)")
print(f"   PR-AUC: {optimized_results['pr_auc']:.4f}")
print(f"   F1-Score: {optimized_results['f1']:.4f}")
print(f"   Recall: {optimized_results['recall']:.4f}")
print(f"   Training Strategy: Semi-supervised (normal data only)")

print("\n💡 Key Insight:")
print("   Isolation Forest achieved {:.1f}% ROC-AUC without using fraud labels!".format(
    optimized_results['roc_auc'] * 100
))

print("\n" + "="*80)
```

---

## 🎯 Expected Results

### **Performance Benchmarks:**

| Training Strategy | ROC-AUC | PR-AUC | F1-Score | Use Case |
|------------------|---------|--------|----------|----------|
| **Semi-supervised** (normal only) | 0.72-0.78 | 0.38-0.50 | 0.48-0.62 | Best performance |
| **Fully unsupervised** (all data) | 0.68-0.74 | 0.35-0.45 | 0.45-0.58 | No labels available |
| **Baseline** | 0.70-0.76 | 0.36-0.48 | 0.46-0.60 | Quick start |

### **Training Time:**
- Baseline: ~10-15 seconds
- Grid Search: ~3-5 minutes
- Feature Importance: ~1-2 minutes
- **Total: ~5-8 minutes**

### **Key Advantages:**
✅ **Unsupervised** - doesn't need fraud labels
✅ **Finds unknown patterns** - discovers new fraud types
✅ **Fast training** - tree-based algorithm
✅ **No feature scaling** required
✅ **Works with imbalanced data** naturally
✅ **Robust to outliers** - designed for them!

### **Limitations:**
⚠️ Lower performance than supervised methods
⚠️ Sensitive to contamination parameter
⚠️ Struggles with subtle patterns
⚠️ No clear feature importance
⚠️ Can be affected by noisy data

---

## 🚀 Running the Complete Script

Save all code to `fraud_detection_isolation_forest.py` and run:

```bash
python fraud_detection_isolation_forest.py
```

**Estimated completion time: 5-8 minutes**

---

## 🔄 Quick Deployment Example

```python
# deploy_isolation_forest.py
import pickle
import numpy as np

# Load model and threshold
with open('isolation_forest_optimized_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('isolation_forest_optimal_threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)

def detect_anomaly_if(transaction_features):
    """
    Fast anomaly detection using Isolation Forest
    
    Args:
        transaction_features: numpy array with engineered features
    
    Returns:
        dict with anomaly score and classification
    """
    # Get anomaly score
    anomaly_score = model.decision_function(transaction_features.reshape(1, -1))[0]
    
    # Classify (negative = anomaly)
    is_anomaly = anomaly_score < threshold
    
    # Normalize score to 0-1 range (higher = more anomalous)
    anomaly_score_normalized = max(0, -anomaly_score) / abs(threshold)
    
    return {
        'anomaly_score': anomaly_score,
        'anomaly_score_normalized': min(1.0, anomaly_score_normalized),
        'is_anomaly': bool(is_anomaly),
        'confidence': 'HIGH' if abs(anomaly_score) > abs(threshold) * 1.5 else 'MEDIUM' if abs(anomaly_score) > abs(threshold) else 'LOW'
    }

# Example usage
# result = detect_anomaly_if(X_test[0])
# print(result)
```

---

## 📊 Comparison with Previous Models

| Metric | Logistic Regression | SVM | Naive Bayes | Isolation Forest | Winner |
|--------|-------------------|-----|-------------|------------------|--------|
| **ROC-AUC** | 0.75-0.82 | 0.80-0.86 | 0.68-0.75 | 0.70-0.78 | SVM |
| **PR-AUC** | 0.40-0.55 | 0.45-0.65 | 0.35-0.50 | 0.38-0.50 | SVM |
| **Training Time** | 5 min | 15-20 min | < 1 min | 5-8 min | NB |
| **Needs Labels?** | Yes | Yes | Yes | **No** ⭐ | **IF** |
| **Finds New Patterns?** | No | No | No | **Yes** ⭐ | **IF** |
| **Interpretability** | High | Low | Medium | Low | LR |
| **Best Use Case** | Baseline | Complex | Speed | **Unknown fraud** | - |

**Unique Value: Isolation Forest is the ONLY unsupervised model - can discover new fraud patterns without labels!**

---

## 🐛 Troubleshooting

### **Issue 1: Low recall (<50%)**
**Solutions:**
- Increase `contamination` parameter (e.g., 0.06 instead of 0.042)
- Lower classification threshold
- Add more anomaly-focused features
- Try training on all data (unsupervised)

### **Issue 2: Too many false positives**
**Solutions:**
- Decrease `contamination` parameter
- Increase classification threshold
- Train only on normal data (semi-supervised)
- Add more behavioral features

### **Issue 3: "Memory Error"**
**Solution:**
```python
# Reduce max_samples to limit memory
model = IsolationForest(max_samples=128, ...)
```

### **Issue 4: Want better performance**
**Solution:** Isolation Forest is unsupervised. For better performance:
- XGBoost (supervised): ROC-AUC 0.88-0.93
- LightGBM (supervised): ROC-AUC 0.87-0.92
- Or use IF as feature (anomaly score) in supervised model

---

## ✅ Checklist

- [ ] Install libraries: `pip install scikit-learn matplotlib seaborn`
- [ ] Place CSV files in directory
- [ ] Run complete script
- [ ] Check ROC-AUC > 0.70 (excellent for unsupervised!)
- [ ] Review anomaly score distributions
- [ ] Test all 3 training strategies
- [ ] Save best model (usually semi-supervised)
- [ ] Note: No labels needed for training!

---

## 🎓 Key Learnings

1. **Contamination parameter is critical** - should match expected fraud rate
2. **Training on normal data works best** (semi-supervised)
3. **Anomaly scores matter more** than binary predictions
4. **Threshold optimization improves results** significantly
5. **Great for discovering unknown fraud** patterns
6. **Can work without ANY labels** - truly unsupervised
7. **Fast and memory-efficient** compared to deep learning

---

## 📈 Model Rankings Updated

| Rank | Model | ROC-AUC | Supervised? | Best For |
|------|-------|---------|-------------|----------|
| 🥇 | **SVM** | 0.80-0.86 | Yes | Accuracy |
| 🥈 | **Logistic Regression** | 0.75-0.82 | Yes | Interpretability |
| 🥉 | **Isolation Forest** | 0.70-0.78 | **No** ⭐ | **Unknown patterns** |
| 4️⃣ | Naive Bayes | 0.68-0.75 | Yes | Speed |

---

## 🚀 Next Steps

You've now trained 4 different approaches! Ready for the **BIG performers**?

### **Highly Recommended Next:**
1. **XGBoost** ⭐⭐⭐ (Expected: 0.88-0.93 ROC-AUC) - Industry leader
2. **LightGBM** ⚡ (Expected: 0.87-0.92 ROC-AUC) - Faster than XGBoost
3. **Gradient Boosting** (Expected: 0.85-0.91 ROC-AUC) - Classic ensemble

### **Or Continue with:**
4. **Neural Networks (MLP)** 🧠 (Expected: 0.82-0.88 ROC-AUC)
5. **Random Forest** 🌲 (Expected: 0.82-0.88 ROC-AUC)
6. **Ensemble Stacking** 🎯 (Combine all models!)

**Let me know which powerhouse model you want next!** 🚀

---

**Pro Tip:** Use Isolation Forest anomaly scores as an additional feature in supervised models (XGBoost, etc.) - this often improves performance by 2-3%! 🎯

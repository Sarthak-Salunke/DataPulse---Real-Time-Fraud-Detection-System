# Support Vector Machine (SVM) Training Guide for Fraud Detection

## 📊 Dataset Summary
- **Total Transactions**: 11,871
- **Fraud Cases**: 501 (4.22%)
- **Normal Cases**: 11,370 (95.78%)
- **Customers**: 100
- **Files**: `customer.csv`, `transactions.csv`

---

## 🎯 Why SVM for Fraud Detection?

- ✅ **Non-linear decision boundaries** via kernel trick
- ✅ **Effective in high-dimensional spaces** (many features)
- ✅ **Robust to outliers** due to margin-based optimization
- ✅ **Memory efficient** (uses subset of training points as support vectors)
- ⚠️ **Computationally expensive** for large datasets
- ⚠️ **Requires careful feature scaling**

**Expected Performance: ROC-AUC 0.80-0.86**

---

## 🚀 Step-by-Step Training Process

### **STEP 1: Environment Setup and Imports**

```python
# fraud_detection_svm.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_curve, auc, roc_curve, f1_score,
    recall_score, precision_score, make_scorer
)
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# For handling class imbalance
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# Set random seed
np.random.seed(42)

print("=" * 80)
print("FRAUD DETECTION - SUPPORT VECTOR MACHINE (SVM) MODEL")
print("=" * 80)
print("\n⚡ SVM Characteristics:")
print("   • Uses kernel trick for non-linear classification")
print("   • Finds optimal hyperplane with maximum margin")
print("   • Excellent for complex decision boundaries")
print("   • Requires feature scaling (critical!)")
```

---

### **STEP 2: Load and Prepare Data**

```python
# Load datasets
print("\n📁 Loading Data...")
customer_df = pd.read_csv('customer.csv')
transactions_df = pd.read_csv('transactions.csv')

print(f"   Customers: {len(customer_df)} records")
print(f"   Transactions: {len(transactions_df)} records")
print(f"   Fraud Rate: {transactions_df['is_fraud'].mean()*100:.2f}%")

# Quick data quality check
print("\n🔍 Data Quality Check:")
print(f"   Customer missing values: {customer_df.isnull().sum().sum()}")
print(f"   Transaction missing values: {transactions_df.isnull().sum().sum()}")
print(f"   Duplicate transactions: {transactions_df.duplicated(subset='trans_num').sum()}")
```

---

### **STEP 3: Comprehensive Feature Engineering**

```python
def engineer_features(trans_df, cust_df):
    """
    Create rich feature set for SVM
    SVMs benefit from well-engineered, scaled features
    """
    print("\n🔧 Engineering Features for SVM...")
    
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
    
    # Cyclical encoding for hour (important for SVM!)
    # Convert hour to circular features (0-23 hours wrap around)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Day of week cyclical
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # High-risk time windows
    df['is_late_night'] = ((df['hour'] >= 23) | (df['hour'] <= 5)).astype(int)
    df['is_business_hours'] = ((df['hour'] >= 9) & (df['hour'] <= 17)).astype(int)
    
    # ------------------------
    # AMOUNT FEATURES
    # ------------------------
    print("   ✓ Amount-based features...")
    
    df['amt_log'] = np.log1p(df['amt'])
    df['amt_sqrt'] = np.sqrt(df['amt'])
    
    # Customer statistics
    customer_stats = df.groupby('cc_num')['amt'].agg([
        'mean', 'std', 'min', 'max', 'median', 'count'
    ]).reset_index()
    customer_stats.columns = ['cc_num', 'cust_amt_mean', 'cust_amt_std', 
                              'cust_amt_min', 'cust_amt_max', 'cust_amt_median', 'cust_trans_count']
    df = df.merge(customer_stats, on='cc_num', how='left')
    
    # Derived amount features
    df['amt_ratio_to_avg'] = df['amt'] / (df['cust_amt_mean'] + 1)
    df['amt_zscore'] = (df['amt'] - df['cust_amt_mean']) / (df['cust_amt_std'] + 1)
    df['amt_above_median'] = (df['amt'] > df['cust_amt_median']).astype(int)
    df['amt_percentile'] = df['amt'] / (df['cust_amt_max'] + 1)
    
    # Amount range features
    df['is_high_amount'] = (df['amt'] > df['amt'].quantile(0.75)).astype(int)
    df['is_very_high_amount'] = (df['amt'] > df['amt'].quantile(0.95)).astype(int)
    
    # ------------------------
    # GEOSPATIAL FEATURES
    # ------------------------
    print("   ✓ Geospatial features...")
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        """Calculate haversine distance in km"""
        R = 6371
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c
    
    df['distance_from_home'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    df['distance_log'] = np.log1p(df['distance_from_home'])
    df['is_local'] = (df['distance_from_home'] < 10).astype(int)
    df['is_distant'] = (df['distance_from_home'] > 100).astype(int)
    df['is_very_distant'] = (df['distance_from_home'] > 500).astype(int)
    
    # Latitude/Longitude bins (regional patterns)
    df['merchant_lat_bin'] = pd.cut(df['merch_lat'], bins=10, labels=False)
    df['merchant_long_bin'] = pd.cut(df['merch_long'], bins=10, labels=False)
    
    # ------------------------
    # BEHAVIORAL FEATURES
    # ------------------------
    print("   ✓ Behavioral features...")
    
    df = df.sort_values(['cc_num', 'unix_time']).reset_index(drop=True)
    
    # Transaction sequence number
    df['trans_sequence'] = df.groupby('cc_num').cumcount() + 1
    
    # Time gaps
    df['time_since_last'] = df.groupby('cc_num')['unix_time'].diff().fillna(0)
    df['time_since_last_hours'] = df['time_since_last'] / 3600
    df['time_since_last_log'] = np.log1p(df['time_since_last_hours'])
    
    # Quick succession flags
    df['is_very_quick'] = (df['time_since_last'] < 300).astype(int)  # < 5 min
    df['is_quick'] = (df['time_since_last'] < 3600).astype(int)  # < 1 hour
    
    # Distance from previous transaction
    df['prev_merch_lat'] = df.groupby('cc_num')['merch_lat'].shift(1)
    df['prev_merch_long'] = df.groupby('cc_num')['merch_long'].shift(1)
    df['distance_from_prev'] = haversine_distance(
        df['prev_merch_lat'].fillna(df['merch_lat']),
        df['prev_merch_long'].fillna(df['merch_long']),
        df['merch_lat'],
        df['merch_long']
    )
    
    # Velocity (distance / time)
    df['transaction_velocity'] = df['distance_from_prev'] / (df['time_since_last_hours'] + 0.01)
    df['is_impossible_velocity'] = (df['transaction_velocity'] > 800).astype(int)  # > 800 km/h
    
    # ------------------------
    # CATEGORY & MERCHANT
    # ------------------------
    print("   ✓ Category and merchant features...")
    
    # Category frequency per customer
    category_counts = df.groupby(['cc_num', 'category']).size().reset_index(name='category_usage_count')
    df = df.merge(category_counts, on=['cc_num', 'category'], how='left')
    
    # Top category per customer
    top_category = df.groupby('cc_num')['category'].agg(
        lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
    ).reset_index()
    top_category.columns = ['cc_num', 'top_category']
    df = df.merge(top_category, on='cc_num', how='left')
    
    df['is_top_category'] = (df['category'] == df['top_category']).astype(int)
    df['is_new_category'] = (df['category_usage_count'] == 1).astype(int)
    
    # Merchant frequency
    merchant_counts = df.groupby(['cc_num', 'merchant']).size().reset_index(name='merchant_usage_count')
    df = df.merge(merchant_counts, on=['cc_num', 'merchant'], how='left')
    df['is_new_merchant'] = (df['merchant_usage_count'] == 1).astype(int)
    
    # ------------------------
    # RISK AGGREGATES
    # ------------------------
    print("   ✓ Risk aggregation features...")
    
    # Merchant fraud rate
    merchant_fraud = df.groupby('merchant')['is_fraud'].agg(['mean', 'count']).reset_index()
    merchant_fraud.columns = ['merchant', 'merchant_fraud_rate', 'merchant_total_trans']
    df = df.merge(merchant_fraud, on='merchant', how='left')
    df['merchant_fraud_rate'] = df['merchant_fraud_rate'].fillna(df['is_fraud'].mean())
    
    # Category fraud rate
    category_fraud = df.groupby('category')['is_fraud'].agg(['mean', 'count']).reset_index()
    category_fraud.columns = ['category', 'category_fraud_rate', 'category_total_trans']
    df = df.merge(category_fraud, on='category', how='left')
    
    # Hour fraud rate
    hour_fraud = df.groupby('hour')['is_fraud'].mean().reset_index()
    hour_fraud.columns = ['hour', 'hour_fraud_rate']
    df = df.merge(hour_fraud, on='hour', how='left')
    
    # High-risk combinations
    df['merchant_category_risk'] = df['merchant_fraud_rate'] * df['category_fraud_rate']
    df['time_amount_risk'] = df['hour_fraud_rate'] * df['amt_zscore']
    
    print(f"\n   ✅ Feature Engineering Complete!")
    print(f"   Total Columns: {len(df.columns)}")
    
    return df

# Apply feature engineering
df_engineered = engineer_features(transactions_df, customer_df)
```

---

### **STEP 4: Feature Selection for SVM**

```python
def prepare_svm_features(df):
    """
    Select and prepare optimal features for SVM
    SVM works best with carefully selected, scaled numerical features
    """
    print("\n📋 Preparing Features for SVM...")
    
    # Numerical features (continuous)
    numerical_features = [
        # Amount features
        'amt', 'amt_log', 'amt_sqrt', 'amt_ratio_to_avg', 'amt_zscore', 
        'amt_percentile',
        
        # Temporal features (cyclical encoded)
        'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
        'day_of_month', 'month',
        
        # Distance features
        'distance_from_home', 'distance_log', 'distance_from_prev',
        'transaction_velocity',
        
        # Behavioral features
        'trans_sequence', 'time_since_last_hours', 'time_since_last_log',
        'cust_trans_count', 'category_usage_count', 'merchant_usage_count',
        
        # Risk features
        'merchant_fraud_rate', 'category_fraud_rate', 'hour_fraud_rate',
        'merchant_category_risk', 'merchant_total_trans',
        
        # Bins
        'merchant_lat_bin', 'merchant_long_bin'
    ]
    
    # Binary features
    binary_features = [
        'is_weekend', 'is_late_night', 'is_business_hours',
        'amt_above_median', 'is_high_amount', 'is_very_high_amount',
        'is_local', 'is_distant', 'is_very_distant',
        'is_very_quick', 'is_quick', 'is_impossible_velocity',
        'is_top_category', 'is_new_category', 'is_new_merchant'
    ]
    
    # Categorical features to encode
    categorical_features = ['category', 'gender']
    
    print(f"   Numerical features: {len(numerical_features)}")
    print(f"   Binary features: {len(binary_features)}")
    print(f"   Categorical features: {len(categorical_features)}")
    
    # Build feature matrix
    X = df[numerical_features + binary_features].copy()
    
    # One-hot encode categorical
    for cat_feature in categorical_features:
        if cat_feature in df.columns:
            dummies = pd.get_dummies(df[cat_feature], prefix=cat_feature, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
    
    # Handle missing values
    X = X.fillna(X.median())
    
    # Target
    y = df['is_fraud'].values
    
    print(f"\n   ✅ Feature Matrix Shape: {X.shape}")
    print(f"   Total Features: {X.shape[1]}")
    print(f"   Fraud Rate: {y.mean()*100:.2f}%")
    
    return X, y, X.columns.tolist()

X, y, feature_names = prepare_svm_features(df_engineered)

print(f"\n📊 Final Dataset:")
print(f"   Samples: {len(X)}")
print(f"   Features: {len(feature_names)}")
print(f"   Normal: {sum(y==0)} ({sum(y==0)/len(y)*100:.1f}%)")
print(f"   Fraud: {sum(y==1)} ({sum(y==1)/len(y)*100:.1f}%)")
```

---

### **STEP 5: Train-Test Split**

```python
# Stratified split to maintain fraud ratio
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"\n✂️ Data Split:")
print(f"   Training: {len(X_train)} samples ({y_train.mean()*100:.2f}% fraud)")
print(f"   Testing: {len(X_test)} samples ({y_test.mean()*100:.2f}% fraud)")
```

---

### **STEP 6: Feature Scaling (CRITICAL for SVM!)**

```python
print("\n" + "="*80)
print("FEATURE SCALING (CRITICAL FOR SVM)")
print("="*80)

print("\n⚠️ SVM is extremely sensitive to feature scales!")
print("   Using StandardScaler for zero mean and unit variance\n")

# StandardScaler: (x - mean) / std
scaler = StandardScaler()

# Fit on training data only
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Scaling Applied:")
print(f"   Training set - Mean: {X_train_scaled.mean():.6f}, Std: {X_train_scaled.std():.6f}")
print(f"   Test set - Mean: {X_test_scaled.mean():.6f}, Std: {X_test_scaled.std():.6f}")

# Verify scaling
print(f"\n📊 Sample Feature Ranges After Scaling:")
sample_features = X_train.columns[:5]
for i, feat in enumerate(sample_features):
    print(f"   {feat}: [{X_train_scaled[:, i].min():.2f}, {X_train_scaled[:, i].max():.2f}]")
```

---

### **STEP 7: Handle Class Imbalance**

```python
print("\n" + "="*80)
print("HANDLING CLASS IMBALANCE")
print("="*80)

print(f"\n📊 Original Distribution:")
print(f"   Normal: {sum(y_train==0)}")
print(f"   Fraud: {sum(y_train==1)}")
print(f"   Ratio: 1:{sum(y_train==0)/sum(y_train==1):.1f}")

# Combined SMOTE + Undersampling
print("\n⚖️ Applying SMOTE + Random Undersampling...")

# SMOTE: Oversample minority to 50% of majority
smote = SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

# Undersampling: Reduce majority to create 1:2 ratio
under = RandomUnderSampler(sampling_strategy=0.67, random_state=42)
X_train_balanced, y_train_balanced = under.fit_resample(X_train_smote, y_train_smote)

print(f"\n✅ Balanced Distribution:")
print(f"   Normal: {sum(y_train_balanced==0)}")
print(f"   Fraud: {sum(y_train_balanced==1)}")
print(f"   Ratio: 1:{sum(y_train_balanced==0)/sum(y_train_balanced==1):.1f}")
print(f"   Total Training Samples: {len(X_train_balanced)}")
```

---

### **STEP 8: Train Baseline SVM with RBF Kernel**

```python
print("\n" + "="*80)
print("TRAINING BASELINE SVM MODEL")
print("="*80)

print("\n🔧 Baseline Configuration:")
print("   Kernel: RBF (Radial Basis Function)")
print("   C: 1.0 (regularization)")
print("   gamma: 'scale' (automatic)")
print("   class_weight: 'balanced'")

# Train baseline SVM
svm_baseline = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,  # Enable probability estimates
    random_state=42,
    verbose=False
)

print("\n⏳ Training baseline SVM (this may take 2-3 minutes)...")
import time
start_time = time.time()

svm_baseline.fit(X_train_balanced, y_train_balanced)

training_time = time.time() - start_time
print(f"✅ Training Complete in {training_time:.1f} seconds")

# Number of support vectors
print(f"\n📊 Support Vector Statistics:")
print(f"   Total Support Vectors: {svm_baseline.n_support_.sum()}")
print(f"   Class 0 (Normal): {svm_baseline.n_support_[0]}")
print(f"   Class 1 (Fraud): {svm_baseline.n_support_[1]}")
print(f"   Percentage of training data: {svm_baseline.n_support_.sum()/len(X_train_balanced)*100:.1f}%")
```

---

### **STEP 9: Evaluate Baseline Model**

```python
def evaluate_svm_model(model, X_test_scaled, y_test, model_name="SVM"):
    """
    Comprehensive evaluation for SVM model
    """
    print(f"\n{'='*80}")
    print(f"{model_name.upper()} - TEST SET EVALUATION")
    print(f"{'='*80}")
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n📊 Confusion Matrix:")
    print(f"                 Predicted")
    print(f"               Normal  Fraud")
    print(f"Actual Normal   {tn:5d}  {fp:5d}")
    print(f"       Fraud    {fn:5d}  {tp:5d}")
    
    # Metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC scores
    roc_auc = roc_auc_score(y_test, y_proba)
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recall_curve, precision_curve)
    
    print(f"\n📈 Performance Metrics:")
    print(f"   Accuracy:     {accuracy:.4f}")
    print(f"   Precision:    {precision:.4f}  ({precision*100:.1f}% of flagged are fraud)")
    print(f"   Recall:       {recall:.4f}  (Caught {recall*100:.1f}% of fraud)")
    print(f"   F1-Score:     {f1:.4f}")
    print(f"   Specificity:  {specificity:.4f}  ({specificity*100:.1f}% normal correctly classified)")
    print(f"   ROC-AUC:      {roc_auc:.4f}")
    print(f"   PR-AUC:       {pr_auc:.4f}  ⭐")
    
    print(f"\n💰 Business Metrics:")
    fpr = fp / (fp + tn) * 100
    fnr = fn / (fn + tp) * 100
    print(f"   False Positive Rate: {fpr:.2f}% ({fp} normal flagged)")
    print(f"   False Negative Rate: {fnr:.2f}% ({fn} fraud missed)")
    print(f"   Fraud Detection Rate: {recall*100:.1f}%")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_proba
    }

# Evaluate baseline
baseline_results = evaluate_svm_model(svm_baseline, X_test_scaled, y_test, "Baseline SVM")
```

---

### **STEP 10: Hyperparameter Tuning with RandomizedSearchCV**

```python
print("\n" + "="*80)
print("HYPERPARAMETER TUNING - RANDOMIZED SEARCH")
print("="*80)

print("\n⚠️ Note: Full GridSearch would test 1000+ combinations and take hours!")
print("   Using RandomizedSearchCV for efficient tuning (100 iterations)\n")

# Define parameter distribution
param_distributions = {
    'C': [0.1, 1, 10, 50, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly'],
    'degree': [2, 3, 4],  # Only used for poly kernel
    'class_weight': ['balanced', {0: 1, 1: 20}, {0: 1, 1: 30}]
}

print("🔍 Search Space:")
for param, values in param_distributions.items():
    print(f"   {param}: {values}")

# Custom scorer for imbalanced data
pr_auc_scorer = make_scorer(
    lambda y_true, y_pred: auc(*precision_recall_curve(y_true, y_pred)[:2][::-1]),
    needs_proba=True
)

# Stratified K-Fold
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# RandomizedSearchCV
random_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    param_distributions=param_distributions,
    n_iter=50,  # Test 50 random combinations
    cv=cv_strategy,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2,
    random_state=42
)

print("\n⏳ Running Randomized Search (this will take 10-15 minutes)...")
print("   Testing 50 random parameter combinations with 3-fold CV")
print("   Estimated time: 10-15 minutes\n")

start_time = time.time()
random_search.fit(X_train_balanced, y_train_balanced)
search_time = time.time() - start_time

print(f"\n✅ Search Complete in {search_time/60:.1f} minutes!")

print(f"\n🏆 Best Parameters Found:")
for param, value in random_search.best_params_.items():
    print(f"   {param}: {value}")

print(f"\n📊 Cross-Validation Performance:")
print(f"   Best CV ROC-AUC: {random_search.best_score_:.4f}")
print(f"   Mean CV Score: {random_search.cv_results_['mean_test_score'].mean():.4f}")
print(f"   Std CV Score: {random_search.cv_results_['std_test_score'].mean():.4f}")

# Get best model
best_svm = random_search.best_estimator_

print(f"\n📊 Best Model Support Vectors:")
print(f"   Total: {best_svm.n_support_.sum()}")
print(f"   Class 0: {best_svm.n_support_[0]}")
print(f"   Class 1: {best_svm.n_support_[1]}")
```

---

### **STEP 11: Evaluate Optimized Model**

```python
print("\n" + "="*80)
print("OPTIMIZED SVM PERFORMANCE")
print("="*80)

optimized_results = evaluate_svm_model(best_svm, X_test_scaled, y_test, "Optimized SVM")

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

### **STEP 12: Probability Calibration (Optional but Recommended)**

```python
print("\n" + "="*80)
print("PROBABILITY CALIBRATION")
print("="*80)

print("\n📊 Why Calibrate?")
print("   SVM's raw probabilities may not reflect true probabilities")
print("   Calibration using Platt scaling improves probability estimates")

# Calibrate probabilities
print("\n⏳ Calibrating probabilities with Platt scaling...")
calibrated_svm = CalibratedClassifierCV(best_svm, method='sigmoid', cv=3)
calibrated_svm.fit(X_train_balanced, y_train_balanced)

print("✅ Calibration Complete!")

# Evaluate calibrated model
calibrated_results = evaluate_svm_model(calibrated_svm, X_test_scaled, y_test, "Calibrated SVM")

print(f"\n📊 CALIBRATED vs OPTIMIZED:")
print(f"   ROC-AUC:  {optimized_results['roc_auc']:.4f} → {calibrated_results['roc_auc']:.4f}")
print(f"   PR-AUC:   {optimized_results['pr_auc']:.4f} → {calibrated_results['pr_auc']:.4f}")
```

---

### **STEP 13: Visualization - ROC and PR Curves**

```python
def plot_svm_performance_curves(y_test, y_proba_baseline, y_proba_optimized, y_proba_calibrated):
    """
    Plot ROC and Precision-Recall curves for all models
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # ROC Curve
    models = [
        ('Baseline SVM', y_proba_baseline, 'blue'),
        ('Optimized SVM', y_proba_optimized, 'green'),
        ('Calibrated SVM', y_proba_calibrated, 'red')
    ]
    
    for name, proba, color in models:
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_auc = auc(fpr, tpr)
        axes[0].plot(fpr, tpr, color=color, lw=2, 
                     label=f'{name} (AUC = {roc_auc:.4f})')
    
    axes[0].plot([0, 1], [0, 1], 'k--', lw=2, label='Random')
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves - SVM Models', fontsize=14, fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    for name, proba, color in models:
        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)
        axes[1].plot(recall, precision, color=color, lw=2,
                     label=f'{name} (AUC = {pr_auc:.4f})')
    
    axes[1].axhline(y=y_test.mean(), color='k', linestyle='--', lw=2,
                    label=f'Baseline (Fraud Rate = {y_test.mean():.4f})')
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curves - SVM Models', fontsize=14, fontweight='bold')
    axes[1].legend(loc='best')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('svm_performance_curves.png', dpi=300, bbox_inches='tight')
    print("\n📊 Performance curves saved to 'svm_performance_curves.png'")
    plt.close()

# Generate plots
plot_svm_performance_curves(
    y_test,
    baseline_results['probabilities'],
    optimized_results['probabilities'],
    calibrated_results['probabilities']
)
```

---

### **STEP 14: Decision Boundary Visualization (2D Projection)**

```python
def visualize_svm_decision_boundary(model, X_test_scaled, y_test, feature_names):
    """
    Visualize SVM decision boundary using 2 most important features
    """
    from sklearn.decomposition import PCA
    
    print("\n📊 Visualizing Decision Boundary...")
    print("   Using PCA to project to 2D for visualization")
    
    # Project to 2D using PCA
    pca = PCA(n_components=2, random_state=42)
    X_test_2d = pca.fit_transform(X_test_scaled)
    
    # Create mesh grid
    h = 0.02
    x_min, x_max = X_test_2d[:, 0].min() - 1, X_test_2d[:, 0].max() + 1
    y_min, y_max = X_test_2d[:, 1].min() - 1, X_test_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # Train SVM on 2D projection
    svm_2d = SVC(kernel='rbf', C=best_svm.C, gamma='scale', probability=True, random_state=42)
    svm_2d.fit(X_test_2d, y_test)
    
    # Predict on mesh
    Z = svm_2d.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn_r', levels=20)
    plt.colorbar(label='Fraud Probability')
    
    # Plot points
    scatter = plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], 
                         c=y_test, cmap='RdYlGn_r', 
                         edgecolors='black', linewidth=0.5, s=50, alpha=0.7)
    
    plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)', fontsize=12)
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)', fontsize=12)
    plt.title('SVM Decision Boundary (2D PCA Projection)', fontsize=14, fontweight='bold')
    plt.legend(*scatter.legend_elements(), title="Class", labels=['Normal', 'Fraud'])
    plt.grid(True, alpha=0.3)
    
    plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
    print("   ✅ Decision boundary saved to 'svm_decision_boundary.png'")
    plt.close()

# Visualize decision boundary
visualize_svm_decision_boundary(best_svm, X_test_scaled, y_test, feature_names)
```

---

### **STEP 15: Threshold Optimization**

```python
def optimize_threshold(y_test, y_proba, metric='f1'):
    """
    Find optimal classification threshold
    """
    print(f"\n🎯 Optimizing Threshold for {metric.upper()}...")
    
    thresholds = np.arange(0.1, 0.9, 0.05)
    scores = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_test, y_pred)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred)
        
        scores.append(score)
    
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_score = scores[optimal_idx]
    
    # Plot threshold vs score
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, scores, 'b-', linewidth=2, marker='o')
    plt.axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
                label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Classification Threshold', fontsize=12)
    plt.ylabel(f'{metric.upper()} Score', fontsize=12)
    plt.title(f'Threshold Optimization for {metric.upper()}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'svm_threshold_optimization_{metric}.png', dpi=300, bbox_inches='tight')
    print(f"   ✅ Threshold plot saved to 'svm_threshold_optimization_{metric}.png'")
    plt.close()
    
    print(f"\n   Optimal Threshold: {optimal_threshold:.2f}")
    print(f"   {metric.upper()} Score: {optimal_score:.4f}")
    
    return optimal_threshold, optimal_score

# Optimize for F1-score
optimal_threshold_f1, optimal_f1 = optimize_threshold(
    y_test, calibrated_results['probabilities'], metric='f1'
)

# Evaluate with optimal threshold
y_pred_optimal = (calibrated_results['probabilities'] >= optimal_threshold_f1).astype(int)

print(f"\n📊 Performance with Optimized Threshold ({optimal_threshold_f1:.2f}):")
cm = confusion_matrix(y_test, y_pred_optimal)
tn, fp, fn, tp = cm.ravel()

print(f"   Precision: {tp/(tp+fp):.4f}")
print(f"   Recall: {tp/(tp+fn):.4f}")
print(f"   F1-Score: {optimal_f1:.4f}")
print(f"   Fraud Detection Rate: {tp/(tp+fn)*100:.1f}%")
print(f"   False Positive Rate: {fp/(fp+tn)*100:.2f}%")
```

---

### **STEP 16: Model Comparison Summary**

```python
print("\n" + "="*80)
print("FINAL MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame({
    'Model': ['Baseline SVM', 'Optimized SVM', 'Calibrated SVM'],
    'ROC-AUC': [
        baseline_results['roc_auc'],
        optimized_results['roc_auc'],
        calibrated_results['roc_auc']
    ],
    'PR-AUC': [
        baseline_results['pr_auc'],
        optimized_results['pr_auc'],
        calibrated_results['pr_auc']
    ],
    'F1-Score': [
        baseline_results['f1'],
        optimized_results['f1'],
        calibrated_results['f1']
    ],
    'Recall': [
        baseline_results['recall'],
        optimized_results['recall'],
        calibrated_results['recall']
    ],
    'Precision': [
        baseline_results['precision'],
        optimized_results['precision'],
        calibrated_results['precision']
    ]
})

print("\n📊 Performance Comparison:")
print(comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('svm_model_comparison.csv', index=False)
print("\n✅ Comparison saved to 'svm_model_comparison.csv'")
```

---

### **STEP 17: Save Models and Artifacts**

```python
print("\n" + "="*80)
print("SAVING MODELS AND ARTIFACTS")
print("="*80)

# Save best model (calibrated)
with open('svm_fraud_model_calibrated.pkl', 'wb') as f:
    pickle.dump(calibrated_svm, f)
print("\n✅ Calibrated SVM saved to 'svm_fraud_model_calibrated.pkl'")

# Save optimized model (non-calibrated)
with open('svm_fraud_model_optimized.pkl', 'wb') as f:
    pickle.dump(best_svm, f)
print("✅ Optimized SVM saved to 'svm_fraud_model_optimized.pkl'")

# Save scaler
with open('svm_feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Feature scaler saved to 'svm_feature_scaler.pkl'")

# Save feature names
with open('svm_feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)
print("✅ Feature names saved to 'svm_feature_names.pkl'")

# Save optimal threshold
with open('svm_optimal_threshold.pkl', 'wb') as f:
    pickle.dump(optimal_threshold_f1, f)
print("✅ Optimal threshold saved to 'svm_optimal_threshold.pkl'")

# Save comprehensive results
results_summary = {
    'model_type': 'Support Vector Machine (SVM)',
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'kernel': random_search.best_params_['kernel'],
    'best_parameters': random_search.best_params_,
    'dataset_info': {
        'total_samples': len(df_engineered),
        'train_samples': len(X_train_balanced),
        'test_samples': len(X_test),
        'n_features': len(feature_names),
        'fraud_rate': y.mean()
    },
    'performance': {
        'baseline': {
            'roc_auc': baseline_results['roc_auc'],
            'pr_auc': baseline_results['pr_auc'],
            'f1': baseline_results['f1']
        },
        'optimized': {
            'roc_auc': optimized_results['roc_auc'],
            'pr_auc': optimized_results['pr_auc'],
            'f1': optimized_results['f1']
        },
        'calibrated': {
            'roc_auc': calibrated_results['roc_auc'],
            'pr_auc': calibrated_results['pr_auc'],
            'f1': calibrated_results['f1']
        }
    },
    'optimal_threshold': optimal_threshold_f1,
    'support_vectors': {
        'total': int(best_svm.n_support_.sum()),
        'class_0': int(best_svm.n_support_[0]),
        'class_1': int(best_svm.n_support_[1])
    }
}

with open('svm_results_summary.pkl', 'wb') as f:
    pickle.dump(results_summary, f)
print("✅ Results summary saved to 'svm_results_summary.pkl'")

print("\n" + "="*80)
print("SVM TRAINING COMPLETE!")
print("="*80)

print(f"\n📦 Generated Files:")
print(f"   1. svm_fraud_model_calibrated.pkl (recommended for deployment)")
print(f"   2. svm_fraud_model_optimized.pkl")
print(f"   3. svm_feature_scaler.pkl")
print(f"   4. svm_feature_names.pkl")
print(f"   5. svm_optimal_threshold.pkl")
print(f"   6. svm_results_summary.pkl")
print(f"   7. svm_model_comparison.csv")
print(f"   8. svm_performance_curves.png")
print(f"   9. svm_decision_boundary.png")
print(f"   10. svm_threshold_optimization_f1.png")

print(f"\n🎯 Best Model Performance (Calibrated SVM):")
print(f"   ROC-AUC: {calibrated_results['roc_auc']:.4f}")
print(f"   PR-AUC: {calibrated_results['pr_auc']:.4f}")
print(f"   F1-Score: {calibrated_results['f1']:.4f}")
print(f"   Recall: {calibrated_results['recall']:.4f}")
print(f"   Precision: {calibrated_results['precision']:.4f}")

print("\n" + "="*80)
```

---

## 🎯 Expected Results

### **Performance Benchmarks:**
- **ROC-AUC**: 0.80 - 0.86
- **PR-AUC**: 0.45 - 0.65
- **F1-Score**: 0.55 - 0.75
- **Recall**: 65-85%
- **Precision**: 45-65%

### **Training Time:**
- Feature Engineering: 2-3 minutes
- Baseline Training: 2-3 minutes
- Hyperparameter Search: 10-15 minutes
- **Total: ~15-20 minutes**

### **Key Advantages of SVM:**
✅ Better than Logistic Regression for non-linear patterns
✅ Robust to outliers
✅ Effective in high-dimensional spaces
✅ Provides probability estimates after calibration

### **Limitations:**
⚠️ Slower training than tree-based models
⚠️ Memory intensive for large datasets
⚠️ Requires careful feature scaling
⚠️ Less interpretable than decision trees

---

## 🚀 Running the Complete Script

Save all code to `fraud_detection_svm.py` and run:

```bash
python fraud_detection_svm.py
```

---

## 🔄 Model Deployment - Quick Inference Example

```python
# load_and_predict.py
import pickle
import numpy as np

# Load saved models
with open('svm_fraud_model_calibrated.pkl', 'rb') as f:
    model = pickle.load(f)

with open('svm_feature_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('svm_optimal_threshold.pkl', 'rb') as f:
    threshold = pickle.load(f)

# Example: Predict on new transaction
def predict_fraud(new_transaction_features):
    """
    new_transaction_features: numpy array with same features as training
    """
    # Scale features
    features_scaled = scaler.transform(new_transaction_features.reshape(1, -1))
    
    # Get probability
    fraud_probability = model.predict_proba(features_scaled)[0, 1]
    
    # Classify using optimal threshold
    is_fraud = fraud_probability >= threshold
    
    return {
        'fraud_probability': fraud_probability,
        'is_fraud': is_fraud,
        'confidence': 'HIGH' if abs(fraud_probability - 0.5) > 0.3 else 'LOW'
    }

# Test prediction
# prediction = predict_fraud(X_test_scaled[0])
# print(prediction)
```

---

## 📊 Comparison with Logistic Regression

| Metric | Logistic Regression | SVM (RBF Kernel) | Winner |
|--------|-------------------|------------------|--------|
| ROC-AUC | 0.75 - 0.82 | 0.80 - 0.86 | **SVM** |
| PR-AUC | 0.40 - 0.55 | 0.45 - 0.65 | **SVM** |
| Training Time | Fast (~5 min) | Slower (~15-20 min) | LR |
| Prediction Speed | Very Fast | Fast | LR |
| Interpretability | High | Low | LR |
| Non-linear Patterns | Limited | Excellent | **SVM** |
| Memory Usage | Low | High | LR |

**Conclusion**: SVM typically outperforms Logistic Regression by 3-5% in AUC metrics but requires more computational resources.

---

## 🐛 Troubleshooting

### **Issue 1: "Memory Error" during training**
**Solution**: Reduce training set size or use `LinearSVC` instead:
```python
from sklearn.svm import LinearSVC
model = LinearSVC(class_weight='balanced', max_iter=1000)
```

### **Issue 2: Training is too slow**
**Solution**: 
- Reduce RandomizedSearchCV iterations: `n_iter=20`
- Use fewer CV folds: `cv=3` instead of `cv=5`
- Consider using only RBF kernel, skip poly

### **Issue 3: Low performance (<0.75 ROC-AUC)**
**Solution**:
- Check feature scaling was applied
- Increase SMOTE ratio
- Try different kernels
- Add more engineered features

### **Issue 4: High false positive rate**
**Solution**:
- Increase classification threshold (0.6 or 0.7)
- Adjust `class_weight` parameter
- Focus on precision optimization

---

## ✅ Checklist

- [ ] Install libraries: `pip install scikit-learn imbalanced-learn matplotlib seaborn`
- [ ] Place CSV files in directory
- [ ] Run complete script
- [ ] Check ROC-AUC > 0.80
- [ ] Review performance curves
- [ ] Compare with Logistic Regression results
- [ ] Save best model for deployment
- [ ] Document hyperparameters

---

## 🎓 Key Learnings

1. **Feature Scaling is MANDATORY** for SVM - never skip this step
2. **Kernel choice matters** - RBF usually best for fraud detection
3. **Probability calibration** improves reliability of fraud scores
4. **Support vectors** tell you model complexity (fewer = more generalizable)
5. **Threshold tuning** is critical for business goals (precision vs recall trade-off)

---

## 📈 Next Steps

After completing SVM, you can move to:

1. **Tree-Based Models** (XGBoost, LightGBM) - typically best performance (AUC: 0.88-0.93)
2. **Neural Networks** - for deep pattern learning
3. **Ensemble Methods** - combine SVM with other models
4. **Isolation Forest** - unsupervised anomaly detection

**Let me know when you're ready for the next algorithm!** 🚀

---

**Pro Tip**: SVM works best when combined with tree-based models in an ensemble. Save this model for stacking later!

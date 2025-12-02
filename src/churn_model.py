"""
Advanced OTT Streaming Analytics - Churn Prediction Model
Trains and evaluates multiple models for churn prediction
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def feature_engineering(users_df, watch_df):
    """Create advanced features for churn prediction"""
    # Calculate recency, frequency, duration (RFM) metrics
    watch_df['watch_date'] = pd.to_datetime(watch_df['watch_date'])
    users_df['last_active_date'] = pd.to_datetime(users_df['last_active_date'])
    
    # Recency: Days since last watch
    if len(watch_df) > 0 and 'watch_date' in watch_df.columns and 'user_id' in watch_df.columns:
        try:
            last_watch = watch_df.groupby('user_id')['watch_date'].max().reset_index()
            last_watch.columns = ['user_id', 'last_watch_date']
            # Ensure user_id columns match types
            if 'user_id' in users_df.columns:
                users_df = users_df.merge(last_watch, on='user_id', how='left', suffixes=('', '_watch'))
                if 'last_watch_date' in users_df.columns:
                    users_df['days_since_last_watch'] = (pd.Timestamp.now() - users_df['last_watch_date']).dt.days
                    users_df['days_since_last_watch'] = users_df['days_since_last_watch'].fillna(999)
                    # Drop the temporary column
                    users_df = users_df.drop(columns=['last_watch_date'], errors='ignore')
                else:
                    users_df['days_since_last_watch'] = 999
            else:
                users_df['days_since_last_watch'] = 999
        except Exception as e:
            print(f"Warning: Error calculating days_since_last_watch: {e}")
            users_df['days_since_last_watch'] = 999
    else:
        users_df['days_since_last_watch'] = 999
    
    # Frequency: Number of watches in last 30 days
    if len(watch_df) > 0:
        recent_date = pd.Timestamp.now() - pd.Timedelta(days=30)
        recent_watches = watch_df[watch_df['watch_date'] >= recent_date].groupby('user_id').size().reset_index()
        recent_watches.columns = ['user_id', 'watches_last_30_days']
        users_df = users_df.merge(recent_watches, on='user_id', how='left', suffixes=('', '_recent'))
        if 'watches_last_30_days' in users_df.columns:
            users_df['watches_last_30_days'] = users_df['watches_last_30_days'].fillna(0)
        else:
            users_df['watches_last_30_days'] = 0
    else:
        users_df['watches_last_30_days'] = 0
    
    # Duration: Total watch time
    if len(watch_df) > 0 and 'watch_time_minutes' in watch_df.columns:
        total_watch_time = watch_df.groupby('user_id')['watch_time_minutes'].sum().reset_index()
        total_watch_time.columns = ['user_id', 'total_watch_time']
        users_df = users_df.merge(total_watch_time, on='user_id', how='left', suffixes=('', '_watch'))
        if 'total_watch_time' in users_df.columns:
            users_df['total_watch_time'] = users_df['total_watch_time'].fillna(0)
        else:
            users_df['total_watch_time'] = 0
    else:
        users_df['total_watch_time'] = 0
    
    # Average rating given
    if len(watch_df) > 0 and 'rating' in watch_df.columns:
        avg_rating = watch_df.groupby('user_id')['rating'].mean().reset_index()
        avg_rating.columns = ['user_id', 'avg_rating_given']
        users_df = users_df.merge(avg_rating, on='user_id', how='left', suffixes=('', '_rating'))
        if 'avg_rating_given' in users_df.columns:
            users_df['avg_rating_given'] = users_df['avg_rating_given'].fillna(0)
        else:
            users_df['avg_rating_given'] = 0
    else:
        users_df['avg_rating_given'] = 0
    
    # Days since join
    users_df['join_date'] = pd.to_datetime(users_df['join_date'])
    users_df['days_since_join'] = (pd.Timestamp.now() - users_df['join_date']).dt.days
    
    return users_df

def prepare_features(users_df):
    """Prepare features for modeling"""
    df = users_df.copy()
    
    # Select features
    categorical_features = ['subscription_plan', 'device_type', 'gender', 'country', 'preferred_genre']
    numerical_features = ['age', 'avg_watch_time', 'sessions_per_week', 
                         'days_since_last_watch', 'watches_last_30_days', 
                         'total_watch_time', 'avg_rating_given', 'days_since_join']
    
    # Check which categorical features exist
    available_categorical = [col for col in categorical_features if col in df.columns]
    missing_categorical = [col for col in categorical_features if col not in df.columns]
    if missing_categorical:
        print(f"Warning: Missing categorical features: {missing_categorical}")
        # Fill missing with default values
        for col in missing_categorical:
            df[col] = 'unknown'
    
    # Check which numerical features exist
    available_numerical = [col for col in numerical_features if col in df.columns]
    missing_numerical = [col for col in numerical_features if col not in df.columns]
    if missing_numerical:
        print(f"Warning: Missing numerical features: {missing_numerical}")
        # Fill missing with default values
        for col in missing_numerical:
            df[col] = 0
    
    # Encode categorical variables
    le_dict = {}
    for col in categorical_features:
        if col in df.columns:
            le = LabelEncoder()
            df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
            le_dict[col] = le
        else:
            df[col + '_encoded'] = 0
            le_dict[col] = None
    
    # Combine all features
    feature_cols = [f + '_encoded' for f in categorical_features] + numerical_features
    # Only use features that exist
    available_features = [col for col in feature_cols if col in df.columns]
    X = df[available_features].fillna(0)
    
    # Check if churn column exists
    if 'churn' not in df.columns:
        raise ValueError("'churn' column not found in users_df. Make sure the dataset includes churn information.")
    y = df['churn']
    
    return X, y, le_dict

def train_and_evaluate_models(X, y):
    """Train and evaluate multiple models"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        # Use scaled data for Logistic Regression, original for tree-based
        if name == 'Logistic Regression':
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        trained_models[name] = model
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        print(f"  ROC-AUC: {roc_auc:.4f}")
    
    return results, trained_models, scaler, X_test, X_test_scaled

def plot_confusion_matrices(results):
    """Plot confusion matrices for all models"""
    from path_utils import get_project_paths
    paths = get_project_paths()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, (name, result) in enumerate(results.items()):
        cm = confusion_matrix(result['y_test'], result['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                   xticklabels=['Active', 'Churned'], yticklabels=['Active', 'Churned'])
        axes[idx].set_title(f'{name}\nAccuracy: {result["accuracy"]:.3f}', fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(os.path.join(paths['charts'], 'churn_confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved churn_confusion_matrices.png")

def plot_roc_curves(results):
    """Plot ROC curves for all models"""
    from path_utils import get_project_paths
    paths = get_project_paths()
    
    plt.figure(figsize=(10, 8))
    
    for name, result in results.items():
        fpr, tpr, _ = roc_curve(result['y_test'], result['y_pred_proba'])
        plt.plot(fpr, tpr, label=f'{name} (AUC = {result["roc_auc"]:.3f})', linewidth=2)
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves - Churn Prediction Models', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(paths['charts'], 'churn_roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved churn_roc_curves.png")

def save_metrics(results):
    """Save model metrics to file"""
    from path_utils import get_project_paths
    paths = get_project_paths()
    
    with open(os.path.join(paths['outputs'], 'churn_model_metrics.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("CHURN PREDICTION MODEL METRICS\n")
        f.write("="*60 + "\n\n")
        
        # Find best model
        best_model = max(results.items(), key=lambda x: x[1]['f1'])
        
        f.write(f"Best Model (by F1-Score): {best_model[0]}\n")
        f.write(f"F1-Score: {best_model[1]['f1']:.4f}\n\n")
        
        f.write("-"*60 + "\n")
        f.write("DETAILED METRICS\n")
        f.write("-"*60 + "\n\n")
        
        for name, metrics in results.items():
            f.write(f"{name}:\n")
            f.write(f"  Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"  Precision: {metrics['precision']:.4f}\n")
            f.write(f"  Recall:    {metrics['recall']:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n\n")
    
    print("✓ Saved churn_model_metrics.txt")

def main():
    """Main churn prediction function"""
    import os
    from path_utils import get_project_paths
    
    paths = get_project_paths()
    print("Loading datasets...")
    users_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_users.csv'))
    watch_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_watch_history.csv'))
    
    print("\n" + "="*50)
    print("CHURN PREDICTION MODEL")
    print("="*50)
    
    print("\n1. Feature Engineering...")
    users_df = feature_engineering(users_df, watch_df)
    
    print("\n2. Preparing Features...")
    X, y, le_dict = prepare_features(users_df)
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]}")
    print(f"   Churn Rate: {y.mean():.2%}")
    
    print("\n3. Training Models...")
    results, trained_models, scaler, X_test, X_test_scaled = train_and_evaluate_models(X, y)
    
    print("\n4. Creating Visualizations...")
    plot_confusion_matrices(results)
    plot_roc_curves(results)
    
    print("\n5. Saving Metrics...")
    save_metrics(results)
    
    print("\n" + "="*50)
    print("Churn Prediction Complete!")
    print("="*50)
    
    return results, trained_models, scaler, le_dict

if __name__ == "__main__":
    main()


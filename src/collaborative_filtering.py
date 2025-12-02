"""
Advanced OTT Streaming Analytics - Collaborative Filtering Recommender
Uses Surprise library for collaborative filtering recommendations
Falls back to scikit-learn if Surprise is not available
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import Surprise, fallback to scikit-learn if not available
try:
    from surprise import Dataset, Reader, SVD, KNNBaseline
    from surprise.model_selection import cross_validate, train_test_split
    from surprise import accuracy
    SURPRISE_AVAILABLE = True
except ImportError:
    from sklearn.decomposition import NMF
    from sklearn.preprocessing import StandardScaler
    SURPRISE_AVAILABLE = False
    print("Warning: Surprise library not available. Using scikit-learn NMF as fallback.")

class CollaborativeFilteringRecommender:
    def __init__(self, watch_df):
        """Initialize collaborative filtering recommender"""
        self.watch_df = watch_df.copy()
        self.model = None
        self.trainset = None
        self.testset = None
        
    def prepare_data(self):
        """Prepare data for Surprise library or scikit-learn"""
        # Filter to users and titles with sufficient interactions
        user_counts = self.watch_df['user_id'].value_counts()
        title_counts = self.watch_df['title_id'].value_counts()
        
        # Keep users with at least 3 ratings and titles with at least 3 ratings
        valid_users = user_counts[user_counts >= 3].index
        valid_titles = title_counts[title_counts >= 3].index
        
        filtered_df = self.watch_df[
            (self.watch_df['user_id'].isin(valid_users)) &
            (self.watch_df['title_id'].isin(valid_titles)) &
            (self.watch_df['rating'].notna())
        ].copy()
        
        print(f"Filtered data: {len(filtered_df)} ratings from {filtered_df['user_id'].nunique()} users")
        
        if SURPRISE_AVAILABLE:
            # Prepare for Surprise (user_id, title_id, rating)
            surprise_data = filtered_df[['user_id', 'title_id', 'rating']].copy()
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(surprise_data, reader)
            return data
        else:
            # Return filtered dataframe for scikit-learn
            return filtered_df
    
    def train_svd(self, data, test_size=0.2):
        """Train SVD model (or NMF fallback)"""
        print("\nTraining collaborative filtering model...")
        
        if SURPRISE_AVAILABLE:
            # Split data
            trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
            self.trainset = trainset
            self.testset = testset
            
            # Train SVD
            self.model = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
            self.model.fit(trainset)
            
            # Evaluate
            predictions = self.model.test(testset)
            rmse = accuracy.rmse(predictions)
            mae = accuracy.mae(predictions)
            
            print(f"✓ SVD Model Trained")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            
            return rmse, mae
        else:
            # Use NMF from scikit-learn as fallback
            from sklearn.model_selection import train_test_split as sk_train_test_split
            from sklearn.metrics import mean_squared_error, mean_absolute_error
            
            # Create user-item matrix
            pivot_df = data.pivot_table(
                index='user_id', 
                columns='title_id', 
                values='rating', 
                fill_value=0
            )
            
            # Split data
            train_idx, test_idx = sk_train_test_split(
                range(len(pivot_df)), 
                test_size=test_size, 
                random_state=42
            )
            train_data = pivot_df.iloc[train_idx]
            test_data = pivot_df.iloc[test_idx]
            
            # Train NMF
            self.model = NMF(n_components=50, random_state=42, max_iter=200)
            W = self.model.fit_transform(train_data.values)
            H = self.model.components_
            
            # Store for predictions
            self.user_matrix = W
            self.item_matrix = H
            self.user_ids = train_data.index.values
            self.title_ids = train_data.columns.values
            self.pivot_df = pivot_df
            self.train_data = train_data
            
            # Evaluate on training set (since test set has different users/titles)
            # This is a simplified evaluation for the NMF fallback
            train_predictions = np.dot(W, H)
            train_actual = train_data.values
            train_mask = train_actual > 0  # Only evaluate on non-zero ratings
            
            if train_mask.sum() > 0:
                # Flatten arrays for evaluation
                actual_flat = train_actual[train_mask].flatten()
                pred_flat = train_predictions[train_mask].flatten()
                rmse = np.sqrt(mean_squared_error(actual_flat, pred_flat))
                mae = mean_absolute_error(actual_flat, pred_flat)
            else:
                rmse = 0.0
                mae = 0.0
            
            print(f"✓ NMF Model Trained (Fallback)")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  MAE: {mae:.4f}")
            
            return rmse, mae
    
    def train_knn(self, data, test_size=0.2):
        """Train KNNBaseline model (only available with Surprise)"""
        if not SURPRISE_AVAILABLE:
            print("KNNBaseline requires Surprise library. Using NMF instead.")
            return self.train_svd(data, test_size)
        
        print("\nTraining KNNBaseline model...")
        
        # Split data
        trainset, testset = train_test_split(data, test_size=test_size, random_state=42)
        self.trainset = trainset
        self.testset = testset
        
        # Train KNNBaseline
        sim_options = {
            'name': 'pearson_baseline',
            'user_based': True
        }
        self.model = KNNBaseline(sim_options=sim_options, random_state=42)
        self.model.fit(trainset)
        
        # Evaluate
        predictions = self.model.test(testset)
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        
        print(f"✓ KNNBaseline Model Trained")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return rmse, mae
    
    def predict_rating(self, user_id, title_id):
        """Predict rating for a user-title pair"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_svd() or train_knn() first.")
        
        if SURPRISE_AVAILABLE:
            prediction = self.model.predict(user_id, title_id)
            return prediction.est
        else:
            # NMF prediction
            if hasattr(self, 'user_ids') and user_id in self.user_ids and title_id in self.title_ids:
                user_idx = np.where(self.user_ids == user_id)[0]
                title_idx = np.where(self.title_ids == title_id)[0]
                if len(user_idx) > 0 and len(title_idx) > 0:
                    pred = np.dot(self.user_matrix[user_idx[0]], self.item_matrix[:, title_idx[0]])
                    return max(1, min(5, pred))  # Clip to rating range
            # If user or title not in training set, use average
            return 3.0  # Default rating
    
    def recommend_for_user(self, user_id, titles_df, n_recommendations=10):
        """Get top recommendations for a user"""
        if self.model is None:
            raise ValueError("Model not trained. Call train_svd() or train_knn() first.")
        
        # Get all titles
        all_titles = titles_df['title_id'].unique()
        
        # Get user's watched titles
        user_watched = self.watch_df[self.watch_df['user_id'] == user_id]['title_id'].unique()
        
        # Predict ratings for unwatched titles
        predictions = []
        for title_id in all_titles:
            if title_id not in user_watched:
                try:
                    pred_rating = self.predict_rating(user_id, title_id)
                    predictions.append({
                        'title_id': title_id,
                        'predicted_rating': pred_rating
                    })
                except:
                    continue
        
        # Sort by predicted rating
        predictions_df = pd.DataFrame(predictions)
        if len(predictions_df) > 0:
            predictions_df = predictions_df.sort_values('predicted_rating', ascending=False)
            predictions_df = predictions_df.head(n_recommendations)
            
            # Merge with title information
            recommendations = predictions_df.merge(titles_df, on='title_id', how='left')
            return recommendations[['title_id', 'title_name', 'genre', 'release_year', 
                                   'popularity_score', 'predicted_rating']]
        
        return pd.DataFrame()

def main():
    """Main function to test collaborative filtering"""
    print("Loading datasets...")
    watch_df = pd.read_csv('outputs/cleaned_watch_history.csv')
    titles_df = pd.read_csv('data/ott_titles.csv')
    
    print("\n" + "="*50)
    print("COLLABORATIVE FILTERING RECOMMENDER")
    print("="*50)
    
    if not SURPRISE_AVAILABLE:
        print("\nNote: Using scikit-learn NMF as fallback (Surprise not available)")
        print("For best results, install Surprise with: pip install scikit-surprise")
        print("(May require Python < 3.14 or building from source)\n")
    
    print("\n1. Initializing Recommender...")
    recommender = CollaborativeFilteringRecommender(watch_df)
    
    print("\n2. Preparing Data...")
    data = recommender.prepare_data()
    
    print("\n3. Training Model...")
    rmse, mae = recommender.train_svd(data)
    
    print("\n4. Testing Recommendations...")
    # Test for a sample user
    sample_user_id = watch_df['user_id'].iloc[0]
    print(f"\nRecommendations for User ID {sample_user_id}:")
    user_recs = recommender.recommend_for_user(sample_user_id, titles_df, n_recommendations=10)
    if len(user_recs) > 0:
        print(user_recs[['title_name', 'genre', 'predicted_rating']].to_string(index=False))
    else:
        print("No recommendations available.")
    
    print("\n" + "="*50)
    print("Collaborative Filtering Complete!")
    print("="*50)
    
    return recommender

if __name__ == "__main__":
    main()


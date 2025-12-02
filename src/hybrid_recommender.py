"""
Advanced OTT Streaming Analytics - Hybrid Recommender Engine
Combines content-based and collaborative filtering approaches
"""

import pandas as pd
import numpy as np
from content_based_recommender import ContentBasedRecommender
from collaborative_filtering import CollaborativeFilteringRecommender
import warnings
warnings.filterwarnings('ignore')

class HybridRecommender:
    def __init__(self, content_recommender, collab_recommender, titles_df, watch_df):
        """Initialize hybrid recommender"""
        self.content_recommender = content_recommender
        self.collab_recommender = collab_recommender
        self.titles_df = titles_df
        self.watch_df = watch_df
        
        # Default weights
        self.content_weight = 0.4
        self.collab_weight = 0.6
    
    def set_weights(self, content_weight, collab_weight):
        """Set weights for hybrid combination"""
        if abs(content_weight + collab_weight - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
        self.content_weight = content_weight
        self.collab_weight = collab_weight
    
    def recommend_for_user(self, user_id, n_recommendations=10):
        """Get hybrid recommendations for a user"""
        # Get content-based recommendations
        content_recs = self.content_recommender.recommend_for_user(
            user_id, self.watch_df, n_recommendations=n_recommendations * 2
        )
        
        # Get collaborative filtering recommendations
        collab_recs = self.collab_recommender.recommend_for_user(
            user_id, self.titles_df, n_recommendations=n_recommendations * 2
        )
        
        # Combine recommendations
        all_recommendations = {}
        
        # Add content-based recommendations
        if len(content_recs) > 0:
            for _, rec in content_recs.iterrows():
                tid = rec['title_id']
                all_recommendations[tid] = {
                    'title_id': tid,
                    'title_name': rec['title_name'],
                    'genre': rec['genre'],
                    'release_year': rec['release_year'],
                    'popularity_score': rec['popularity_score'],
                    'content_score': rec.get('similarity_score', 0),
                    'collab_score': 0,
                    'hybrid_score': 0
                }
        
        # Add collaborative filtering recommendations
        if len(collab_recs) > 0:
            for _, rec in collab_recs.iterrows():
                tid = rec['title_id']
                if tid in all_recommendations:
                    all_recommendations[tid]['collab_score'] = rec.get('predicted_rating', 0)
                else:
                    all_recommendations[tid] = {
                        'title_id': tid,
                        'title_name': rec['title_name'],
                        'genre': rec['genre'],
                        'release_year': rec['release_year'],
                        'popularity_score': rec['popularity_score'],
                        'content_score': 0,
                        'collab_score': rec.get('predicted_rating', 0),
                        'hybrid_score': 0
                    }
        
        # Normalize scores
        if len(all_recommendations) > 0:
            content_scores = [r['content_score'] for r in all_recommendations.values()]
            collab_scores = [r['collab_score'] for r in all_recommendations.values()]
            
            max_content = max(content_scores) if max(content_scores) > 0 else 1
            max_collab = max(collab_scores) if max(collab_scores) > 0 else 1
            
            # Calculate hybrid scores
            for tid in all_recommendations:
                norm_content = all_recommendations[tid]['content_score'] / max_content
                norm_collab = all_recommendations[tid]['collab_score'] / max_collab
                all_recommendations[tid]['hybrid_score'] = (
                    self.content_weight * norm_content + 
                    self.collab_weight * norm_collab
                )
        
        # Convert to DataFrame and sort
        recommendations_df = pd.DataFrame(list(all_recommendations.values()))
        if len(recommendations_df) > 0:
            recommendations_df = recommendations_df.sort_values('hybrid_score', ascending=False)
            recommendations_df = recommendations_df.head(n_recommendations)
        
        return recommendations_df

def main():
    """Main function to test hybrid recommender"""
    import os
    from path_utils import get_project_paths
    
    paths = get_project_paths()
    print("Loading datasets...")
    titles_df = pd.read_csv(os.path.join(paths['data'], 'ott_titles.csv'))
    watch_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_watch_history.csv'))
    
    print("\n" + "="*50)
    print("HYBRID RECOMMENDER ENGINE")
    print("="*50)
    
    print("\n1. Initializing Content-Based Recommender...")
    content_recommender = ContentBasedRecommender(titles_df)
    content_recommender.fit()
    
    print("\n2. Initializing Collaborative Filtering Recommender...")
    collab_recommender = CollaborativeFilteringRecommender(watch_df)
    data = collab_recommender.prepare_data()
    collab_recommender.train_svd(data)
    
    print("\n3. Creating Hybrid Recommender...")
    hybrid_recommender = HybridRecommender(
        content_recommender, collab_recommender, titles_df, watch_df
    )
    hybrid_recommender.set_weights(content_weight=0.4, collab_weight=0.6)
    
    print("\n4. Testing Hybrid Recommendations...")
    # Test for sample users
    sample_users = watch_df['user_id'].unique()[:5]
    
    all_recommendations = []
    for user_id in sample_users:
        recs = hybrid_recommender.recommend_for_user(user_id, n_recommendations=10)
        if len(recs) > 0:
            recs['user_id'] = user_id
            all_recommendations.append(recs)
            print(f"\nUser {user_id} - Top 3 Recommendations:")
            print(recs[['title_name', 'genre', 'hybrid_score']].head(3).to_string(index=False))
    
    # Save sample recommendations
    if all_recommendations:
        final_recs = pd.concat(all_recommendations, ignore_index=True)
        final_recs.to_csv(os.path.join(paths['outputs'], 'sample_recommendations.csv'), index=False)
        print(f"\n✓ Saved {len(final_recs)} recommendations to sample_recommendations.csv")
    
    # Save hybrid engine results
    with open(os.path.join(paths['outputs'], 'hybrid_engine_results.txt'), 'w') as f:
        f.write("="*60 + "\n")
        f.write("HYBRID RECOMMENDATION ENGINE RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write(f"Content-Based Weight: {hybrid_recommender.content_weight}\n")
        f.write(f"Collaborative Filtering Weight: {hybrid_recommender.collab_weight}\n\n")
        f.write(f"Total Recommendations Generated: {len(final_recs) if all_recommendations else 0}\n")
        f.write(f"Average Hybrid Score: {final_recs['hybrid_score'].mean():.4f}\n" if all_recommendations else "")
    
    print("\n" + "="*50)
    print("Hybrid Recommender Complete!")
    print("="*50)
    
    return hybrid_recommender

if __name__ == "__main__":
    main()


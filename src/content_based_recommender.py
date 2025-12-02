"""
Advanced OTT Streaming Analytics - Content-Based Recommender
Uses TF-IDF and cosine similarity for content-based recommendations
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class ContentBasedRecommender:
    def __init__(self, titles_df):
        """Initialize the content-based recommender"""
        self.titles_df = titles_df.copy()
        self.vectorizer = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        
    def prepare_content(self):
        """Prepare content features for TF-IDF"""
        # Combine genre and description
        self.titles_df['content'] = (
            self.titles_df['genre'] + ' ' + 
            self.titles_df['description'].fillna('')
        )
        return self.titles_df['content']
    
    def fit(self):
        """Fit the TF-IDF vectorizer and compute similarity matrix"""
        print("Fitting TF-IDF vectorizer...")
        content = self.prepare_content()
        
        # Initialize and fit TF-IDF
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )
        
        self.tfidf_matrix = self.vectorizer.fit_transform(content)
        
        # Compute cosine similarity matrix
        print("Computing cosine similarity matrix...")
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        
        print(f"✓ Similarity matrix shape: {self.similarity_matrix.shape}")
        return self
    
    def recommend(self, title_id, n_recommendations=10):
        """Get recommendations for a given title"""
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get index of the title
        title_idx = self.titles_df[self.titles_df['title_id'] == title_id].index
        if len(title_idx) == 0:
            return pd.DataFrame()
        
        title_idx = title_idx[0]
        
        # Get similarity scores
        similarity_scores = list(enumerate(self.similarity_matrix[title_idx]))
        
        # Sort by similarity (excluding the title itself)
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)[1:n_recommendations+1]
        
        # Get recommended titles
        recommended_indices = [i[0] for i in similarity_scores]
        recommended_scores = [i[1] for i in similarity_scores]
        
        recommendations = self.titles_df.iloc[recommended_indices].copy()
        recommendations['similarity_score'] = recommended_scores
        
        return recommendations[['title_id', 'title_name', 'genre', 'release_year', 
                               'popularity_score', 'similarity_score']]
    
    def recommend_for_user(self, user_id, watch_df, n_recommendations=10):
        """Get recommendations for a user based on their watch history"""
        if self.similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Get user's watched titles
        user_watched = watch_df[watch_df['user_id'] == user_id]['title_id'].unique()
        
        if len(user_watched) == 0:
            return pd.DataFrame()
        
        # Get all recommendations for user's watched titles
        all_recommendations = {}
        
        for title_id in user_watched[:10]:  # Limit to top 10 watched titles
            recs = self.recommend(title_id, n_recommendations=20)
            for _, rec in recs.iterrows():
                tid = rec['title_id']
                if tid not in user_watched:  # Exclude already watched
                    if tid not in all_recommendations:
                        all_recommendations[tid] = {
                            'title_id': tid,
                            'title_name': rec['title_name'],
                            'genre': rec['genre'],
                            'release_year': rec['release_year'],
                            'popularity_score': rec['popularity_score'],
                            'similarity_score': rec['similarity_score']
                        }
                    else:
                        # Average similarity if title appears multiple times
                        all_recommendations[tid]['similarity_score'] = (
                            all_recommendations[tid]['similarity_score'] + rec['similarity_score']
                        ) / 2
        
        # Convert to DataFrame and sort by similarity
        recommendations_df = pd.DataFrame(list(all_recommendations.values()))
        if len(recommendations_df) > 0:
            recommendations_df = recommendations_df.sort_values('similarity_score', ascending=False)
            recommendations_df = recommendations_df.head(n_recommendations)
        
        return recommendations_df

def main():
    """Main function to test content-based recommender"""
    print("Loading datasets...")
    titles_df = pd.read_csv('data/ott_titles.csv')
    watch_df = pd.read_csv('outputs/cleaned_watch_history.csv')
    
    print("\n" + "="*50)
    print("CONTENT-BASED RECOMMENDER")
    print("="*50)
    
    print("\n1. Initializing Recommender...")
    recommender = ContentBasedRecommender(titles_df)
    
    print("\n2. Fitting Model...")
    recommender.fit()
    
    print("\n3. Testing Recommendations...")
    # Test with a sample title
    sample_title_id = titles_df['title_id'].iloc[0]
    print(f"\nRecommendations for Title ID {sample_title_id}:")
    recs = recommender.recommend(sample_title_id, n_recommendations=5)
    print(recs[['title_name', 'genre', 'similarity_score']].to_string(index=False))
    
    # Test for a sample user
    sample_user_id = watch_df['user_id'].iloc[0]
    print(f"\nRecommendations for User ID {sample_user_id}:")
    user_recs = recommender.recommend_for_user(sample_user_id, watch_df, n_recommendations=10)
    if len(user_recs) > 0:
        print(user_recs[['title_name', 'genre', 'similarity_score']].to_string(index=False))
    else:
        print("No recommendations available.")
    
    print("\n" + "="*50)
    print("Content-Based Recommender Complete!")
    print("="*50)
    
    return recommender

if __name__ == "__main__":
    main()


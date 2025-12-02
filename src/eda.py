"""
Advanced OTT Streaming Analytics - Exploratory Data Analysis
Performs comprehensive EDA including clustering and LTV estimation
"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
import os
warnings.filterwarnings('ignore')

def get_project_paths():
    """Get project root and common paths"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return {
        'root': project_root,
        'data': os.path.join(project_root, 'data'),
        'outputs': os.path.join(project_root, 'outputs'),
        'charts': os.path.join(project_root, 'outputs', 'charts')
    }

def genre_engagement_analysis(watch_df, titles_df):
    """Analyze genre-level engagement metrics"""
    merged = watch_df.merge(titles_df, on='title_id', how='left')
    
    genre_stats = merged.groupby('genre').agg({
        'watch_time_minutes': ['sum', 'mean', 'count'],
        'rating': 'mean',
        'user_id': 'nunique'
    }).round(2)
    
    genre_stats.columns = ['total_watch_time', 'avg_watch_time', 'watch_count', 
                          'avg_rating', 'unique_users']
    genre_stats = genre_stats.sort_values('total_watch_time', ascending=False)
    
    return genre_stats

def session_pattern_clustering(users_df):
    """Cluster users based on session patterns using KMeans"""
    # Prepare features for clustering
    features = ['avg_watch_time', 'sessions_per_week', 'age']
    X = users_df[features].copy()
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply KMeans clustering
    n_clusters = 4
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    users_df['cluster'] = kmeans.fit_predict(X_scaled)
    
    # Analyze clusters
    cluster_summary = users_df.groupby('cluster')[features + ['churn']].agg({
        'avg_watch_time': 'mean',
        'sessions_per_week': 'mean',
        'age': 'mean',
        'churn': 'mean'
    }).round(2)
    
    return users_df, cluster_summary, kmeans, scaler

def country_wise_analysis(users_df, watch_df):
    """Analyze watch behavior by country"""
    country_stats = users_df.groupby('country').agg({
        'user_id': 'count',
        'churn': 'mean',
        'avg_watch_time': 'mean',
        'sessions_per_week': 'mean'
    }).round(2)
    
    country_stats.columns = ['total_users', 'churn_rate', 'avg_watch_time', 
                            'avg_sessions_per_week']
    country_stats = country_stats.sort_values('total_users', ascending=False)
    
    # Merge with watch data for deeper insights
    merged = watch_df.merge(users_df[['user_id', 'country']], on='user_id', how='left')
    country_watch = merged.groupby('country').agg({
        'watch_time_minutes': 'sum',
        'rating': 'mean',
        'user_id': 'nunique'
    }).round(2)
    
    country_watch.columns = ['total_watch_time', 'avg_rating', 'active_users']
    
    return country_stats, country_watch

def time_based_engagement(watch_df):
    """Analyze engagement trends over time"""
    watch_df['watch_date'] = pd.to_datetime(watch_df['watch_date'])
    watch_df['day_of_week'] = watch_df['watch_date'].dt.day_name()
    watch_df['hour'] = watch_df['watch_date'].dt.hour
    watch_df['month'] = watch_df['watch_date'].dt.month
    
    # Daily patterns
    daily_patterns = watch_df.groupby('day_of_week').agg({
        'watch_time_minutes': 'sum',
        'user_id': 'nunique'
    }).round(2)
    
    # Hourly patterns
    hourly_patterns = watch_df.groupby('hour').agg({
        'watch_time_minutes': 'sum',
        'user_id': 'nunique'
    }).round(2)
    
    # Monthly trends
    monthly_trends = watch_df.groupby('month').agg({
        'watch_time_minutes': 'sum',
        'user_id': 'nunique'
    }).round(2)
    
    return daily_patterns, hourly_patterns, monthly_trends

def completion_analysis(watch_df, titles_df):
    """Analyze show completion rates"""
    # Estimate completion based on watch time
    # Assume average show length is 45 minutes
    avg_show_length = 45
    
    merged = watch_df.merge(titles_df[['title_id', 'genre']], on='title_id', how='left')
    merged['completion_rate'] = (merged['watch_time_minutes'] / avg_show_length).clip(0, 1)
    
    # Overall completion
    overall_completion = merged['completion_rate'].mean()
    
    # By genre
    genre_completion = merged.groupby('genre')['completion_rate'].mean().sort_values(ascending=False)
    
    # Series funnel (episode completion)
    # Simulate series by grouping titles
    merged['series_id'] = (merged['title_id'] // 10)  # Group every 10 titles as a series
    series_completion = merged.groupby('series_id')['completion_rate'].agg(['mean', 'count'])
    series_completion = series_completion[series_completion['count'] >= 5]  # Series with at least 5 episodes
    
    return overall_completion, genre_completion, series_completion

def calculate_ltv(users_df, watch_df):
    """Estimate User Lifetime Value"""
    # Calculate revenue per user (simplified model)
    plan_prices = {'basic': 9.99, 'standard': 14.99, 'premium': 19.99}
    users_df['monthly_revenue'] = users_df['subscription_plan'].map(plan_prices)
    
    # Calculate months active
    users_df['join_date'] = pd.to_datetime(users_df['join_date'])
    users_df['last_active_date'] = pd.to_datetime(users_df['last_active_date'])
    users_df['months_active'] = ((users_df['last_active_date'] - users_df['join_date']).dt.days / 30).clip(1)
    
    # Calculate LTV
    users_df['ltv'] = users_df['monthly_revenue'] * users_df['months_active']
    
    # Add engagement factor
    watch_per_user = watch_df.groupby('user_id')['watch_time_minutes'].sum()
    users_df = users_df.merge(watch_per_user.reset_index(), on='user_id', how='left')
    users_df['total_watch_time'] = users_df['watch_time_minutes'].fillna(0)
    
    # Engagement multiplier (users who watch more are more valuable)
    users_df['engagement_multiplier'] = 1 + (users_df['total_watch_time'] / users_df['total_watch_time'].quantile(0.75))
    users_df['ltv_adjusted'] = users_df['ltv'] * users_df['engagement_multiplier']
    
    ltv_summary = users_df.groupby('subscription_plan').agg({
        'ltv': 'mean',
        'ltv_adjusted': 'mean',
        'user_id': 'count'
    }).round(2)
    
    return users_df, ltv_summary

def main():
    """Main EDA function"""
    paths = get_project_paths()
    print("Loading cleaned datasets...")
    users_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_users.csv'))
    watch_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_watch_history.csv'))
    titles_df = pd.read_csv(os.path.join(paths['data'], 'ott_titles.csv'))
    
    print("\n" + "="*50)
    print("EXPLORATORY DATA ANALYSIS")
    print("="*50)
    
    print("\n1. Genre Engagement Analysis...")
    genre_stats = genre_engagement_analysis(watch_df, titles_df)
    print(genre_stats.head(10))
    
    print("\n2. Session Pattern Clustering...")
    users_df, cluster_summary, kmeans, scaler = session_pattern_clustering(users_df)
    print(cluster_summary)
    paths = get_project_paths()
    users_df.to_csv(os.path.join(paths['outputs'], 'cleaned_users.csv'), index=False)  # Save with clusters
    
    print("\n3. Country-wise Analysis...")
    country_stats, country_watch = country_wise_analysis(users_df, watch_df)
    print(country_stats.head(10))
    
    print("\n4. Time-based Engagement...")
    daily, hourly, monthly = time_based_engagement(watch_df)
    print("\nDaily Patterns:")
    print(daily)
    print("\nPeak Hours:")
    print(hourly.sort_values('watch_time_minutes', ascending=False).head(5))
    
    print("\n5. Completion Analysis...")
    overall_comp, genre_comp, series_comp = completion_analysis(watch_df, titles_df)
    print(f"\nOverall Completion Rate: {overall_comp:.2%}")
    print("\nCompletion by Genre:")
    print(genre_comp)
    
    print("\n6. User Lifetime Value...")
    users_df, ltv_summary = calculate_ltv(users_df, watch_df)
    print(ltv_summary)
    paths = get_project_paths()
    users_df.to_csv(os.path.join(paths['outputs'], 'cleaned_users.csv'), index=False)  # Save with LTV
    
    print("\n" + "="*50)
    print("EDA Complete!")
    print("="*50)
    
    return users_df, watch_df, titles_df

if __name__ == "__main__":
    main()


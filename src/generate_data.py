"""
Advanced OTT Streaming Analytics - Data Generation Script
Generates realistic synthetic datasets for OTT platform analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Configuration
N_USERS = 10000
N_TITLES = 2000
N_WATCH_RECORDS = 150000
START_DATE = datetime(2020, 1, 1)
END_DATE = datetime(2024, 12, 31)

# Genre and country lists
GENRES = ['Action', 'Comedy', 'Drama', 'Thriller', 'Romance', 'Horror', 
          'Sci-Fi', 'Documentary', 'Animation', 'Crime', 'Fantasy', 'Mystery']
COUNTRIES = ['USA', 'UK', 'Canada', 'Australia', 'India', 'Germany', 
             'France', 'Japan', 'Brazil', 'Mexico', 'Spain', 'Italy']
SUBSCRIPTION_PLANS = ['basic', 'standard', 'premium']
DEVICES = ['tv', 'tablet', 'mobile', 'laptop']
AGE_RATINGS = ['G', 'PG', 'PG-13', 'R', 'NC-17']

def generate_ott_users(n_users=N_USERS):
    """Generate realistic user dataset"""
    users = []
    
    for i in range(1, n_users + 1):
        age = int(np.random.normal(35, 12))
        age = max(18, min(80, age))  # Clamp between 18-80
        
        gender = np.random.choice(['M', 'F', 'Other'], p=[0.48, 0.48, 0.04])
        country = np.random.choice(COUNTRIES)
        subscription_plan = np.random.choice(SUBSCRIPTION_PLANS, 
                                            p=[0.3, 0.45, 0.25])
        device_type = np.random.choice(DEVICES, 
                                      p=[0.35, 0.20, 0.30, 0.15])
        preferred_genre = np.random.choice(GENRES)
        
        # Join date (earlier users more likely to churn)
        days_since_start = np.random.exponential(scale=400)
        join_date = START_DATE + timedelta(days=int(days_since_start))
        join_date = min(join_date, END_DATE - timedelta(days=30))
        
        # Last active date (some users churned)
        churn_prob = 0.15 if subscription_plan == 'basic' else 0.10
        churn_prob = 0.20 if days_since_start > 500 else churn_prob
        
        if np.random.random() < churn_prob:
            churn = 1
            # Churned users: last active 30-180 days ago
            days_inactive = np.random.randint(30, 180)
            last_active_date = END_DATE - timedelta(days=days_inactive)
        else:
            churn = 0
            # Active users: last active 0-7 days ago
            days_inactive = np.random.randint(0, 7)
            last_active_date = END_DATE - timedelta(days=days_inactive)
        
        # Watch behavior correlated with subscription plan
        if subscription_plan == 'premium':
            avg_watch_time = np.random.normal(120, 30)
            sessions_per_week = np.random.poisson(8)
        elif subscription_plan == 'standard':
            avg_watch_time = np.random.normal(90, 25)
            sessions_per_week = np.random.poisson(5)
        else:
            avg_watch_time = np.random.normal(60, 20)
            sessions_per_week = np.random.poisson(3)
        
        avg_watch_time = max(10, avg_watch_time)
        sessions_per_week = max(1, sessions_per_week)
        
        # Rating behavior (some users rate more, some less)
        rating_behavior = np.random.choice(['frequent', 'occasional', 'rare'], 
                                         p=[0.3, 0.5, 0.2])
        
        users.append({
            'user_id': i,
            'age': age,
            'gender': gender,
            'country': country,
            'subscription_plan': subscription_plan,
            'device_type': device_type,
            'churn': churn,
            'avg_watch_time': round(avg_watch_time, 1),
            'sessions_per_week': sessions_per_week,
            'preferred_genre': preferred_genre,
            'rating_behavior': rating_behavior,
            'join_date': join_date.strftime('%Y-%m-%d'),
            'last_active_date': last_active_date.strftime('%Y-%m-%d')
        })
    
    return pd.DataFrame(users)

def generate_ott_titles(n_titles=N_TITLES):
    """Generate title catalog dataset"""
    titles = []
    
    # Popular titles (top 20%)
    popular_titles = int(n_titles * 0.2)
    
    for i in range(1, n_titles + 1):
        genre = np.random.choice(GENRES)
        
        # Popularity score (some titles are more popular)
        if i <= popular_titles:
            popularity_score = np.random.uniform(0.7, 1.0)
        else:
            popularity_score = np.random.uniform(0.1, 0.7)
        
        release_year = np.random.randint(1990, 2024)
        age_rating = np.random.choice(AGE_RATINGS, p=[0.1, 0.2, 0.4, 0.25, 0.05])
        
        # Generate realistic title names
        title_name = f"{genre} {np.random.choice(['Chronicles', 'Tales', 'Legacy', 'Secrets', 'Journey', 'Quest'])} {i}"
        
        # Generate description
        description = f"An engaging {genre.lower()} story that captivates audiences. Released in {release_year}, this title has received critical acclaim."
        
        titles.append({
            'title_id': i,
            'title_name': title_name,
            'genre': genre,
            'release_year': release_year,
            'age_rating': age_rating,
            'popularity_score': round(popularity_score, 3),
            'description': description
        })
    
    return pd.DataFrame(titles)

def generate_ott_watch_history(n_records=N_WATCH_RECORDS, users_df=None, titles_df=None):
    """Generate watch history dataset"""
    watch_records = []
    
    # Create user-title preference matrix (some users prefer certain genres)
    user_genre_prefs = {}
    for _, user in users_df.iterrows():
        user_genre_prefs[user['user_id']] = user['preferred_genre']
    
    # Create title genre mapping
    title_genres = {}
    for _, title in titles_df.iterrows():
        title_genres[title['title_id']] = title['genre']
    
    # Generate watch records
    for _ in range(n_records):
        user_id = np.random.randint(1, len(users_df) + 1)
        title_id = np.random.randint(1, len(titles_df) + 1)
        
        user = users_df[users_df['user_id'] == user_id].iloc[0]
        title = titles_df[titles_df['title_id'] == title_id].iloc[0]
        
        # Watch time influenced by user's avg_watch_time and title popularity
        base_watch_time = user['avg_watch_time']
        popularity_factor = title['popularity_score']
        watch_time = base_watch_time * (0.7 + 0.6 * popularity_factor)
        watch_time = max(5, min(180, watch_time + np.random.normal(0, 15)))
        
        # Rating probability based on user's rating behavior
        rating_prob = {'frequent': 0.7, 'occasional': 0.4, 'rare': 0.1}
        prob = rating_prob[user['rating_behavior']]
        
        if np.random.random() < prob:
            # Ratings tend to be higher for preferred genres
            if title_genres[title_id] == user_genre_prefs[user_id]:
                rating = np.random.choice([3, 4, 5], p=[0.2, 0.3, 0.5])
            else:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.1, 0.2, 0.3, 0.3, 0.1])
        else:
            rating = np.nan
        
        # Watch date (more recent for active users)
        if user['churn'] == 0:
            days_ago = np.random.exponential(scale=30)
        else:
            days_ago = np.random.exponential(scale=200)
        
        days_ago = min(days_ago, (END_DATE - datetime.strptime(user['join_date'], '%Y-%m-%d')).days)
        watch_date = END_DATE - timedelta(days=int(days_ago))
        watch_date = max(watch_date, datetime.strptime(user['join_date'], '%Y-%m-%d'))
        
        watch_records.append({
            'user_id': user_id,
            'title_id': title_id,
            'watch_time_minutes': round(watch_time, 1),
            'rating': int(rating) if not pd.isna(rating) else np.nan,
            'watch_date': watch_date.strftime('%Y-%m-%d')
        })
    
    df = pd.DataFrame(watch_records)
    # Remove some ratings to simulate missing data
    df.loc[df.sample(frac=0.15).index, 'rating'] = np.nan
    return df

def main():
    """Main function to generate all datasets"""
    # Get project root directory (parent of src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create directories relative to project root
    data_dir = os.path.join(project_root, 'data')
    outputs_dir = os.path.join(project_root, 'outputs')
    charts_dir = os.path.join(project_root, 'outputs', 'charts')
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(charts_dir, exist_ok=True)
    
    print("Generating OTT Users dataset...")
    users_df = generate_ott_users()
    users_df.to_csv(os.path.join(data_dir, 'ott_users.csv'), index=False)
    print(f"✓ Generated {len(users_df)} users")
    
    print("\nGenerating OTT Titles dataset...")
    titles_df = generate_ott_titles()
    titles_df.to_csv(os.path.join(data_dir, 'ott_titles.csv'), index=False)
    print(f"✓ Generated {len(titles_df)} titles")
    
    print("\nGenerating OTT Watch History dataset...")
    watch_df = generate_ott_watch_history(users_df=users_df, titles_df=titles_df)
    watch_df.to_csv(os.path.join(data_dir, 'ott_watch_history.csv'), index=False)
    print(f"✓ Generated {len(watch_df)} watch records")
    
    print("\n" + "="*50)
    print("Dataset Generation Complete!")
    print("="*50)
    print(f"\nUsers: {len(users_df)}")
    print(f"Titles: {len(titles_df)}")
    print(f"Watch Records: {len(watch_df)}")
    print(f"\nChurn Rate: {users_df['churn'].mean():.2%}")
    print(f"Average Watch Time: {users_df['avg_watch_time'].mean():.1f} minutes")

if __name__ == "__main__":
    main()


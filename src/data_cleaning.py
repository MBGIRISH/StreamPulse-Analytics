"""
Advanced OTT Streaming Analytics - Data Cleaning Script
Cleans and preprocesses the OTT datasets
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os

def clean_users_data(df):
    """Clean and preprocess users dataset"""
    df = df.copy()
    
    # Convert dates to datetime
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['last_active_date'] = pd.to_datetime(df['last_active_date'])
    
    # Fix categorical inconsistencies
    df['subscription_plan'] = df['subscription_plan'].str.lower().str.strip()
    df['device_type'] = df['device_type'].str.lower().str.strip()
    df['gender'] = df['gender'].str.upper().str.strip()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['user_id'], keep='first')
    
    # Handle missing values
    if df['age'].isna().any():
        df['age'] = df['age'].fillna(df['age'].median())
    
    # Ensure churn is binary
    df['churn'] = df['churn'].astype(int)
    
    # Validate ranges
    df['age'] = df['age'].clip(lower=18, upper=100)
    df['avg_watch_time'] = df['avg_watch_time'].clip(lower=0)
    df['sessions_per_week'] = df['sessions_per_week'].clip(lower=0)
    
    return df

def clean_watch_history_data(df):
    """Clean and preprocess watch history dataset"""
    df = df.copy()
    
    # Convert date to datetime
    df['watch_date'] = pd.to_datetime(df['watch_date'])
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['user_id', 'title_id', 'watch_date'], keep='first')
    
    # Handle missing values in watch_time
    df['watch_time_minutes'] = df['watch_time_minutes'].fillna(df['watch_time_minutes'].median())
    
    # Validate ranges
    df['watch_time_minutes'] = df['watch_time_minutes'].clip(lower=0, upper=500)
    df['rating'] = df['rating'].clip(lower=1, upper=5) if df['rating'].notna().any() else df['rating']
    
    # Remove invalid dates
    df = df[df['watch_date'] <= pd.Timestamp.now()]
    
    return df

def clean_titles_data(df):
    """Clean and preprocess titles dataset"""
    df = df.copy()
    
    # Remove duplicates
    df = df.drop_duplicates(subset=['title_id'], keep='first')
    
    # Standardize genre names
    df['genre'] = df['genre'].str.strip()
    
    # Handle missing values
    df['popularity_score'] = df['popularity_score'].fillna(df['popularity_score'].median())
    df['release_year'] = df['release_year'].fillna(df['release_year'].median())
    
    # Validate ranges
    df['popularity_score'] = df['popularity_score'].clip(lower=0, upper=1)
    df['release_year'] = df['release_year'].clip(lower=1900, upper=2024)
    
    return df

def merge_datasets(users_df, watch_df, titles_df):
    """Merge datasets for deeper insights"""
    # Merge watch history with titles
    watch_titles = watch_df.merge(titles_df, on='title_id', how='left')
    
    # Merge with users
    full_df = watch_titles.merge(users_df, on='user_id', how='left')
    
    return full_df

def main():
    """Main cleaning function"""
    # Get project root directory (parent of src)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Create outputs directory if it doesn't exist
    outputs_dir = os.path.join(project_root, 'outputs')
    data_dir = os.path.join(project_root, 'data')
    os.makedirs(outputs_dir, exist_ok=True)
    
    print("Loading datasets...")
    users_df = pd.read_csv(os.path.join(data_dir, 'ott_users.csv'))
    watch_df = pd.read_csv(os.path.join(data_dir, 'ott_watch_history.csv'))
    titles_df = pd.read_csv(os.path.join(data_dir, 'ott_titles.csv'))
    
    print("\nCleaning users data...")
    users_clean = clean_users_data(users_df)
    users_clean.to_csv(os.path.join(outputs_dir, 'cleaned_users.csv'), index=False)
    print(f"✓ Cleaned {len(users_clean)} users")
    
    print("\nCleaning watch history data...")
    watch_clean = clean_watch_history_data(watch_df)
    watch_clean.to_csv(os.path.join(outputs_dir, 'cleaned_watch_history.csv'), index=False)
    print(f"✓ Cleaned {len(watch_clean)} watch records")
    
    print("\nCleaning titles data...")
    titles_clean = clean_titles_data(titles_df)
    print(f"✓ Cleaned {len(titles_clean)} titles")
    
    print("\nMerging datasets...")
    merged_df = merge_datasets(users_clean, watch_clean, titles_clean)
    print(f"✓ Merged dataset: {len(merged_df)} records")
    
    print("\n" + "="*50)
    print("Data Cleaning Complete!")
    print("="*50)
    print(f"\nUsers: {len(users_clean)}")
    print(f"Watch Records: {len(watch_clean)}")
    print(f"Titles: {len(titles_clean)}")
    print(f"Merged Records: {len(merged_df)}")
    
    return users_clean, watch_clean, titles_clean, merged_df

if __name__ == "__main__":
    main()


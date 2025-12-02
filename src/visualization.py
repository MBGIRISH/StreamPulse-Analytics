"""
Advanced OTT Streaming Analytics - Visualization Script
Creates comprehensive visualizations for the dashboard
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
os.makedirs('outputs/charts', exist_ok=True)

def plot_engagement_heatmap(watch_df):
    """Create engagement heatmap by day of week and hour"""
    watch_df['watch_date'] = pd.to_datetime(watch_df['watch_date'])
    watch_df['day_of_week'] = watch_df['watch_date'].dt.day_name()
    watch_df['hour'] = watch_df['watch_date'].dt.hour
    
    # Create pivot table
    heatmap_data = watch_df.groupby(['day_of_week', 'hour'])['watch_time_minutes'].sum().reset_index()
    heatmap_pivot = heatmap_data.pivot(index='day_of_week', columns='hour', values='watch_time_minutes')
    
    # Order days properly
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    heatmap_pivot = heatmap_pivot.reindex([d for d in day_order if d in heatmap_pivot.index])
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(heatmap_pivot, cmap='YlOrRd', annot=False, fmt='.0f', cbar_kws={'label': 'Watch Time (minutes)'})
    plt.title('Engagement Heatmap: Watch Time by Day of Week and Hour', fontsize=16, fontweight='bold')
    plt.xlabel('Hour of Day', fontsize=12)
    plt.ylabel('Day of Week', fontsize=12)
    plt.tight_layout()
    plt.savefig('outputs/charts/engagement_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved engagement_heatmap.png")

def plot_genre_popularity_over_time(watch_df, titles_df):
    """Plot genre popularity trends over time"""
    merged = watch_df.merge(titles_df[['title_id', 'genre']], on='title_id', how='left')
    merged['watch_date'] = pd.to_datetime(merged['watch_date'])
    merged['month'] = merged['watch_date'].dt.to_period('M')
    
    genre_monthly = merged.groupby(['month', 'genre'])['watch_time_minutes'].sum().reset_index()
    genre_monthly['month'] = genre_monthly['month'].astype(str)
    
    # Get top 6 genres
    top_genres = merged.groupby('genre')['watch_time_minutes'].sum().nlargest(6).index
    
    plt.figure(figsize=(14, 8))
    for genre in top_genres:
        data = genre_monthly[genre_monthly['genre'] == genre]
        plt.plot(data['month'], data['watch_time_minutes'], marker='o', label=genre, linewidth=2)
    
    plt.title('Genre Popularity Over Time', fontsize=16, fontweight='bold')
    plt.xlabel('Month', fontsize=12)
    plt.ylabel('Total Watch Time (minutes)', fontsize=12)
    plt.legend(loc='best', ncol=2)
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/charts/genre_popularity_over_time.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved genre_popularity_over_time.png")

def plot_rating_distribution(watch_df):
    """Plot rating distribution"""
    ratings = watch_df['rating'].dropna()
    
    plt.figure(figsize=(10, 6))
    plt.hist(ratings, bins=5, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel('Rating', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title('Rating Distribution', fontsize=16, fontweight='bold')
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig('outputs/charts/rating_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved rating_distribution.png")

def plot_device_watch_patterns(users_df, watch_df):
    """Plot watch patterns by device type"""
    merged = watch_df.merge(users_df[['user_id', 'device_type']], on='user_id', how='left')
    
    device_stats = merged.groupby('device_type').agg({
        'watch_time_minutes': ['mean', 'sum'],
        'user_id': 'nunique'
    }).round(2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Average watch time by device
    device_stats['watch_time_minutes']['mean'].plot(kind='bar', ax=axes[0], color='coral')
    axes[0].set_title('Average Watch Time by Device', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Device Type', fontsize=11)
    axes[0].set_ylabel('Average Watch Time (minutes)', fontsize=11)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # User count by device
    device_stats['user_id']['nunique'].plot(kind='bar', ax=axes[1], color='skyblue')
    axes[1].set_title('User Count by Device', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Device Type', fontsize=11)
    axes[1].set_ylabel('Number of Users', fontsize=11)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/charts/device_watch_patterns.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved device_watch_patterns.png")

def plot_churn_breakdown(users_df):
    """Plot churn behavior breakdown"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Churn by subscription plan
    churn_by_plan = users_df.groupby('subscription_plan')['churn'].mean().sort_values(ascending=False)
    churn_by_plan.plot(kind='bar', ax=axes[0, 0], color='salmon')
    axes[0, 0].set_title('Churn Rate by Subscription Plan', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Churn Rate', fontsize=10)
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Churn by device
    churn_by_device = users_df.groupby('device_type')['churn'].mean().sort_values(ascending=False)
    churn_by_device.plot(kind='bar', ax=axes[0, 1], color='lightcoral')
    axes[0, 1].set_title('Churn Rate by Device Type', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Churn Rate', fontsize=10)
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Churn by age group
    users_df['age_group'] = pd.cut(users_df['age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
    churn_by_age = users_df.groupby('age_group')['churn'].mean()
    churn_by_age.plot(kind='bar', ax=axes[1, 0], color='indianred')
    axes[1, 0].set_title('Churn Rate by Age Group', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Churn Rate', fontsize=10)
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Overall churn distribution
    churn_counts = users_df['churn'].value_counts()
    axes[1, 1].pie(churn_counts.values, labels=['Active', 'Churned'], autopct='%1.1f%%', 
                   colors=['lightgreen', 'lightcoral'], startangle=90)
    axes[1, 1].set_title('Overall Churn Distribution', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/charts/churn_breakdown.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved churn_breakdown.png")

def plot_kmeans_clusters(users_df):
    """Visualize KMeans clusters"""
    if 'cluster' not in users_df.columns:
        print("⚠ Clusters not found. Run EDA first.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Cluster by avg_watch_time and sessions_per_week
    scatter = axes[0].scatter(users_df['avg_watch_time'], users_df['sessions_per_week'], 
                             c=users_df['cluster'], cmap='viridis', alpha=0.6, s=50)
    axes[0].set_xlabel('Average Watch Time (minutes)', fontsize=11)
    axes[0].set_ylabel('Sessions per Week', fontsize=11)
    axes[0].set_title('User Clusters: Watch Time vs Sessions', fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter, ax=axes[0], label='Cluster')
    
    # Cluster distribution
    cluster_counts = users_df['cluster'].value_counts().sort_index()
    cluster_counts.plot(kind='bar', ax=axes[1], color='steelblue')
    axes[1].set_title('User Distribution by Cluster', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Cluster', fontsize=11)
    axes[1].set_ylabel('Number of Users', fontsize=11)
    axes[1].tick_params(axis='x', rotation=0)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('outputs/charts/kmeans_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Saved kmeans_clusters.png")

def main():
    """Main visualization function"""
    print("Loading datasets...")
    users_df = pd.read_csv('outputs/cleaned_users.csv')
    watch_df = pd.read_csv('outputs/cleaned_watch_history.csv')
    titles_df = pd.read_csv('data/ott_titles.csv')
    
    print("\n" + "="*50)
    print("GENERATING VISUALIZATIONS")
    print("="*50)
    
    print("\n1. Creating engagement heatmap...")
    plot_engagement_heatmap(watch_df)
    
    print("\n2. Creating genre popularity over time...")
    plot_genre_popularity_over_time(watch_df, titles_df)
    
    print("\n3. Creating rating distribution...")
    plot_rating_distribution(watch_df)
    
    print("\n4. Creating device watch patterns...")
    plot_device_watch_patterns(users_df, watch_df)
    
    print("\n5. Creating churn breakdown...")
    plot_churn_breakdown(users_df)
    
    print("\n6. Creating KMeans cluster visualization...")
    plot_kmeans_clusters(users_df)
    
    print("\n" + "="*50)
    print("All visualizations saved to outputs/charts/")
    print("="*50)

if __name__ == "__main__":
    main()


"""
Advanced OTT Streaming Analytics - Premium Netflix-Style Dashboard
Modern, interactive analytics and recommendation dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
import os
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.path_utils import get_project_paths

# Page config
st.set_page_config(
    page_title="OTT Streaming Analytics",
    page_icon="📺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Theme CSS
st.markdown("""
    <style>
    /* Main App Container */
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
        animation: fadeIn 1s ease;
    }
    
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(10px);}
        to {opacity: 1; transform: translateY(0);}
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d29 0%, #0E1117 100%);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] {
        color: #ffffff;
    }
    
    /* Main Content Area */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Glassmorphism Metric Cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 25px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        text-align: center;
        transition: transform 0.3s ease, box-shadow 0.3s ease;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    .metric-card:hover {
        transform: scale(1.03);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.5);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #ffffff;
        margin: 10px 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #b0b0b0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-icon {
        font-size: 2rem;
        margin-bottom: 10px;
    }
    
    /* Header Styling */
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 3rem;
        padding: 1rem;
    }
    
    /* Section Headers */
    h1, h2, h3 {
        color: #ffffff !important;
    }
    
    /* Chart Containers */
    .chart-container {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        padding: 20px;
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3);
    }
    
    /* Recommendation Cards */
    .rec-card {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        transition: transform 0.3s ease;
        height: 100%;
    }
    
    .rec-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px 0 rgba(0, 0, 0, 0.4);
    }
    
    /* Genre Filter Chips */
    .genre-chip {
        display: inline-block;
        padding: 8px 16px;
        margin: 5px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        color: #ffffff;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .genre-chip:hover {
        background: rgba(255, 255, 255, 0.2);
        transform: scale(1.05);
    }
    
    .genre-chip.active {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-color: #667eea;
    }
    
    /* Sticky Header */
    .sticky-header {
        position: sticky;
        top: 0;
        z-index: 100;
        background: rgba(14, 17, 23, 0.95);
        backdrop-filter: blur(10px);
        padding: 1rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 2rem;
    }
    
    /* Dataframe Styling */
    .dataframe {
        background: rgba(255, 255, 255, 0.03);
        border-radius: 10px;
    }
    
    /* Streamlit Widgets */
    .stSelectbox, .stMultiselect {
        background-color: rgba(255, 255, 255, 0.05);
    }
    
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all datasets"""
    try:
        paths = get_project_paths()
        users_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_users.csv'))
        watch_df = pd.read_csv(os.path.join(paths['outputs'], 'cleaned_watch_history.csv'))
        titles_df = pd.read_csv(os.path.join(paths['data'], 'ott_titles.csv'))
        return users_df, watch_df, titles_df
    except FileNotFoundError as e:
        st.error(f"Data files not found: {e}. Please run the data generation and cleaning scripts first.")
        return None, None, None

def load_recommendations():
    """Load sample recommendations"""
    try:
        paths = get_project_paths()
        return pd.read_csv(os.path.join(paths['outputs'], 'sample_recommendations.csv'))
    except:
        return None

def render_metric_card(icon, label, value, delta=None):
    """Render a glassmorphism metric card"""
    delta_html = f'<div style="font-size: 0.8rem; color: #4ade80; margin-top: 5px;">{delta}</div>' if delta else ''
    return f"""
    <div class="metric-card">
        <div class="metric-icon">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """

def main():
    """Main dashboard function"""
    # Sticky Header
    st.markdown("""
        <div class="sticky-header">
            <h1 class="main-header">📺 OTT Streaming Analytics Dashboard</h1>
        </div>
    """, unsafe_allow_html=True)
    
    # Load data
    users_df, watch_df, titles_df = load_data()
    
    if users_df is None:
        st.stop()
    
    # Sidebar with improved styling
    with st.sidebar:
        st.markdown("## 🎬 Navigation")
        st.markdown("---")
        page = st.selectbox(
            "Choose a page",
            ["📊 OTT Analytics Overview", "🧩 User Segments", "🔥 Trending Genres", 
             "💔 Churn Prediction", "🎯 Recommendation Engine"],
            label_visibility="collapsed"
        )
        st.markdown("---")
        st.markdown("### 📈 Quick Stats")
        st.metric("Total Users", f"{len(users_df):,}")
        st.metric("Active Users", f"{len(users_df[users_df['churn'] == 0]):,}")
    
    # Convert dates
    users_df['join_date'] = pd.to_datetime(users_df['join_date'])
    watch_df['watch_date'] = pd.to_datetime(watch_df['watch_date'])
    
    if page == "📊 OTT Analytics Overview":
        render_overview(users_df, watch_df, titles_df)
    elif page == "🧩 User Segments":
        render_user_segments(users_df, watch_df)
    elif page == "🔥 Trending Genres":
        render_genre_page(watch_df, titles_df, users_df)
    elif page == "💔 Churn Prediction":
        render_churn_prediction(users_df, watch_df)
    elif page == "🎯 Recommendation Engine":
        render_recommendations(users_df, watch_df, titles_df)

def render_overview(users_df, watch_df, titles_df):
    """Render overview analytics with premium UI"""
    st.markdown("## 📊 OTT Analytics Overview")
    st.markdown("---")
    
    # KPI Cards
    col1, col2, col3, col4 = st.columns(4)
    
    total_users = len(users_df)
    active_users = len(users_df[users_df['churn'] == 0])
    churn_rate = users_df['churn'].mean()
    total_watch_time = watch_df['watch_time_minutes'].sum()
    
    with col1:
        st.markdown(render_metric_card("👥", "Total Users", f"{total_users:,}"), unsafe_allow_html=True)
    
    with col2:
        st.markdown(render_metric_card("✅", "Active Users", f"{active_users:,}"), unsafe_allow_html=True)
    
    with col3:
        st.markdown(render_metric_card("💔", "Churn Rate", f"{churn_rate:.2%}"), unsafe_allow_html=True)
    
    with col4:
        st.markdown(render_metric_card("⏱️", "Total Watch Time", f"{total_watch_time/60:.0f} hours"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts Section
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 User Distribution by Subscription Plan")
        plan_counts = users_df['subscription_plan'].value_counts()
        fig = px.bar(
            x=plan_counts.index,
            y=plan_counts.values,
            color=plan_counts.values,
            color_continuous_scale='Viridis',
            labels={'x': 'Subscription Plan', 'y': 'Number of Users'},
            title=''
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=False,
            height=400
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📱 Device Type Distribution")
        device_counts = users_df['device_type'].value_counts()
        colors = ['#667eea', '#764ba2', '#f093fb', '#4facfe']
        fig = px.pie(
            values=device_counts.values,
            names=device_counts.index,
            color_discrete_sequence=colors,
            title=''
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            height=400,
            showlegend=True
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Watch Time Trends
    st.markdown("### 📈 Watch Time Trends Over Time")
    watch_df['month'] = watch_df['watch_date'].dt.to_period('M')
    monthly_watch = watch_df.groupby('month')['watch_time_minutes'].sum().reset_index()
    monthly_watch['month'] = monthly_watch['month'].astype(str)
    
    fig = px.line(
        monthly_watch,
        x='month',
        y='watch_time_minutes',
        markers=True,
        labels={'watch_time_minutes': 'Total Watch Time (minutes)', 'month': 'Month'},
        title=''
    )
    fig.update_traces(line_color='#667eea', line_width=3, marker_size=8)
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        height=450,
        xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Top Titles
    st.markdown("### 🎬 Top 10 Most Watched Titles")
    top_titles = watch_df.groupby('title_id').agg({
        'watch_time_minutes': 'sum',
        'user_id': 'nunique'
    }).reset_index()
    top_titles = top_titles.merge(titles_df[['title_id', 'title_name', 'genre']], on='title_id')
    top_titles = top_titles.sort_values('watch_time_minutes', ascending=False).head(10)
    top_titles_display = top_titles[['title_name', 'genre', 'watch_time_minutes', 'user_id']].copy()
    top_titles_display.columns = ['Title Name', 'Genre', 'Watch Time (min)', 'Viewers']
    top_titles_display['Watch Time (min)'] = top_titles_display['Watch Time (min)'].astype(int)
    st.dataframe(top_titles_display, use_container_width=True, height=400)

def render_user_segments(users_df, watch_df):
    """Render user segmentation analysis"""
    st.markdown("## 🧩 User Segments")
    st.markdown("---")
    
    if 'cluster' not in users_df.columns:
        st.warning("⚠️ Clusters not found. Please run the EDA script first.")
        st.info("Run: `python src/eda.py` to generate user clusters.")
        return
    
    # Cluster Summary
    st.markdown("### 📊 User Cluster Summary")
    cluster_summary = users_df.groupby('cluster', observed=True).agg({
        'user_id': 'count',
        'avg_watch_time': 'mean',
        'sessions_per_week': 'mean',
        'age': 'mean',
        'churn': 'mean'
    }).round(2)
    cluster_summary.columns = ['User Count', 'Avg Watch Time', 'Sessions/Week', 'Avg Age', 'Churn Rate']
    cluster_summary['Churn Rate'] = cluster_summary['Churn Rate'].apply(lambda x: f"{x:.2%}")
    st.dataframe(cluster_summary, use_container_width=True, height=300)
    
    # Cluster Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Cluster Distribution")
        cluster_counts = users_df['cluster'].value_counts().sort_index()
        fig = px.bar(
            x=cluster_counts.index.astype(str),
            y=cluster_counts.values,
            color=cluster_counts.values,
            color_continuous_scale='Blues',
            labels={'x': 'Cluster', 'y': 'Number of Users'},
            title=''
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=False,
            height=400
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 💔 Churn Rate by Cluster")
        churn_by_cluster = users_df.groupby('cluster', observed=True)['churn'].mean()
        fig = px.bar(
            x=churn_by_cluster.index.astype(str),
            y=churn_by_cluster.values,
            color=churn_by_cluster.values,
            color_continuous_scale='Reds',
            labels={'x': 'Cluster', 'y': 'Churn Rate'},
            title=''
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=False,
            height=400
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Cluster Characteristics
    st.markdown("### 🔍 Cluster Characteristics")
    selected_cluster = st.selectbox("Select Cluster to Explore", sorted(users_df['cluster'].unique()))
    
    cluster_users = users_df[users_df['cluster'] == selected_cluster]
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(render_metric_card("👥", "Users in Cluster", f"{len(cluster_users):,}"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("⏱️", "Avg Watch Time", f"{cluster_users['avg_watch_time'].mean():.1f} min"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("💔", "Churn Rate", f"{cluster_users['churn'].mean():.2%}"), unsafe_allow_html=True)

def render_genre_page(watch_df, titles_df, users_df):
    """Render trending genres page with premium UI"""
    st.markdown("## 🔥 Trending Genres")
    st.markdown("### Discover what's hot in streaming")
    st.markdown("---")
    
    merged = watch_df.merge(titles_df[['title_id', 'genre']], on='title_id', how='left')
    
    # Genre Popularity Table
    st.markdown("### 📊 Genre Popularity Rankings")
    genre_stats = merged.groupby('genre').agg({
        'watch_time_minutes': 'sum',
        'user_id': 'nunique',
        'rating': 'mean'
    }).round(2)
    genre_stats.columns = ['Total Watch Time (min)', 'Unique Viewers', 'Avg Rating']
    genre_stats = genre_stats.sort_values('Total Watch Time (min)', ascending=False)
    st.dataframe(genre_stats, use_container_width=True, height=400)
    
    # Genre Trends Over Time
    st.markdown("### 📈 Genre Trends Over Time")
    st.markdown("Select genres to compare their popularity trends")
    
    merged['month'] = merged['watch_date'].dt.to_period('M')
    all_genres = sorted(merged['genre'].dropna().unique())
    top_genres = merged['genre'].value_counts().head(6).index.tolist()
    
    selected_genres = st.multiselect(
        "Select Genres",
        options=all_genres,
        default=top_genres,
        label_visibility="collapsed"
    )
    
    if selected_genres:
        genre_monthly = merged[merged['genre'].isin(selected_genres)].groupby(['month', 'genre'])['watch_time_minutes'].sum().reset_index()
        genre_monthly['month'] = genre_monthly['month'].astype(str)
        
        fig = px.line(
            genre_monthly,
            x='month',
            y='watch_time_minutes',
            color='genre',
            markers=True,
            labels={'watch_time_minutes': 'Total Watch Time (minutes)', 'month': 'Month', 'genre': 'Genre'},
            title=''
        )
        fig.update_traces(line_width=3, marker_size=8)
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            height=500,
            legend=dict(bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.2)'),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.1)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Genre by Device Heatmap
    st.markdown("### 🔥 Genre Preference by Device")
    merged_with_device = merged.merge(users_df[['user_id', 'device_type']], on='user_id', how='left')
    device_genre = merged_with_device.groupby(['device_type', 'genre'])['watch_time_minutes'].sum().reset_index()
    device_genre_pivot = device_genre.pivot(index='genre', columns='device_type', values='watch_time_minutes').fillna(0)
    
    fig = px.imshow(
        device_genre_pivot.values,
        labels=dict(x="Device Type", y="Genre", color="Watch Time"),
        x=device_genre_pivot.columns,
        y=device_genre_pivot.index,
        color_continuous_scale='YlOrRd',
        aspect="auto",
        text_auto='.0f'
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        height=600
    )
    st.plotly_chart(fig, use_container_width=True)

def render_churn_prediction(users_df, watch_df):
    """Render churn prediction analysis"""
    st.markdown("## 💔 Churn Prediction Analysis")
    st.markdown("---")
    
    # Churn Statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(render_metric_card("📊", "Overall Churn Rate", f"{users_df['churn'].mean():.2%}"), unsafe_allow_html=True)
    with col2:
        st.markdown(render_metric_card("👋", "Churned Users", f"{users_df['churn'].sum():,}"), unsafe_allow_html=True)
    with col3:
        st.markdown(render_metric_card("✅", "Active Users", f"{len(users_df[users_df['churn'] == 0]):,}"), unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Churn by Dimensions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 💳 Churn Rate by Subscription Plan")
        churn_by_plan = users_df.groupby('subscription_plan', observed=True)['churn'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=churn_by_plan.index,
            y=churn_by_plan.values,
            color=churn_by_plan.values,
            color_continuous_scale='Reds',
            labels={'x': 'Subscription Plan', 'y': 'Churn Rate'},
            title=''
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=False,
            height=400
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### 📱 Churn Rate by Device Type")
        churn_by_device = users_df.groupby('device_type', observed=True)['churn'].mean().sort_values(ascending=False)
        fig = px.bar(
            x=churn_by_device.index,
            y=churn_by_device.values,
            color=churn_by_device.values,
            color_continuous_scale='Oranges',
            labels={'x': 'Device Type', 'y': 'Churn Rate'},
            title=''
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#ffffff',
            showlegend=False,
            height=400
        )
        fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
        fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
        st.plotly_chart(fig, use_container_width=True)
    
    # Churn by Age Groups
    users_df['age_group'] = pd.cut(users_df['age'], bins=[0, 25, 35, 50, 100], labels=['18-25', '26-35', '36-50', '50+'])
    churn_by_age = users_df.groupby('age_group', observed=True)['churn'].mean()
    
    st.markdown("### 👥 Churn Rate by Age Group")
    fig = px.bar(
        x=churn_by_age.index.astype(str),
        y=churn_by_age.values,
        color=churn_by_age.values,
        color_continuous_scale='Reds',
        labels={'x': 'Age Group', 'y': 'Churn Rate'},
        title=''
    )
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font_color='#ffffff',
        showlegend=False,
        height=450
    )
    fig.update_xaxes(gridcolor='rgba(255,255,255,0.1)')
    fig.update_yaxes(gridcolor='rgba(255,255,255,0.1)')
    st.plotly_chart(fig, use_container_width=True)
    
    # Model Metrics
    try:
        paths = get_project_paths()
        with open(os.path.join(paths['outputs'], 'churn_model_metrics.txt'), 'r') as f:
            metrics_text = f.read()
        st.markdown("### 📈 Model Performance Metrics")
        st.code(metrics_text, language='text')
    except:
        st.info("💡 Run the churn prediction model to see performance metrics: `python src/churn_model.py`")

def render_recommendations(users_df, watch_df, titles_df):
    """Render recommendation engine with premium UI"""
    st.markdown("## 🎯 Recommendation Engine")
    st.markdown("### Personalized content recommendations")
    st.markdown("---")
    
    # User Selection
    user_id = st.selectbox(
        "👤 Select User ID",
        options=sorted(users_df['user_id'].unique())[:100],
        label_visibility="visible"
    )
    
    if user_id:
        user_info = users_df[users_df['user_id'] == user_id].iloc[0]
        
        # User Summary Cards
        st.markdown("### 👤 User Profile")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(render_metric_card("💳", "Subscription Plan", user_info['subscription_plan'].title()), unsafe_allow_html=True)
        with col2:
            st.markdown(render_metric_card("🎬", "Preferred Genre", user_info['preferred_genre']), unsafe_allow_html=True)
        with col3:
            st.markdown(render_metric_card("⏱️", "Avg Watch Time", f"{user_info['avg_watch_time']:.1f} min"), unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # User's Watch History
        st.markdown("### 📺 Watch History")
        user_watched = watch_df[watch_df['user_id'] == user_id].merge(
            titles_df[['title_id', 'title_name', 'genre']], on='title_id'
        )
        user_watched_summary = user_watched.groupby(['title_id', 'title_name', 'genre']).agg({
            'watch_time_minutes': 'sum',
            'rating': 'mean'
        }).reset_index()
        user_watched_summary = user_watched_summary.sort_values('watch_time_minutes', ascending=False).head(10)
        user_watched_display = user_watched_summary[['title_name', 'genre', 'watch_time_minutes', 'rating']].copy()
        user_watched_display.columns = ['Title', 'Genre', 'Watch Time (min)', 'Rating']
        user_watched_display['Watch Time (min)'] = user_watched_display['Watch Time (min)'].astype(int)
        st.dataframe(user_watched_display, use_container_width=True, height=300)
        
        # Recommendations
        st.markdown("### 🎬 Recommended Titles")
        
        sample_recs = load_recommendations()
        
        if sample_recs is not None and user_id in sample_recs['user_id'].values:
            user_recs = sample_recs[sample_recs['user_id'] == user_id].head(12)
            recommendations_list = user_recs[['title_name', 'genre', 'hybrid_score']].to_dict('records')
        else:
            # Fallback recommendations
            user_preferred_genre = user_info['preferred_genre']
            user_watched_ids = user_watched['title_id'].unique()
            recommendations = titles_df[
                (titles_df['genre'] == user_preferred_genre) & 
                (~titles_df['title_id'].isin(user_watched_ids))
            ].sort_values('popularity_score', ascending=False).head(12)
            recommendations_list = recommendations[['title_name', 'genre', 'popularity_score']].rename(
                columns={'popularity_score': 'hybrid_score'}
            ).to_dict('records')
        
        if recommendations_list:
            # Display as cards in 3 columns
            cols = st.columns(3)
            for idx, rec in enumerate(recommendations_list):
                with cols[idx % 3]:
                    score = rec.get('hybrid_score', 0)
                    score_label = 'Score' if 'hybrid_score' in rec else 'Popularity'
                    st.markdown(f"""
                    <div class="rec-card">
                        <h4 style="color: #ffffff; margin-bottom: 10px;">{rec['title_name']}</h4>
                        <p style="color: #b0b0b0; font-size: 0.9rem;">🎭 {rec['genre']}</p>
                        <p style="color: #667eea; font-weight: bold;">⭐ {score:.2f} {score_label}</p>
                        <button style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                      color: white; border: none; padding: 8px 16px; 
                                      border-radius: 8px; cursor: pointer; width: 100%; margin-top: 10px;">
                            ▶ Watch Now
                        </button>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("No recommendations available. Try running the hybrid recommender: `python src/hybrid_recommender.py`")

if __name__ == "__main__":
    main()

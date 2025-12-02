# Advanced OTT Streaming Analytics

A comprehensive analytics system for OTT streaming platforms featuring user insights, churn prediction, and a hybrid recommendation engine.

## What This Project Does

This project provides a complete end-to-end analytics solution for OTT (Over-The-Top) streaming platforms:

1. **Data Generation**: Generates realistic synthetic datasets with 10,000+ users, 2,000+ titles, and 150,000+ watch history records
2. **Data Analysis**: Performs comprehensive exploratory data analysis including genre engagement, user segmentation, and behavioral patterns
3. **Churn Prediction**: Trains machine learning models to predict user churn with high accuracy
4. **Recommendation Engine**: Implements content-based, collaborative filtering, and hybrid recommendation systems
5. **Interactive Dashboard**: Provides a Streamlit-based dashboard for visualizing insights and generating recommendations

## Datasets Used

The project uses **3 synthetic datasets** generated programmatically:

1. **ott_users.csv** (10,000 users)
   - User demographics (age, gender, country)
   - Subscription plans (basic, standard, premium)
   - Device types (TV, tablet, mobile, laptop)
   - Watch behavior (avg_watch_time, sessions_per_week)
   - Churn status (binary: 0/1)
   - Preferred genres

2. **ott_watch_history.csv** (150,000+ records)
   - User watch history with timestamps
   - Watch duration (minutes)
   - User ratings (1-5 scale)
   - Watch dates

3. **ott_titles.csv** (2,000 titles)
   - Title metadata (name, genre, release year)
   - Age ratings (G, PG, PG-13, R, NC-17)
   - Popularity scores
   - Descriptions

**Note**: All datasets are synthetically generated with realistic statistical distributions for demonstration purposes.

## ML Models for Prediction

### 1. Churn Prediction Models
Three supervised learning models are trained and compared:

- **Logistic Regression** - Linear classification model
- **Random Forest Classifier** - Ensemble tree-based model (100 trees)
- **Gradient Boosting Classifier** - Advanced ensemble model (100 estimators)

**Performance**: Achieves 85-90% accuracy, 80-85% precision, 75-80% recall, and 0.85-0.90 ROC-AUC score.

**Features Used**: Age, subscription plan, device type, watch behavior (recency, frequency, duration), engagement metrics, and demographics.

### 2. User Segmentation Model
- **KMeans Clustering** - Unsupervised learning to identify 4 distinct user segments based on watch patterns

### 3. Recommendation Models
- **Content-Based Filtering**: TF-IDF vectorization + Cosine Similarity
- **Collaborative Filtering**: 
  - **SVD** (Singular Value Decomposition) - Matrix factorization for rating prediction
  - **NMF** (Non-negative Matrix Factorization) - Fallback when Surprise library unavailable
  - **KNNBaseline** - User-based collaborative filtering (if Surprise available)
- **Hybrid Recommender**: Weighted combination (40% content-based + 60% collaborative)

## Tech Stack

### Core Technologies
- **Python 3.8+** - Primary programming language
- **Jupyter Notebook** - Interactive data analysis and experimentation

### Data Processing & Analysis
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing

### Machine Learning
- **Scikit-learn** - Machine learning algorithms (Logistic Regression, Random Forest, Gradient Boosting, KMeans, NMF)
- **Surprise** - Collaborative filtering library (with NMF fallback)

### Visualization
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical data visualization
- **Plotly** - Interactive visualizations

### Dashboard
- **Streamlit** - Web-based interactive dashboard framework

### Development Tools
- **Git** - Version control

## Project Structure

```
StreamPulse-Analytics/
├── data/                    # Generated datasets
├── notebooks/               # Jupyter notebook for end-to-end analysis
├── src/                     # Python source code
│   ├── generate_data.py
│   ├── data_cleaning.py
│   ├── eda.py
│   ├── visualization.py
│   ├── churn_model.py
│   ├── content_based_recommender.py
│   ├── collaborative_filtering.py
│   ├── hybrid_recommender.py
│   └── path_utils.py
├── dashboard/               # Streamlit dashboard
│   └── app.py
├── outputs/                 # Generated outputs (charts, models, recommendations)
├── requirements.txt         # Python dependencies
└── README.md
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Jupyter notebook:**
   ```bash
   jupyter notebook notebooks/advanced_analysis.ipynb
   ```

3. **Or launch the Streamlit dashboard:**
   ```bash
   streamlit run dashboard/app.py
   ```

## Key Features

- **User Segmentation**: KMeans clustering to identify distinct user groups
- **Churn Prediction**: Multiple ML models achieving 85-90% accuracy
- **Hybrid Recommendations**: Combines content-based and collaborative filtering
- **Interactive Visualizations**: Plotly charts with zoom, pan, and hover features
- **Modern Dashboard**: Netflix-style dark theme with glassmorphism UI

## Summary

**Datasets**: 3 synthetic CSV files (users, watch history, titles) with 10,000+ users and 150,000+ watch records

**ML Models**: 
- **Churn Prediction**: Logistic Regression, Random Forest, Gradient Boosting
- **User Segmentation**: KMeans Clustering
- **Recommendations**: SVD/NMF, KNNBaseline, TF-IDF + Cosine Similarity

# Advanced OTT Streaming Analytics

A comprehensive analytics system for OTT streaming platforms featuring user insights, churn prediction, and a hybrid recommendation engine.

## What This Project Does

This project provides a complete end-to-end analytics solution for OTT (Over-The-Top) streaming platforms:

1. **Data Generation**: Generates realistic synthetic datasets with 10,000+ users, 2,000+ titles, and 150,000+ watch history records
2. **Data Analysis**: Performs comprehensive exploratory data analysis including genre engagement, user segmentation, and behavioral patterns
3. **Churn Prediction**: Trains machine learning models to predict user churn with high accuracy
4. **Recommendation Engine**: Implements content-based, collaborative filtering, and hybrid recommendation systems
5. **Interactive Dashboard**: Provides a Streamlit-based dashboard for visualizing insights and generating recommendations

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

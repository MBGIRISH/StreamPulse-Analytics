export enum SubscriptionPlan {
  BASIC = 'Basic',
  STANDARD = 'Standard',
  PREMIUM = 'Premium'
}

export enum DeviceType {
  TV = 'TV',
  TABLET = 'Tablet',
  MOBILE = 'Mobile',
  LAPTOP = 'Laptop'
}

export interface User {
  user_id: string;
  age: number;
  gender: 'Male' | 'Female' | 'Other';
  country: string;
  subscription_plan: SubscriptionPlan;
  device_type: DeviceType;
  churn: number; // 0 or 1
  avg_watch_time: number;
  sessions_per_week: number;
  preferred_genre: string;
  join_date: string;
  last_active_date: string;
}

export interface Title {
  title_id: string;
  title_name: string;
  genre: string;
  release_year: number;
  age_rating: string;
  popularity_score: number;
  description: string;
}

export interface WatchHistory {
  user_id: string;
  title_id: string;
  watch_time_minutes: number;
  rating: number; // 1-5
  watch_date: string;
}

export interface ChurnPrediction {
  probability: number;
  riskLevel: 'Low' | 'Medium' | 'High';
  factors: string[];
  retentionStrategy: string;
}

export interface Recommendation {
  title_name: string;
  match_score: number;
  reason: string;
}

export enum ViewState {
  DASHBOARD = 'DASHBOARD',
  USERS = 'USERS',
  RECOMMENDATIONS = 'RECOMMENDATIONS',
  CHURN_ANALYSIS = 'CHURN_ANALYSIS',
  AI_INSIGHTS = 'AI_INSIGHTS'
}
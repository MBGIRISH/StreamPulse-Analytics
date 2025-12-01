import { User, Title, WatchHistory, SubscriptionPlan, DeviceType } from '../types';

// Helper to get random item from array
const getRandom = <T,>(arr: T[]): T => arr[Math.floor(Math.random() * arr.length)];
const getRandomInt = (min: number, max: number): number => Math.floor(Math.random() * (max - min + 1)) + min;

const GENRES = ['Action', 'Sci-Fi', 'Drama', 'Comedy', 'Thriller', 'Documentary', 'Horror', 'Romance'];
const COUNTRIES = ['USA', 'Canada', 'UK', 'India', 'Germany', 'Australia', 'Japan', 'Brazil'];

export const generateUsers = (count: number): User[] => {
  const users: User[] = [];
  for (let i = 0; i < count; i++) {
    const isChurned = Math.random() < 0.2 ? 1 : 0; // 20% churn rate simulation
    users.push({
      user_id: `U-${1000 + i}`,
      age: getRandomInt(18, 70),
      gender: getRandom(['Male', 'Female', 'Other']),
      country: getRandom(COUNTRIES),
      subscription_plan: getRandom(Object.values(SubscriptionPlan)),
      device_type: getRandom(Object.values(DeviceType)),
      churn: isChurned,
      avg_watch_time: getRandomInt(15, 180),
      sessions_per_week: getRandomInt(1, 14),
      preferred_genre: getRandom(GENRES),
      join_date: new Date(Date.now() - getRandomInt(1, 730) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      last_active_date: new Date(Date.now() - getRandomInt(0, 30) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
    });
  }
  return users;
};

export const generateTitles = (count: number): Title[] => {
  const titles: Title[] = [];
  const adjectives = ['Dark', 'Rising', 'Lost', 'Eternal', 'Silent', 'Brave', 'Hidden', 'Last'];
  const nouns = ['Kingdom', 'Star', 'Memory', 'Future', 'Detectives', 'Love', 'Warrior', 'Mystery'];

  for (let i = 0; i < count; i++) {
    titles.push({
      title_id: `T-${1000 + i}`,
      title_name: `The ${getRandom(adjectives)} ${getRandom(nouns)} ${getRandomInt(1, 99)}`,
      genre: getRandom(GENRES),
      release_year: getRandomInt(1990, 2024),
      age_rating: getRandom(['PG', 'PG-13', 'R', 'TV-MA']),
      popularity_score: getRandomInt(50, 100),
      description: 'A gripping tale of adventure and suspense that will keep you on the edge of your seat.',
    });
  }
  return titles;
};

export const generateWatchHistory = (users: User[], titles: Title[], entriesPerUser: number): WatchHistory[] => {
  const history: WatchHistory[] = [];
  users.forEach(user => {
    const numEntries = getRandomInt(1, entriesPerUser);
    for (let i = 0; i < numEntries; i++) {
      const title = getRandom(titles);
      history.push({
        user_id: user.user_id,
        title_id: title.title_id,
        watch_time_minutes: getRandomInt(10, 150),
        rating: getRandomInt(1, 5),
        watch_date: new Date(Date.now() - getRandomInt(0, 90) * 24 * 60 * 60 * 1000).toISOString().split('T')[0],
      });
    }
  });
  return history;
};

export const getAggregatedGenreData = (history: WatchHistory[], titles: Title[]) => {
  const genreCounts: Record<string, number> = {};
  history.forEach(h => {
    const title = titles.find(t => t.title_id === h.title_id);
    if (title) {
      genreCounts[title.genre] = (genreCounts[title.genre] || 0) + 1;
    }
  });
  return Object.entries(genreCounts).map(([name, value]) => ({ name, value })).sort((a, b) => b.value - a.value);
};

export const getEngagementOverTime = (history: WatchHistory[]) => {
  const dateCounts: Record<string, number> = {};
  history.forEach(h => {
    dateCounts[h.watch_date] = (dateCounts[h.watch_date] || 0) + h.watch_time_minutes;
  });
  // Sort by date and take last 14 entries for cleaner chart
  return Object.entries(dateCounts)
    .sort((a, b) => new Date(a[0]).getTime() - new Date(b[0]).getTime())
    .slice(-14)
    .map(([date, minutes]) => ({ date, minutes }));
};
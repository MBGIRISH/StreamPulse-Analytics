import React, { useState, useEffect } from 'react';
import { Sidebar } from './components/Sidebar';
import { Dashboard } from './components/Dashboard';
import { ChurnAnalysis } from './components/ChurnAnalysis';
import { RecommendationEngine } from './components/RecommendationEngine';
import { AIAnalyst } from './components/AIAnalyst';
import { ViewState, User, Title, WatchHistory } from './types';
import { generateUsers, generateTitles, generateWatchHistory } from './services/dataService';

const App: React.FC = () => {
  const [currentView, setCurrentView] = useState<ViewState>(ViewState.DASHBOARD);
  
  // State for mock data
  const [users, setUsers] = useState<User[]>([]);
  const [titles, setTitles] = useState<Title[]>([]);
  const [history, setHistory] = useState<WatchHistory[]>([]);
  const [isDataLoaded, setIsDataLoaded] = useState(false);

  // Initialize Data
  useEffect(() => {
    // Simulate fetching data
    const u = generateUsers(100); // Generate 100 users
    const t = generateTitles(50); // Generate 50 movies
    const h = generateWatchHistory(u, t, 20); // Generate Logs

    setUsers(u);
    setTitles(t);
    setHistory(h);
    setIsDataLoaded(true);
  }, []);

  if (!isDataLoaded) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center text-white">
        <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mb-4"></div>
            <p className="text-gray-400">Loading Datasets...</p>
        </div>
      </div>
    );
  }

  const renderView = () => {
    switch (currentView) {
      case ViewState.DASHBOARD:
        return <Dashboard users={users} history={history} titles={titles} />;
      case ViewState.CHURN_ANALYSIS:
        return <ChurnAnalysis users={users} history={history} />;
      case ViewState.RECOMMENDATIONS:
        return <RecommendationEngine users={users} history={history} titles={titles} />;
      case ViewState.AI_INSIGHTS:
        return <AIAnalyst users={users} titles={titles} />;
      case ViewState.USERS:
        return (
          <div className="p-8 text-white">
            <h2 className="text-2xl font-bold mb-4">User Management</h2>
            <div className="bg-gray-800 rounded-lg overflow-hidden border border-gray-700">
               <table className="w-full text-left">
                 <thead className="bg-gray-900 text-gray-400">
                    <tr>
                        <th className="p-4">User ID</th>
                        <th className="p-4">Plan</th>
                        <th className="p-4">Country</th>
                        <th className="p-4">Status</th>
                    </tr>
                 </thead>
                 <tbody className="divide-y divide-gray-700">
                    {users.slice(0, 15).map(u => (
                        <tr key={u.user_id} className="hover:bg-gray-700/50">
                            <td className="p-4 font-mono text-sm">{u.user_id}</td>
                            <td className="p-4">{u.subscription_plan}</td>
                            <td className="p-4">{u.country}</td>
                            <td className="p-4">
                                <span className={`px-2 py-1 rounded text-xs font-bold ${u.churn ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                                    {u.churn ? 'Churned' : 'Active'}
                                </span>
                            </td>
                        </tr>
                    ))}
                 </tbody>
               </table>
            </div>
          </div>
        );
      default:
        return <div className="p-8 text-white">Select a view</div>;
    }
  };

  return (
    <div className="flex h-screen bg-gray-900 text-gray-100 overflow-hidden font-sans">
      <Sidebar currentView={currentView} setView={setCurrentView} />
      <main className="flex-1 overflow-hidden relative bg-[url('https://grainy-gradients.vercel.app/noise.svg')] bg-opacity-5">
        {/* Background gradient overlay for aesthetics */}
        <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-gray-900 to-purple-900/10 pointer-events-none z-0" />
        <div className="relative z-10 h-full">
            {renderView()}
        </div>
      </main>
    </div>
  );
};

export default App;
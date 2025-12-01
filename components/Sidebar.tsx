import React from 'react';
import { ViewState } from '../types';
import { 
  LayoutDashboard, 
  Users, 
  Tv, 
  Activity, 
  BrainCircuit,
  Settings 
} from 'lucide-react';

interface SidebarProps {
  currentView: ViewState;
  setView: (view: ViewState) => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ currentView, setView }) => {
  const menuItems = [
    { id: ViewState.DASHBOARD, label: 'Analytics Hub', icon: LayoutDashboard },
    { id: ViewState.USERS, label: 'User Segments', icon: Users },
    { id: ViewState.RECOMMENDATIONS, label: 'Hybrid Recommender', icon: Tv },
    { id: ViewState.CHURN_ANALYSIS, label: 'Churn Prediction', icon: Activity },
    { id: ViewState.AI_INSIGHTS, label: 'AI Analyst', icon: BrainCircuit },
  ];

  return (
    <div className="w-64 h-screen bg-gray-900 border-r border-gray-800 flex flex-col shadow-xl">
      <div className="p-6 flex items-center space-x-3">
        <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-lg flex items-center justify-center">
          <Activity className="text-white w-5 h-5" />
        </div>
        <span className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-white to-gray-400">
          StreamPulse
        </span>
      </div>

      <nav className="flex-1 px-4 space-y-2 mt-4">
        {menuItems.map((item) => {
          const Icon = item.icon;
          const isActive = currentView === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setView(item.id)}
              className={`w-full flex items-center space-x-3 px-4 py-3 rounded-xl transition-all duration-200 group ${
                isActive 
                  ? 'bg-purple-600/10 text-purple-400 border border-purple-600/20' 
                  : 'text-gray-400 hover:bg-gray-800 hover:text-white'
              }`}
            >
              <Icon size={20} className={isActive ? 'text-purple-400' : 'text-gray-500 group-hover:text-white'} />
              <span className="font-medium text-sm">{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 border-t border-gray-800">
        <button className="flex items-center space-x-3 text-gray-400 hover:text-white transition-colors px-4 py-2 w-full">
          <Settings size={18} />
          <span className="text-sm">Settings</span>
        </button>
      </div>
    </div>
  );
};
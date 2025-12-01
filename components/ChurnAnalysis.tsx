import React, { useState } from 'react';
import { User, WatchHistory, ChurnPrediction } from '../types';
import { analyzeChurnRisk } from '../services/geminiService';
import { AlertTriangle, CheckCircle, RefreshCcw, Search } from 'lucide-react';

interface ChurnAnalysisProps {
  users: User[];
  history: WatchHistory[];
}

export const ChurnAnalysis: React.FC<ChurnAnalysisProps> = ({ users, history }) => {
  const [selectedUserId, setSelectedUserId] = useState<string>('');
  const [prediction, setPrediction] = useState<ChurnPrediction | null>(null);
  const [loading, setLoading] = useState(false);

  // Filter high-risk users just for the list view (simple heuristic for demo)
  const atRiskUsers = users.filter(u => u.sessions_per_week < 3 && u.churn === 0).slice(0, 10);

  const handleAnalyze = async (userId: string) => {
    setLoading(true);
    const user = users.find(u => u.user_id === userId);
    const userHistory = history.filter(h => h.user_id === userId);
    
    if (user) {
      const result = await analyzeChurnRisk(user, userHistory);
      setPrediction(result);
    }
    setLoading(false);
  };

  return (
    <div className="p-8 h-full overflow-y-auto">
      <div className="mb-8">
        <h2 className="text-3xl font-bold text-white mb-2">Churn Prediction Model</h2>
        <p className="text-gray-400">AI-driven risk assessment and retention strategies</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* User Selection List */}
        <div className="bg-gray-800/50 border border-gray-700 rounded-2xl overflow-hidden flex flex-col h-[600px]">
          <div className="p-4 border-b border-gray-700 bg-gray-800/80">
            <h3 className="font-semibold text-white flex items-center">
              <AlertTriangle className="text-yellow-500 mr-2" size={18} />
              At-Risk Users (Detected)
            </h3>
          </div>
          <div className="overflow-y-auto flex-1 p-2 space-y-2">
            {atRiskUsers.map(user => (
              <button
                key={user.user_id}
                onClick={() => {
                  setSelectedUserId(user.user_id);
                  setPrediction(null);
                  handleAnalyze(user.user_id);
                }}
                className={`w-full text-left p-4 rounded-xl transition-all border ${
                  selectedUserId === user.user_id 
                    ? 'bg-purple-600/20 border-purple-500/50' 
                    : 'bg-gray-800 border-transparent hover:bg-gray-700'
                }`}
              >
                <div className="flex justify-between items-center mb-1">
                  <span className="font-mono text-sm text-gray-300">{user.user_id}</span>
                  <span className="text-xs bg-gray-700 px-2 py-1 rounded text-gray-300">{user.country}</span>
                </div>
                <div className="text-xs text-gray-400 flex justify-between">
                    <span>{user.sessions_per_week} sessions/wk</span>
                    <span>Plan: {user.subscription_plan}</span>
                </div>
              </button>
            ))}
          </div>
        </div>

        {/* Prediction Result Area */}
        <div className="lg:col-span-2 space-y-6">
          {!selectedUserId && (
             <div className="h-full flex flex-col items-center justify-center text-gray-500 border-2 border-dashed border-gray-700 rounded-2xl p-12">
                <Search size={48} className="mb-4 opacity-50" />
                <p>Select a user to run the churn prediction model.</p>
             </div>
          )}

          {loading && (
             <div className="h-full flex flex-col items-center justify-center text-purple-400 border border-gray-700 bg-gray-800/30 rounded-2xl p-12 animate-pulse">
                <RefreshCcw size={48} className="mb-4 animate-spin" />
                <p>Running Gradient Boosting & Gemini Analysis...</p>
             </div>
          )}

          {!loading && prediction && selectedUserId && (
            <div className="space-y-6 animate-fade-in">
              {/* Risk Score Card */}
              <div className="bg-gray-800 border border-gray-700 rounded-2xl p-8 relative overflow-hidden">
                <div className={`absolute top-0 right-0 w-32 h-32 blur-3xl rounded-full opacity-20 pointer-events-none ${
                    prediction.riskLevel === 'High' ? 'bg-red-500' : prediction.riskLevel === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'
                }`}></div>
                
                <div className="relative z-10">
                   <div className="flex justify-between items-start mb-6">
                      <div>
                        <h3 className="text-gray-400 font-medium mb-1">Churn Probability</h3>
                        <div className="flex items-baseline space-x-2">
                            <span className={`text-5xl font-bold ${
                                prediction.riskLevel === 'High' ? 'text-red-500' : prediction.riskLevel === 'Medium' ? 'text-yellow-500' : 'text-green-500'
                            }`}>
                                {prediction.probability}%
                            </span>
                            <span className="text-xl text-gray-500 font-medium">{prediction.riskLevel} Risk</span>
                        </div>
                      </div>
                      <div className={`px-4 py-2 rounded-lg font-bold text-sm ${
                           prediction.riskLevel === 'High' ? 'bg-red-500/10 text-red-500' : 'bg-green-500/10 text-green-500'
                      }`}>
                          Model: Gemini Flash
                      </div>
                   </div>
                   
                   <div className="w-full bg-gray-700 h-4 rounded-full overflow-hidden">
                      <div 
                        className={`h-full transition-all duration-1000 ease-out ${
                            prediction.riskLevel === 'High' ? 'bg-red-500' : prediction.riskLevel === 'Medium' ? 'bg-yellow-500' : 'bg-green-500'
                        }`} 
                        style={{ width: `${prediction.probability}%` }}
                      ></div>
                   </div>
                </div>
              </div>

              {/* Factors & Strategy */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="bg-gray-800 border border-gray-700 rounded-2xl p-6">
                    <h4 className="text-lg font-semibold text-white mb-4">Risk Factors</h4>
                    <ul className="space-y-3">
                        {prediction.factors.map((factor, i) => (
                            <li key={i} className="flex items-start text-gray-300">
                                <span className="mr-2 text-red-400">•</span>
                                {factor}
                            </li>
                        ))}
                    </ul>
                </div>
                <div className="bg-gray-800 border border-gray-700 rounded-2xl p-6">
                    <h4 className="text-lg font-semibold text-white mb-4">Retention Strategy</h4>
                    <div className="bg-purple-900/20 border border-purple-500/30 p-4 rounded-xl">
                        <p className="text-purple-200 leading-relaxed">
                            {prediction.retentionStrategy}
                        </p>
                    </div>
                    <button className="mt-4 w-full bg-white text-gray-900 py-2 rounded-lg font-semibold hover:bg-gray-100 transition-colors">
                        Apply Strategy
                    </button>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
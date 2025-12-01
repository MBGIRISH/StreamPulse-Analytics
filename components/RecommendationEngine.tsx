import React, { useState } from 'react';
import { User, WatchHistory, Title, Recommendation } from '../types';
import { getHybridRecommendations } from '../services/geminiService';
import { Sparkles, Play, ThumbsUp } from 'lucide-react';

interface RecommendationEngineProps {
  users: User[];
  history: WatchHistory[];
  titles: Title[];
}

export const RecommendationEngine: React.FC<RecommendationEngineProps> = ({ users, history, titles }) => {
  const [targetUserId, setTargetUserId] = useState<string>('');
  const [recs, setRecs] = useState<Recommendation[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);

  // Pick random user for demo if none selected, or specific
  const sampleUsers = users.slice(0, 8);

  const generate = async (uid: string) => {
    setTargetUserId(uid);
    setIsGenerating(true);
    const user = users.find(u => u.user_id === uid);
    const userHistory = history.filter(h => h.user_id === uid);
    
    if (user) {
      const results = await getHybridRecommendations(user, userHistory, titles);
      setRecs(results);
    }
    setIsGenerating(false);
  };

  return (
    <div className="p-8 h-full overflow-y-auto">
       <div className="mb-8 max-w-4xl mx-auto text-center">
        <h2 className="text-4xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-purple-400 to-pink-600 mb-4">
            Hybrid Recommendation Engine
        </h2>
        <p className="text-gray-400">
            Combining Collaborative Filtering patterns with Content-Based Analysis using Gemini AI.
        </p>
      </div>

      <div className="max-w-6xl mx-auto space-y-12">
        {/* User Selector */}
        <div>
            <h3 className="text-lg font-semibold text-gray-300 mb-4">Select a User Profile</h3>
            <div className="flex flex-wrap gap-4">
                {sampleUsers.map(u => (
                    <button
                        key={u.user_id}
                        onClick={() => generate(u.user_id)}
                        className={`px-6 py-3 rounded-xl border transition-all duration-300 flex flex-col items-center ${
                            targetUserId === u.user_id 
                            ? 'bg-purple-600 border-purple-400 text-white shadow-lg shadow-purple-900/50 scale-105' 
                            : 'bg-gray-800 border-gray-700 text-gray-400 hover:border-gray-500 hover:text-white'
                        }`}
                    >
                        <span className="font-bold">{u.user_id}</span>
                        <span className="text-xs opacity-70">{u.preferred_genre} Fan</span>
                    </button>
                ))}
            </div>
        </div>

        {/* Loading State */}
        {isGenerating && (
            <div className="py-20 flex flex-col items-center justify-center space-y-4 animate-pulse">
                <div className="relative">
                    <div className="w-16 h-16 border-4 border-purple-500/30 border-t-purple-500 rounded-full animate-spin"></div>
                    <Sparkles className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 text-white" size={24} />
                </div>
                <p className="text-gray-400 text-lg">Analyzing watch patterns & content metadata...</p>
            </div>
        )}

        {/* Results */}
        {!isGenerating && recs.length > 0 && (
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 animate-fade-in-up">
                {recs.map((rec, idx) => (
                    <div key={idx} className="group relative bg-gray-900 rounded-2xl overflow-hidden border border-gray-800 hover:border-purple-500/50 transition-all duration-300 hover:shadow-2xl hover:shadow-purple-900/20">
                        {/* Placeholder Poster */}
                        <div className="h-48 bg-gray-800 w-full relative overflow-hidden">
                             <img 
                                src={`https://picsum.photos/400/250?random=${idx}`} 
                                alt="Movie Poster" 
                                className="w-full h-full object-cover opacity-60 group-hover:opacity-100 transition-opacity"
                            />
                            <div className="absolute top-2 right-2 bg-black/60 backdrop-blur-md px-3 py-1 rounded-full text-xs font-bold text-green-400 border border-green-500/30">
                                {rec.match_score}% Match
                            </div>
                        </div>
                        
                        <div className="p-6">
                            <h3 className="text-xl font-bold text-white mb-2 group-hover:text-purple-400 transition-colors">
                                {rec.title_name}
                            </h3>
                            <p className="text-sm text-gray-400 mb-4 line-clamp-3">
                                {rec.reason}
                            </p>
                            
                            <div className="flex gap-2">
                                <button className="flex-1 bg-white text-black py-2 rounded-lg font-bold flex items-center justify-center hover:bg-gray-200 transition-colors">
                                    <Play size={16} className="mr-2" /> Watch
                                </button>
                                <button className="p-2 border border-gray-700 rounded-lg text-gray-400 hover:text-white hover:border-white transition-colors">
                                    <ThumbsUp size={18} />
                                </button>
                            </div>
                        </div>
                    </div>
                ))}
            </div>
        )}
      </div>
    </div>
  );
};
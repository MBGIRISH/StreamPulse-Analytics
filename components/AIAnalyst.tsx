import React, { useState } from 'react';
import { User, Title } from '../types';
import { getGeneralInsights } from '../services/geminiService';
import { Send, Bot, User as UserIcon } from 'lucide-react';

interface AIAnalystProps {
  users: User[];
  titles: Title[];
}

interface Message {
  role: 'user' | 'ai';
  content: string;
}

export const AIAnalyst: React.FC<AIAnalystProps> = ({ users, titles }) => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    { role: 'ai', content: 'Hello! I am your OTT Data Analyst. Ask me about user trends, popular content, or revenue projections.' }
  ]);
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMsg = input;
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMsg }]);
    setLoading(true);

    // Prepare context summary
    const summary = `
      Total Users: ${users.length}.
      Active Genres: ${[...new Set(titles.map(t => t.genre))].join(', ')}.
      Average User Age: ${Math.round(users.reduce((acc, u) => acc + u.age, 0) / users.length)}.
      Platform Regions: ${[...new Set(users.map(u => u.country))].join(', ')}.
    `;

    const response = await getGeneralInsights(userMsg, summary);
    setMessages(prev => [...prev, { role: 'ai', content: response }]);
    setLoading(false);
  };

  return (
    <div className="flex flex-col h-full bg-gray-900">
      <div className="p-6 border-b border-gray-800">
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Bot className="text-purple-500" />
            AI Data Analyst
        </h2>
        <p className="text-gray-400 text-sm">Powered by Gemini 2.5 Flash</p>
      </div>

      <div className="flex-1 overflow-y-auto p-6 space-y-6">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-2xl p-4 ${
              m.role === 'user' 
                ? 'bg-purple-600 text-white rounded-br-none' 
                : 'bg-gray-800 text-gray-200 border border-gray-700 rounded-bl-none'
            }`}>
              {m.role === 'ai' && <div className="text-xs text-gray-500 mb-1 font-bold">GEMINI ANALYST</div>}
              <div className="markdown-body text-sm leading-relaxed whitespace-pre-wrap">
                {m.content}
              </div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="flex justify-start">
             <div className="bg-gray-800 rounded-2xl p-4 border border-gray-700 flex items-center space-x-2">
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-75"></div>
                <div className="w-2 h-2 bg-gray-500 rounded-full animate-bounce delay-150"></div>
             </div>
          </div>
        )}
      </div>

      <div className="p-6 bg-gray-800/50 border-t border-gray-800">
        <div className="relative">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && handleSend()}
            placeholder="Ask a question about your data..."
            className="w-full bg-gray-900 border border-gray-700 text-white rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
          />
          <button 
            onClick={handleSend}
            disabled={loading || !input.trim()}
            className="absolute right-2 top-2 p-1.5 bg-purple-600 rounded-lg text-white hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <Send size={18} />
          </button>
        </div>
      </div>
    </div>
  );
};
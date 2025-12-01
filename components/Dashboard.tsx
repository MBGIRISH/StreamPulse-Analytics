import React from 'react';
import { 
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  LineChart, Line, PieChart, Pie, Cell 
} from 'recharts';
import { User, WatchHistory, Title } from '../types';
import { getAggregatedGenreData, getEngagementOverTime } from '../services/dataService';
import { ArrowUpRight, Users, Clock, PlayCircle } from 'lucide-react';

interface DashboardProps {
  users: User[];
  history: WatchHistory[];
  titles: Title[];
}

const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff8042', '#0088fe', '#00C49F'];

export const Dashboard: React.FC<DashboardProps> = ({ users, history, titles }) => {
  const genreData = getAggregatedGenreData(history, titles);
  const engagementData = getEngagementOverTime(history);

  const totalWatchTime = history.reduce((acc, curr) => acc + curr.watch_time_minutes, 0);
  const activeUsers = users.filter(u => new Date(u.last_active_date) > new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)).length;
  const churnRate = ((users.filter(u => u.churn === 1).length / users.length) * 100).toFixed(1);

  const StatCard = ({ title, value, sub, icon: Icon, color }: any) => (
    <div className="bg-gray-800/50 border border-gray-700 p-6 rounded-2xl backdrop-blur-sm hover:border-gray-600 transition-all">
      <div className="flex justify-between items-start mb-4">
        <div>
          <p className="text-gray-400 text-sm font-medium mb-1">{title}</p>
          <h3 className="text-2xl font-bold text-white">{value}</h3>
        </div>
        <div className={`p-2 rounded-lg ${color}`}>
          <Icon size={20} className="text-white" />
        </div>
      </div>
      <div className="flex items-center text-xs text-green-400 font-medium">
        <ArrowUpRight size={14} className="mr-1" />
        {sub}
      </div>
    </div>
  );

  return (
    <div className="p-8 space-y-8 animate-fade-in">
      <div className="flex justify-between items-end">
        <div>
          <h2 className="text-3xl font-bold text-white mb-2">Platform Overview</h2>
          <p className="text-gray-400">Real-time metrics from the OTT ecosystem</p>
        </div>
        <div className="text-right">
          <p className="text-sm text-gray-500">Last updated</p>
          <p className="text-white font-mono">{new Date().toLocaleTimeString()}</p>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard 
          title="Total Users" 
          value={users.length.toLocaleString()} 
          sub="+12.5% vs last month" 
          icon={Users} 
          color="bg-blue-600" 
        />
        <StatCard 
          title="Avg Watch Time" 
          value={`${Math.round(totalWatchTime / users.length)} mins`} 
          sub="+5.2% vs last month" 
          icon={Clock} 
          color="bg-purple-600" 
        />
        <StatCard 
          title="Churn Rate" 
          value={`${churnRate}%`} 
          sub="-1.8% vs last month" 
          icon={ArrowUpRight} 
          color="bg-red-500" // Red usually bad, but context is churn
        />
        <StatCard 
          title="Content Library" 
          value={titles.length} 
          sub="+24 new titles this week" 
          icon={PlayCircle} 
          color="bg-indigo-600" 
        />
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-gray-800/50 border border-gray-700 p-6 rounded-2xl">
          <h3 className="text-lg font-semibold text-white mb-6">Engagement Trends (Last 14 Days)</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={engagementData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                <XAxis dataKey="date" stroke="#9ca3af" tick={{fontSize: 12}} tickFormatter={(val) => val.slice(5)} />
                <YAxis stroke="#9ca3af" tick={{fontSize: 12}} />
                <Tooltip 
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Line type="monotone" dataKey="minutes" stroke="#8b5cf6" strokeWidth={3} dot={{r: 4, fill: '#8b5cf6'}} activeDot={{r: 6}} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="bg-gray-800/50 border border-gray-700 p-6 rounded-2xl">
          <h3 className="text-lg font-semibold text-white mb-6">Genre Popularity</h3>
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={genreData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {genreData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip 
                   contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                />
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-wrap gap-2 justify-center mt-4">
              {genreData.slice(0, 4).map((g, i) => (
                 <div key={i} className="flex items-center text-xs text-gray-400">
                    <span className="w-2 h-2 rounded-full mr-1" style={{ backgroundColor: COLORS[i] }}></span>
                    {g.name}
                 </div>
              ))}
            </div>
          </div>
        </div>
      </div>
      
       {/* Charts Row 2 */}
       <div className="grid grid-cols-1 gap-8">
         <div className="bg-gray-800/50 border border-gray-700 p-6 rounded-2xl">
           <h3 className="text-lg font-semibold text-white mb-6">Device Usage Distribution</h3>
           <div className="h-64">
             <ResponsiveContainer width="100%" height="100%">
                <BarChart data={[
                  { name: 'Mobile', value: users.filter(u => u.device_type === 'Mobile').length },
                  { name: 'TV', value: users.filter(u => u.device_type === 'TV').length },
                  { name: 'Laptop', value: users.filter(u => u.device_type === 'Laptop').length },
                  { name: 'Tablet', value: users.filter(u => u.device_type === 'Tablet').length },
                ]}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
                  <XAxis dataKey="name" stroke="#9ca3af" />
                  <YAxis stroke="#9ca3af" />
                  <Tooltip cursor={{fill: '#374151', opacity: 0.4}} contentStyle={{ backgroundColor: '#1f2937', border: 'none', color: '#fff' }}/>
                  <Bar dataKey="value" fill="#3b82f6" radius={[4, 4, 0, 0]} barSize={50} />
                </BarChart>
             </ResponsiveContainer>
           </div>
         </div>
       </div>
    </div>
  );
};
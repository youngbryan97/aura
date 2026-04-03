import React, { startTransition, useEffect, useState } from 'react';
import StatCard from './components/StatCard';
import ShardNode from './components/ShardNode';
import Singularity from './components/Singularity';
import { fetchMemoryStats, getLocalState, setLocalState } from './utils/apiProxy';

const DEFAULT_STATS = {
    status: 'offline',
    horcrux: 'UNKNOWN',
    shards_active: 0,
    total_nodes: 0,
    total_mass_kb: 0,
    bit_density: 0,
    radius_cm: 10,
    evaporation_count: 0,
    entropy_fidelity: 0,
    recent_events: [],
};

const normalizeStats = (payload) => ({
    ...DEFAULT_STATS,
    ...(payload || {}),
    recent_events: Array.isArray(payload?.recent_events) ? payload.recent_events : [],
});

function App() {
    const cachedStats = getLocalState('last_stats', null);
    const [stats, setStats] = useState(() => normalizeStats(cachedStats));
    const [loading, setLoading] = useState(!cachedStats);
    const [syncState, setSyncState] = useState(cachedStats ? 'stale' : 'loading');
    const [error, setError] = useState('');

    useEffect(() => {
        let active = true;
        let timerId;

        const fetchStats = async () => {
            try {
                const data = normalizeStats(await fetchMemoryStats());
                if (!active) {
                    return;
                }
                startTransition(() => {
                    setStats(data);
                });
                setLocalState('last_stats', data);
                setSyncState('live');
                setError('');
            } catch (e) {
                if (!active) {
                    return;
                }
                console.error("Dashboard link severed:", e);
                setSyncState(cachedStats ? 'stale' : 'offline');
                setError(e instanceof Error ? e.message : 'Unable to reach memory dashboard');
            } finally {
                if (!active) {
                    return;
                }
                setLoading(false);
                timerId = window.setTimeout(fetchStats, 5000);
            }
        };

        fetchStats();
        return () => {
            active = false;
            if (timerId) {
                window.clearTimeout(timerId);
            }
        };
    }, [cachedStats]);

    if (loading) return (
        <div className="h-screen w-screen flex flex-col items-center justify-center bg-black">
            <div className="w-16 h-16 border-4 border-purple-500/20 border-t-purple-500 rounded-full animate-spin mb-4"></div>
            <span className="mono text-purple-400 animate-pulse tracking-widest text-xs">RECONSTRUCTING EVENT HORIZON...</span>
        </div>
    );

    const statusPillClass = syncState === 'live'
        ? 'border-emerald-400/40 text-emerald-300'
        : syncState === 'stale'
            ? 'border-amber-400/40 text-amber-200'
            : 'border-rose-400/40 text-rose-200';

    return (
        <div className="h-screen w-screen flex flex-col p-6 overflow-hidden">
            {/* Background and Overlay elements handled via index.css/index.html */}
            
            {/* Top Header */}
            <div className="flex justify-between items-start mb-8 z-10">
                <div className="flex flex-col">
                    <h1 className="text-2xl font-bold tracking-tighter flex items-center gap-2">
                        <span className="text-purple-500">AURA</span>
                        <span className="text-gray-400 font-light">| BLACK HOLE MEMORY</span>
                    </h1>
                    <span className="text-[10px] mono text-purple-400 tracking-[0.3em] uppercase opacity-70">Sovereign Encryption Node v6.3</span>
                </div>
                <div className="flex items-center gap-6 glass px-6 py-3 rounded-2xl border-purple-500/20">
                    <div className={`rounded-full border px-3 py-1 mono text-[9px] uppercase tracking-[0.25em] ${statusPillClass}`}>
                        {syncState === 'live' ? 'Live Sync' : syncState === 'stale' ? 'Cached Sync' : 'Offline'}
                    </div>
                    <div className="flex flex-col items-end">
                        <span className="text-[9px] uppercase tracking-widest text-gray-500 mb-1">Horcrux Integrity</span>
                        <div className="flex gap-1.5">
                            {[...Array(5)].map((_, i) => (
                                <ShardNode key={i} active={i < stats.shards_active} />
                            ))}
                        </div>
                    </div>
                    <div className="h-8 w-px bg-gray-800"></div>
                    <div className="flex flex-col">
                        <span className="text-xs font-bold text-green-400 tracking-tighter uppercase">{stats.horcrux}</span>
                        <span className="text-[9px] text-gray-500 uppercase font-bold tracking-widest">System Lock</span>
                    </div>
                </div>
            </div>

            {/* Main Content Area */}
            <div className="flex-1 grid grid-cols-12 gap-6 min-h-0">
                {/* Left: Physics & Metrics */}
                <div className="col-span-3 flex flex-col gap-4 overflow-y-auto pr-2 no-scrollbar">
                    {error && (
                        <div className="glass rounded-2xl border border-amber-400/20 px-4 py-3 text-[11px] leading-relaxed text-amber-100">
                            <div className="mono text-[9px] uppercase tracking-[0.3em] text-amber-300 mb-2">
                                Memory Link Status
                            </div>
                            {syncState === 'stale'
                                ? `Showing cached dashboard data while live sync is unavailable. ${error}`
                                : `Live memory telemetry is unavailable. ${error}`}
                        </div>
                    )}
                    <StatCard label="Total Nodes" value={stats.total_nodes} unit="Entities" color="purple" />
                    <StatCard label="Vault Mass" value={stats.total_mass_kb} unit="KB" color="blue" />
                    <StatCard label="Bit Density" value={stats.bit_density} unit="b/cm²" color="pink" />
                    <StatCard label="Hawking Decay" value={stats.evaporation_count} unit="Leaking" color="orange" />
                    
                    <div className="glass mt-auto p-5 rounded-2xl relative overflow-hidden">
                        <div className="absolute top-0 left-0 w-full h-1 bg-gradient-to-r from-purple-500 to-transparent"></div>
                        <h3 className="text-[10px] uppercase font-bold tracking-widest text-gray-500 mb-4">Physics Boundaries</h3>
                        <div className="space-y-4 mono text-[11px]">
                            <div className="flex justify-between">
                                <span className="text-gray-400">Radius</span>
                                <span className="text-purple-400">{stats.radius_cm}cm</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Entropy σ</span>
                                <span className="text-blue-400">{stats.entropy_fidelity}</span>
                            </div>
                            <div className="flex justify-between">
                                <span className="text-gray-400">Spin ω</span>
                                <span className="text-pink-400">0.82c</span>
                            </div>
                        </div>
                    </div>
                </div>

                {/* Center: Singularity Visualization */}
                <div className="col-span-6 relative flex items-center justify-center">
                    <div className="absolute inset-x-0 bottom-12 flex flex-col items-center">
                        <div className="text-center mb-8">
                            <div className="mono text-gray-500 text-[10px] uppercase tracking-[0.5em] mb-2 leading-none">Event Horizon Equilibrium</div>
                            <div className="text-4xl font-bold tracking-tighter leading-none glitch-text">STABLE SINGULARITY</div>
                        </div>
                    </div>
                    
                    <Singularity />
                </div>

                {/* Right: Spaghettified Streams */}
                <div className="col-span-3 flex flex-col min-h-0 glass rounded-3xl p-6 border-purple-500/10">
                    <h3 className="text-[10px] uppercase font-bold tracking-widest text-purple-400 mb-6 flex items-center gap-2">
                        <div className="w-1.5 h-1.5 bg-purple-500 rounded-full animate-pulse"></div>
                        Neural Stream (Recent Recall)
                    </h3>
                    <div className="flex-1 overflow-y-auto space-y-4 pr-2 custom-scrollbar">
                        {(stats.recent_events || []).map((ev, i) => (
                            <div key={i} className="group relative">
                                <div className="absolute -left-2 top-0 h-full w-0.5 bg-gray-800 group-hover:bg-purple-500 transition-all"></div>
                                <div className="flex flex-col gap-1">
                                    <div className="flex justify-between items-center text-[9px] mono text-gray-500 mb-0.5">
                                        <span>T-{ev.age_seconds}s</span>
                                        <span className="text-purple-500/60 font-bold">G: {ev.gravity}</span>
                                    </div>
                                    <p className="text-[11px] leading-relaxed text-gray-300 font-light group-hover:text-white transition-colors tracking-tight">
                                        {ev.text}
                                    </p>
                                </div>
                            </div>
                        ))}
                        {(!stats.recent_events || stats.recent_events.length === 0) && (
                            <div className="h-full flex items-center justify-center italic text-gray-600 mono text-xs">
                                Singularity is void...
                            </div>
                        )}
                    </div>
                    <div className="mt-4 pt-4 border-t border-white/5 text-[9px] mono text-gray-600 flex justify-between uppercase">
                        <span>Buffer: 1024mb</span>
                        <span className="text-purple-700 font-bold tracking-tighter leading-none">Scrambler Active</span>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default App;

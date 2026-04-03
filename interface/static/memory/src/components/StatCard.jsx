import React from 'react';

const COLOR_STYLES = {
    purple: {
        border: 'border-purple-500',
        unit: 'text-purple-400',
    },
    blue: {
        border: 'border-blue-500',
        unit: 'text-blue-400',
    },
    pink: {
        border: 'border-pink-500',
        unit: 'text-pink-400',
    },
    orange: {
        border: 'border-orange-500',
        unit: 'text-orange-400',
    },
};

const StatCard = ({ label, value, unit, color }) => {
    const styles = COLOR_STYLES[color] || COLOR_STYLES.purple;

    return (
    <div className={`glass p-4 rounded-xl border-l-4 ${styles.border} flex flex-col justify-center`}>
        <span className="text-[10px] uppercase tracking-[0.2em] text-gray-400 mb-1 font-bold">{label}</span>
        <div className="flex items-baseline gap-1">
            <span className="text-2xl font-bold mono leading-none tracking-tight">{value}</span>
            <span className={`text-[10px] ${styles.unit} font-bold uppercase`}>{unit}</span>
        </div>
    </div>
    );
};

export default StatCard;

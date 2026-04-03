import React from 'react';

const Singularity = () => (
    <div className="relative w-64 h-64 flex items-center justify-center">
        <div className="absolute inset-0 singularity-glow rounded-full blur-2xl opacity-40"></div>
        <div className="w-56 h-56 rounded-full border border-purple-500/30 animate-[spin_10s_linear_infinite] p-4">
            <div className="w-full h-full rounded-full border border-blue-500/20 animate-[spin_15s_linear_infinite] flex items-center justify-center">
                <div className="w-4 h-4 bg-white rounded-full shadow-[0_0_30px_#fff] blur-[1px]"></div>
            </div>
        </div>
        {/* Dark Center */}
        <div className="absolute w-40 h-40 bg-black rounded-full shadow-[inset_0_0_40px_#a855f7] border border-white/5"></div>
    </div>
);

export default Singularity;

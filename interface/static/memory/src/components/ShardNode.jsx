import React from 'react';

const ShardNode = ({ active }) => (
    <div className={`w-3 h-3 rounded-full transition-all duration-1000 ${active ? 'bg-purple-500 shadow-[0_0_10px_#a855f7]' : 'bg-gray-800'}`}></div>
);

export default ShardNode;

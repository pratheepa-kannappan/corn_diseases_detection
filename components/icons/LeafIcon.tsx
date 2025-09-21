import React from 'react';

export const LeafIcon: React.FC<{ className?: string }> = ({ className }) => (
  <svg
    xmlns="http://www.w3.org/2000/svg"
    className={className}
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <path d="M11 20A7 7 0 0 1 4 13V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
    <path d="M11 4A7 7 0 0 1 18 11v1a2 2 0 0 1-2 2h-2a2 2 0 0 1-2-2v-1" />
    <line x1="11" y1="4" x2="11" y2="20" />
  </svg>
);

import React from 'react';
import type { DiseaseInfo } from '../types';
import { LeafIcon } from './icons/LeafIcon';

interface ResultCardProps {
  result: DiseaseInfo;
}

const InfoSection: React.FC<{ title: string; items: string[] }> = ({ title, items }) => (
  <div>
    <h3 className="text-lg font-semibold text-gray-700 mb-2">{title}</h3>
    <ul className="list-disc list-inside space-y-1 text-gray-600">
      {items.map((item, index) => (
        <li key={index}>{item}</li>
      ))}
    </ul>
  </div>
);


export const ResultCard: React.FC<ResultCardProps> = ({ result }) => {
  const isHealthy = result.isHealthy;
  const cardColor = isHealthy ? 'border-green-400' : 'border-red-400';
  const titleColor = isHealthy ? 'text-green-600' : 'text-red-600';

  return (
    <div className={`bg-white/80 backdrop-blur-sm rounded-2xl shadow-lg p-6 md:p-8 border-t-4 ${cardColor} w-full max-w-4xl`}>
      <div className="flex flex-col md:flex-row items-start md:items-center gap-4 mb-6">
        <LeafIcon className={`w-12 h-12 ${titleColor}`} />
        <div>
          <p className="text-sm font-medium text-gray-500">Diagnosis Result</p>
          <h2 className={`text-3xl font-bold ${titleColor}`}>{result.diseaseName}</h2>
        </div>
      </div>
      
      <p className="text-gray-700 mb-8 text-lg">{result.description}</p>
      
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-8">
        {!isHealthy && result.causes?.length > 0 && <InfoSection title="Possible Causes" items={result.causes} />}
        {!isHealthy && result.treatment?.length > 0 && <InfoSection title="Treatment" items={result.treatment} />}
        {!isHealthy && result.prevention?.length > 0 && <InfoSection title="Prevention" items={result.prevention} />}
        {isHealthy && result.prevention?.length > 0 && <InfoSection title="Care Tips" items={result.prevention} />}
      </div>
    </div>
  );
};

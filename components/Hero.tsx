import React from 'react';

interface HeroProps {
  onDiagnoseClick: () => void;
}

export const Hero: React.FC<HeroProps> = ({ onDiagnoseClick }) => {
  return (
    <section className="relative w-full min-h-screen flex items-center justify-center px-4 overflow-hidden">
      {/* Background shapes */}
      <div className="absolute top-[-10%] left-[-15%] w-72 h-72 bg-emerald-100/50 rounded-full filter blur-xl opacity-70"></div>
      <div className="absolute bottom-[-5%] right-[-10%] w-96 h-96 bg-green-100/40 rounded-full filter blur-2xl opacity-60"></div>
      <div className="absolute top-1/2 left-1/4 w-48 h-48 bg-teal-100/30 rounded-full filter blur-xl opacity-50 animate-pulse"></div>

      <div className="container mx-auto flex flex-col lg:flex-row items-center justify-between gap-12 z-10">
        <div className="w-full lg:w-1/2 text-center lg:text-left">
          <h1 className="text-4xl md:text-6xl font-bold text-emerald-800 leading-tight mb-4">
            Identify and Cure
            <br />
            Corn Plant Diseases
          </h1>
          <p className="text-lg md:text-xl text-gray-600 mb-8 max-w-xl mx-auto lg:mx-0">
            Is your corn crop looking sick? Try our AI-powered tool to identify the cause and get extensive disease and care info in a snap.
          </p>
          <button
            onClick={onDiagnoseClick}
            className="px-8 py-4 bg-gradient-to-r from-orange-400 to-yellow-500 text-white text-lg font-bold rounded-full shadow-lg hover:scale-105 transition-transform duration-300"
          >
            Diagnose Now
          </button>
        </div>
        <div className="w-full lg:w-1/2 flex justify-center lg:justify-end mt-12 lg:mt-0">
          <img
            src="https://picsum.photos/id/375/600/800"
            alt="Healthy corn field"
            className="rounded-3xl shadow-2xl w-full max-w-sm lg:max-w-md object-cover h-[500px] lg:h-[600px] border-8 border-white/50 transform lg:rotate-3 transition-transform duration-500 hover:rotate-0"
          />
        </div>
      </div>
    </section>
  );
};

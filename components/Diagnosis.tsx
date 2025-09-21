import React, { useState, useRef } from 'react';
import { diagnoseCornPlant } from '../services/geminiService';
import type { DiseaseInfo } from '../types';
import { ResultCard } from './ResultCard';
import { UploadIcon } from './icons/UploadIcon';

export const Diagnosis: React.FC<{}> = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [result, setResult] = useState<DiseaseInfo | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      if (file.type.startsWith('image/')) {
        setImageFile(file);
        setPreviewUrl(URL.createObjectURL(file));
        setResult(null);
        setError(null);
      } else {
        setError("Please upload a valid image file (e.g., JPG, PNG).");
        setImageFile(null);
        setPreviewUrl(null);
      }
    }
  };

  const handleDiagnose = async () => {
    if (!imageFile) {
      setError("Please select an image first.");
      return;
    }
    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const diagnosisResult = await diagnoseCornPlant(imageFile);
      if(diagnosisResult.error) {
          setError(diagnosisResult.error);
      } else {
          setResult(diagnosisResult);
      }
    } catch (e: any) {
      setError(e.message || "An unexpected error occurred.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleBoxClick = () => {
    fileInputRef.current?.click();
  };

  const resetState = () => {
    setImageFile(null);
    setPreviewUrl(null);
    setIsLoading(false);
    setResult(null);
    setError(null);
    if(fileInputRef.current) {
        fileInputRef.current.value = "";
    }
  };

  return (
    <section id="diagnose" className="w-full max-w-5xl mx-auto py-20 px-4 flex flex-col items-center">
      <h2 className="text-3xl md:text-4xl font-bold text-center text-emerald-800 mb-4">
        Your Personal Plant Doctor
      </h2>
      <p className="text-lg text-gray-600 text-center mb-12 max-w-3xl">
        Simply snap a photo of the issue to get a diagnosis. Our AI will give you detailed info on the disease, what caused it, how to treat it, and how to prevent it.
      </p>

      <div className="w-full flex flex-col items-center gap-8">
        {!previewUrl && (
          <div
            onClick={handleBoxClick}
            className="w-full max-w-md h-72 border-2 border-dashed border-gray-400 rounded-2xl flex flex-col justify-center items-center text-gray-500 hover:border-emerald-500 hover:text-emerald-600 transition-colors duration-300 cursor-pointer bg-green-50/50"
          >
            <UploadIcon className="w-16 h-16 mb-4" />
            <span className="text-xl font-medium">Click to upload an image</span>
            <span className="mt-1">or drag and drop</span>
          </div>
        )}

        {previewUrl && (
          <div className="w-full max-w-md p-4 bg-white rounded-2xl shadow-lg">
            <img src={previewUrl} alt="Corn leaf preview" className="w-full h-auto object-contain rounded-xl max-h-80" />
          </div>
        )}
        
        <input
          type="file"
          ref={fileInputRef}
          onChange={handleFileChange}
          className="hidden"
          accept="image/*"
        />

        {imageFile && (
             <div className="flex items-center gap-4 mt-4">
                <button
                    onClick={handleDiagnose}
                    disabled={isLoading}
                    className="px-8 py-3 bg-gradient-to-r from-orange-400 to-yellow-500 text-white font-bold rounded-full shadow-lg hover:scale-105 transition-transform duration-300 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                    {isLoading ? (
                        <>
                            <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                            </svg>
                            Analyzing...
                        </>
                    ) : 'Diagnose Now'}
                </button>
                 <button onClick={resetState} className="px-6 py-3 bg-gray-200 text-gray-700 font-semibold rounded-full hover:bg-gray-300 transition-colors duration-300">
                    Clear
                </button>
            </div>
        )}
        

        {error && <div className="mt-4 text-red-600 bg-red-100 p-4 rounded-lg w-full max-w-md text-center">{error}</div>}
        
        {result && <div className="mt-8 w-full"><ResultCard result={result} /></div>}
      </div>
    </section>
  );
};

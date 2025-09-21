import { GoogleGenAI, Type } from "@google/genai";
import type { DiseaseInfo } from '../types';

const fileToGenerativePart = async (file: File) => {
  const base64EncodedDataPromise = new Promise<string>((resolve) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      if (typeof reader.result === 'string') {
        resolve(reader.result.split(',')[1]);
      }
    };
    reader.readAsDataURL(file);
  });
  return {
    inlineData: { data: await base64EncodedDataPromise, mimeType: file.type },
  };
};


export const diagnoseCornPlant = async (imageFile: File): Promise<DiseaseInfo> => {
    if (!process.env.API_KEY) {
        throw new Error("API_KEY environment variable is not set");
    }
    const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });
    const imagePart = await fileToGenerativePart(imageFile);

    const prompt = `You are an expert agricultural botanist specializing in corn plant diseases. Analyze the provided image of a corn leaf. Identify if the plant is healthy or suffering from one of the following diseases: Common Rust, Northern Corn Leaf Blight, or Gray Leaf Spot.

    Provide your diagnosis. If the image is not a corn leaf or the quality is too poor to make a diagnosis, respond with an error message within the JSON structure.`;

    try {
        const response = await ai.models.generateContent({
            model: "gemini-2.5-flash",
            contents: {
                parts: [
                    imagePart,
                    { text: prompt }
                ]
            },
            config: {
                responseMimeType: "application/json",
                responseSchema: {
                    type: Type.OBJECT,
                    properties: {
                        diseaseName: {
                            type: Type.STRING,
                            description: "The common name of the identified disease (e.g., 'Common Rust', 'Healthy', 'Northern Corn Leaf Blight', 'Gray Leaf Spot').",
                        },
                        isHealthy: {
                            type: Type.BOOLEAN,
                            description: "True if the plant is identified as healthy, otherwise false.",
                        },
                        description: {
                            type: Type.STRING,
                            description: "A detailed but concise description of the condition.",
                        },
                        causes: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING },
                            description: "A list of common causes for the disease.",
                        },
                        treatment: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING },
                            description: "A list of recommended steps or methods for treatment.",
                        },
                        prevention: {
                            type: Type.ARRAY,
                            items: { type: Type.STRING },
                            description: "A list of measures to prevent this disease in the future.",
                        },
                        error: {
                            type: Type.STRING,
                            description: "An error message if diagnosis is not possible (e.g., 'Unable to diagnose. Please provide a clear image of a corn leaf.').",
                            nullable: true,
                        }
                    },
                    required: ["diseaseName", "isHealthy", "description", "causes", "treatment", "prevention"]
                }
            }
        });

        const jsonText = response.text.trim();
        const parsedResult = JSON.parse(jsonText);
        return parsedResult;

    } catch (e) {
        console.error("Error calling Gemini API:", e);
        throw new Error("Failed to get a diagnosis from the AI. Please try again.");
    }
};

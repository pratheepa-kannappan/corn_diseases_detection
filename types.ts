export interface DiseaseInfo {
  diseaseName: string;
  isHealthy: boolean;
  description: string;
  causes: string[];
  treatment: string[];
  prevention: string[];
  error?: string;
}

import React, { useState, useRef } from "react";
import { motion } from "framer-motion";

interface PredictionResult {
  disease: string;
  confidence: number;
  advice: string;
}

export default function App() {
  const [image, setImage] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const fakePredict = async (file: File | null): Promise<PredictionResult> => {
    await new Promise((r) => setTimeout(r, 900));
    const name = file?.name.toLowerCase() || "";
    if (name.includes("blight") || name.includes("n")) {
      return { disease: "Northern Corn Leaf Blight", confidence: 0.92, advice: "Remove severely infected leaves, rotate crops, consider resistant hybrids." };
    }
    if (name.includes("rust")) {
      return { disease: "Common Rust", confidence: 0.88, advice: "Use fungicides when threshold reached, plant resistant varieties next season." };
    }
    const options: PredictionResult[] = [
      { disease: "Gray Leaf Spot", confidence: 0.79, advice: "Maintain residue management, rotate crops, and apply fungicide if necessary." },
      { disease: "Physiological Leaf Spot (Nutrient deficiency)", confidence: 0.65, advice: "Test soil and adjust fertilization, ensure balanced nutrients." },
      { disease: "Healthy", confidence: 0.97, advice: "No action required. Continue monitoring and good agronomic practices." },
    ];
    const seed = file?.size ? file.size % options.length : Math.floor(Math.random() * options.length);
    return options[seed];
  };

  const handleFile = (f: File | undefined) => {
    setResult(null);
    if (!f) {
      setImage(null);
      setPreviewUrl(null);
      if (fileRef.current) fileRef.current.value = "";
      return;
    }
    setImage(f);
    setPreviewUrl(URL.createObjectURL(f));
  };
  
  const onAnalyze = async () => {
    if (!image) {
      alert("Please upload an image of a corn leaf or plant first.");
      return;
    }
    setLoading(true);
    try {
      const prediction = await fakePredict(image);
      setResult(prediction);
    } catch (e) {
      console.error(e);
      alert("Analysis failed. See console for details.");
    } finally {
      setLoading(false);
    }
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const f = e.dataTransfer.files?.[0];
    if (f) handleFile(f);
  };
  
  const downloadReport = () => {
    if (!result) return;
    const payload = { timestamp: new Date().toISOString(), disease: result.disease, confidence: result.confidence, advice: result.advice };
    const blob = new Blob([JSON.stringify(payload, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `corn-disease-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };
  
  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
      handleFile(e.target.files?.[0]);
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-green-50 via-emerald-50 to-green-100 p-6 flex flex-col items-center font-sans text-gray-700">
      <h1 className="text-4xl font-bold text-center mb-2">
        Corn <span className="text-emerald-700">Disease Detection</span>
      </h1>
      
      <p className="text-center text-xl font-semibold text-emerald-700 mb-6">
        Is your green buddy dying? <br />
        Try to identify the cause and get extensive disease and care info in a snap.
      </p>
      <div className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 bg-white rounded-2xl p-8 shadow-md">
          <div className="flex flex-col lg:flex-row items-start justify-between gap-6">
            <div className="flex-1">
              <p className="mt-2 text-gray-600 max-w-xl">
                Upload a photo of a corn leaf or plant and our model will identify common diseases — with actionable advice for farmers and agronomists.
              </p>

              <div className="mt-6 grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-xl border bg-emerald-50">
                  <h4 className="font-semibold text-emerald-700">Fast Analysis</h4>
                  <p className="text-sm text-gray-600 mt-1">Get a diagnosis in seconds (demo: mock model).</p>
                </div>
                <div className="p-4 rounded-xl border bg-emerald-50">
                  <h4 className="font-semibold text-emerald-700">Field-ready Tips</h4>
                  <p className="text-sm text-gray-600 mt-1">Practical treatment & prevention suggestions for corn growers.</p>
                </div>
              </div>
            </div>
            <img src="https://images.unsplash.com/photo-1501004318641-b39e6451bec6?auto=format&fit=crop&w=400&q=60" alt="corn" className="hidden sm:block w-36 h-36 object-cover rounded-lg" />
            <p className="text-center font-semibold text-emerald-700">
               Personal plant doctor is with you.<br />
               Have you ever searched ‘what’s wrong with my plant’? The results may have been disappointing… No more with Us! Simply snap a photo of the issue to get a diagnosis. We will give you detailed info on the disease, what caused it, how to treat it, and how to prevent it.
             </p>
          </div>

          <div className="mt-6">
            <div onDrop={onDrop} onDragOver={(e) => e.preventDefault()} className="border-2 border-dashed border-gray-200 rounded-xl p-6 flex flex-col sm:flex-row items-center gap-6 bg-emerald-50">
              <div className="flex-1 text-center sm:text-left">
                <p className="font-medium text-emerald-700">Upload an image</p>
                <p className="text-sm text-gray-600 mt-1">Drag & drop or click to select a photo (leaf or ear close-ups work best).</p>

                <div className="mt-4 flex items-center gap-3 justify-center sm:justify-start">
                  <button onClick={() => fileRef.current?.click()} className="px-4 py-2 rounded-lg bg-emerald-600 text-white font-medium shadow">Select photo</button>
                  <button onClick={onAnalyze} disabled={!image || loading} className="px-4 py-2 rounded-lg bg-indigo-600 text-white font-medium shadow disabled:opacity-50">
                    {loading ? "Analyzing..." : "Analyze"}
                  </button>
                </div>
              </div>
              <input type="file" accept="image/*" ref={fileRef} className="hidden" onChange={handleFileChange} />
              <div className="w-40 h-40 bg-gray-100 rounded-lg flex items-center justify-center overflow-hidden">
                {previewUrl ? <img src={previewUrl} alt="preview" className="w-full h-full object-cover" /> : <div className="text-sm text-gray-400">No image selected</div>}
              </div>
            </div>

            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: result ? 1 : 0, y: result ? 0 : 8, transition: { duration: 0.4 } }} className="mt-6">
              {result && (
                <div className="rounded-xl p-6 bg-white border shadow-sm">
                  <h3 className="text-xl font-semibold text-emerald-700">Detection Result</h3>
                  <p className="mt-2 text-gray-700">Disease: <span className="font-medium text-emerald-700">{result.disease}</span></p>
                  <p className="mt-1 text-gray-700">Confidence: <span className="font-medium text-emerald-700">{(result.confidence*100).toFixed(1)}%</span></p>

                  <div className="mt-3">
                    <h4 className="font-semibold text-emerald-800">Recommended actions</h4>
                    <p className="text-sm text-gray-700 mt-1">{result.advice}</p>
                  </div>

                  <div className="mt-4 flex items-center gap-3">
                    <button onClick={downloadReport} className="px-4 py-2 rounded-lg border bg-emerald-50 text-emerald-700">Download report</button>
                    <button onClick={() => { handleFile(undefined); }} className="px-4 py-2 rounded-lg bg-red-50 text-red-600 border">Clear</button>
                  </div>
                </div>
              )}
            </motion.div>

            {!result && (
              <div className="mt-6 rounded-xl p-6 bg-white border text-gray-500">
                No result yet — upload an image and press <strong>Analyze</strong>.
              </div>
            )}
          </div>

          <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="p-4 rounded-xl bg-white border">
              <h5 className="font-semibold text-emerald-800">How it works</h5>
              <p className="text-xs text-gray-600 mt-1">A trained model classifies the leaf image against known corn disease patterns.</p>
            </div>
            <div className="p-4 rounded-xl bg-white border">
              <h5 className="font-semibold text-emerald-800">Model</h5>
              <p className="text-xs text-gray-600 mt-1">Demo uses mock predictions. Swap fakePredict with your inference API endpoint.</p>
            </div>
            <div className="p-4 rounded-xl bg-white border">
              <h5 className="font-semibold text-emerald-800">Best practices</h5>
              <p className="text-xs text-gray-600 mt-1">Capture close-ups, good lighting, multiple angles, and show symptomatic leaves clearly.</p>
            </div>
          </div>
        </div>

        <aside className="bg-white rounded-2xl p-6 shadow-md">
          <h4 className="text-lg font-semibold text-emerald-700">Corn Disease Library</h4>
          <p className="text-sm text-gray-600 mt-2">Tap a disease to see quick ID tips.</p>
          <div className="mt-4 space-y-3">
            {[
              { title: "Northern Corn Leaf Blight", desc: "Long gray-green lesions that turn tan; common in humid regions." },
              { title: "Gray Leaf Spot", desc: "Rectangular lesions following leaf veins; favors warm, wet conditions." },
              { title: "Common Rust", desc: "Orange-brown pustules on both leaf surfaces; easily visible." },
              { title: "Nutrient Deficiency", desc: "Yellowing or spotting not caused by pathogen; soil test recommended." },
            ].map((d) => (
              <div key={d.title} className="p-3 border rounded-lg hover:shadow-sm cursor-pointer bg-emerald-50">
                <h5 className="font-medium text-emerald-700">{d.title}</h5>
                <p className="text-xs text-gray-600 mt-1">{d.desc}</p>
              </div>
            ))}
          </div>
          <div className="mt-6 border-t pt-4">
            <h5 className="font-semibold text-emerald-800">Want field integration?</h5>
            <div className="mt-3">
              <a href="#contact" className="inline-block px-4 py-2 rounded-lg bg-emerald-600 text-white">Contact us</a>
            </div>
          </div>
        </aside>
      </div>
    </div>
  );
}
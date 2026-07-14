import { useState } from "react";

function UploadBox({ onAnalysisComplete }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  function handleFileChange(event) {
    const file = event.target.files[0];

    if (file) {
      setSelectedFile(file);
      setResult(null);
      setError("");

      if (onAnalysisComplete) {
        onAnalysisComplete(null);
      }
    }
  }

  async function handleAnalyze() {
    if (!selectedFile) {
      setError("Please choose a PDF first.");
      return;
    }

    setIsAnalyzing(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await fetch("http://127.0.0.1:8000/upload-contract", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed.");
      }

      const data = await response.json();

      setResult(data);

      if (onAnalysisComplete) {
        onAnalysisComplete(data);
      }
    } catch (err) {
      setError("Could not connect to backend. Make sure FastAPI is running.");
    } finally {
      setIsAnalyzing(false);
    }
  }

  return (
    <div>
      <div className="border border-dashed border-slate-300 bg-[#fbfaf7] rounded-2xl p-8 text-center">
        <p className="text-sm text-slate-500 mb-2">
          Upload legal document
        </p>

        <h4 className="text-2xl font-semibold text-slate-950">
          Start by uploading a contract PDF
        </h4>

        <p className="text-slate-600 mt-3 max-w-xl mx-auto">
          The backend will extract text, detect basic risk signals, and prepare clauses for ML classification.
        </p>

        <div className="mt-6 flex flex-col sm:flex-row items-center justify-center gap-3">
          <label className="inline-block bg-white border border-slate-300 text-slate-950 px-6 py-3 rounded-full font-medium cursor-pointer hover:border-slate-500">
            Choose PDF
            <input
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleFileChange}
            />
          </label>

          <button
            onClick={handleAnalyze}
            disabled={!selectedFile || isAnalyzing}
            className="bg-slate-950 text-white px-6 py-3 rounded-full font-medium hover:bg-slate-800 disabled:bg-slate-300 disabled:cursor-not-allowed"
          >
            {isAnalyzing ? "Analyzing..." : "Analyze Contract"}
          </button>
        </div>

        {selectedFile && (
          <p className="mt-4 text-sm text-slate-600">
            Selected: <span className="font-medium text-slate-950">{selectedFile.name}</span>
          </p>
        )}

        {error && (
          <p className="mt-4 text-sm text-red-600">
            {error}
          </p>
        )}
      </div>

      {result && (
        <div className="mt-4 bg-slate-950 text-white rounded-2xl p-5">
          <p className="text-sm text-slate-300">Analysis ready</p>

          <div className="mt-3 grid grid-cols-1 md:grid-cols-3 gap-3">
            <div>
              <p className="text-xs text-slate-400">File</p>
              <p className="text-sm font-medium truncate">{result.filename}</p>
            </div>

            <div>
              <p className="text-xs text-slate-400">Characters</p>
              <p className="text-sm font-medium">{result.total_characters}</p>
            </div>

            <div>
              <p className="text-xs text-slate-400">Risk level</p>
              <p className="text-sm font-medium">{result.analysis?.risk_level}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default UploadBox;
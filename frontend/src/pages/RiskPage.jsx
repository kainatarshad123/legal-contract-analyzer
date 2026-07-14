function RiskPage({ analysisResult, onBack }) {
  const analysis = analysisResult?.analysis;
  const keywords = analysis?.risk_keywords_found || [];

  return (
    <main className="max-w-5xl mx-auto px-6 py-10">
      <button
        onClick={onBack}
        className="mb-6 text-sm text-slate-600 hover:text-slate-950"
      >
        ← Back to dashboard
      </button>

      <div className="bg-white border border-slate-200 rounded-[28px] p-8">
        <p className="text-sm text-slate-500 mb-2">Risk signals</p>
        <h1 className="text-4xl font-semibold tracking-tight">
          Risk Analysis
        </h1>

        {!analysisResult ? (
          <p className="text-slate-600 mt-6">
            Upload and analyze a contract first.
          </p>
        ) : (
          <div className="mt-8 space-y-6">
            <div className="bg-slate-950 text-white rounded-3xl p-8">
              <p className="text-sm text-slate-300">Overall risk level</p>
              <p className="text-5xl font-semibold mt-2">
                {analysis?.risk_level}
              </p>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">
                Risk keywords detected
              </h3>

              {keywords.length > 0 ? (
                <div className="flex flex-wrap gap-2">
                  {keywords.map((keyword) => (
                    <span
                      key={keyword}
                      className="bg-[#f7f5f0] border border-slate-200 rounded-full px-4 py-2 text-sm"
                    >
                      {keyword}
                    </span>
                  ))}
                </div>
              ) : (
                <p className="text-slate-600">
                  No risk keywords detected.
                </p>
              )}
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default RiskPage;
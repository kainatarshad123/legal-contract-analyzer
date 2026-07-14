function ReviewPage({ analysisResult, onBack }) {
  return (
    <main className="max-w-5xl mx-auto px-6 py-10">
      <button
        onClick={onBack}
        className="mb-6 text-sm text-slate-600 hover:text-slate-950"
      >
        ← Back to dashboard
      </button>

      <div className="bg-white border border-slate-200 rounded-[28px] p-8">
        <p className="text-sm text-slate-500 mb-2">Document review</p>
        <h1 className="text-4xl font-semibold tracking-tight">
          Review Summary
        </h1>

        {!analysisResult ? (
          <p className="text-slate-600 mt-6">
            Upload and analyze a contract first.
          </p>
        ) : (
          <div className="mt-8 space-y-6">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className="bg-[#f7f5f0] rounded-2xl p-5">
                <p className="text-sm text-slate-500">File</p>
                <p className="font-medium mt-1">{analysisResult.filename}</p>
              </div>

              <div className="bg-[#f7f5f0] rounded-2xl p-5">
                <p className="text-sm text-slate-500">Characters extracted</p>
                <p className="font-medium mt-1">{analysisResult.total_characters}</p>
              </div>

              <div className="bg-[#f7f5f0] rounded-2xl p-5">
                <p className="text-sm text-slate-500">Risk level</p>
                <p className="font-medium mt-1">{analysisResult.analysis?.risk_level}</p>
              </div>
            </div>

            <div>
              <h3 className="text-lg font-semibold mb-3">Extracted text preview</h3>
              <div className="bg-[#f7f5f0] rounded-2xl p-5 text-sm leading-6 text-slate-700 whitespace-pre-line">
                {analysisResult.text_preview}
              </div>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}

export default ReviewPage;
function ClausesPage({ analysisResult, onBack }) {
  const clauses = analysisResult?.clauses || [];

  return (
    <main className="max-w-5xl mx-auto px-6 py-10">
      <button
        onClick={onBack}
        className="mb-6 text-sm text-slate-600 hover:text-slate-950"
      >
        ← Back to dashboard
      </button>

      <div className="bg-white border border-slate-200 rounded-[28px] p-8">
        <p className="text-sm text-slate-500 mb-2">Clause workspace</p>
        <h1 className="text-4xl font-semibold tracking-tight">
          Detected Clauses
        </h1>

        {clauses.length === 0 ? (
          <p className="text-slate-600 mt-6">
            Upload and analyze a contract first.
          </p>
        ) : (
          <div className="mt-8 space-y-4">
            {clauses.map((clause) => (
              <details
                key={clause.clause_number}
                className="bg-[#f7f5f0] rounded-2xl p-5"
              >
                <summary className="cursor-pointer list-none">
                  <div className="flex items-center justify-between gap-4">
                    <h3 className="font-semibold">
                      Clause {clause.clause_number}
                    </h3>

                    <span className="text-xs bg-white border border-slate-200 rounded-full px-3 py-1">
                      {clause.risk_level}
                    </span>
                  </div>
                </summary>

                <p className="text-sm leading-6 text-slate-700 mt-4">
                  {clause.text}
                </p>

                {clause.risk_keywords_found?.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-4">
                    {clause.risk_keywords_found.map((keyword) => (
                      <span
                        key={keyword}
                        className="text-xs bg-white border border-slate-200 rounded-full px-3 py-1 text-slate-600"
                      >
                        {keyword}
                      </span>
                    ))}
                  </div>
                )}
              </details>
            ))}
          </div>
        )}
      </div>
    </main>
  );
}

export default ClausesPage;
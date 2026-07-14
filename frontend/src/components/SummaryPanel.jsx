import { useState } from "react";

function SummaryPanel({ analysisResult }) {
  const [activeTab, setActiveTab] = useState("overview");

  const analysis = analysisResult?.analysis;
  const clauses = analysisResult?.clauses || [];

  return (
    <aside className="w-96 bg-white rounded-xl border border-slate-200 p-6 overflow-y-auto">
      <h3 className="text-xl font-semibold text-slate-950">
        Review Summary
      </h3>

      <p className="text-sm text-slate-500 mt-1">
        Results will appear after analysis.
      </p>

      <div className="mt-5 flex border border-slate-200 rounded-lg overflow-hidden">
        <button
          onClick={() => setActiveTab("overview")}
          className={`flex-1 px-4 py-2 text-sm font-medium ${
            activeTab === "overview"
              ? "bg-slate-950 text-white"
              : "bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          Overview
        </button>

        <button
          onClick={() => setActiveTab("clauses")}
          className={`flex-1 px-4 py-2 text-sm font-medium ${
            activeTab === "clauses"
              ? "bg-slate-950 text-white"
              : "bg-white text-slate-600 hover:bg-slate-50"
          }`}
        >
          Clauses
        </button>
      </div>

      {activeTab === "overview" && (
        <div className="mt-6 space-y-5">
          <div className="border border-slate-200 rounded-xl p-4">
            <p className="text-sm text-slate-500">Risk level</p>
            <p className="text-2xl font-semibold text-slate-950 mt-1">
              {analysis?.risk_level || "Pending"}
            </p>
          </div>

          <div>
            <h4 className="font-medium text-slate-950 mb-3">
              Risk keywords found
            </h4>

            {analysis?.risk_keywords_found?.length > 0 ? (
              <div className="flex flex-wrap gap-2">
                {analysis.risk_keywords_found.map((keyword) => (
                  <span
                    key={keyword}
                    className="text-xs border border-slate-300 rounded-full px-3 py-1 text-slate-700 bg-slate-50"
                  >
                    {keyword}
                  </span>
                ))}
              </div>
            ) : (
              <p className="text-sm text-slate-500">
                Upload a contract to detect risky keywords.
              </p>
            )}
          </div>

          {analysisResult && (
            <div className="border-t border-slate-200 pt-5">
              <h4 className="font-medium text-slate-950 mb-2">
                File details
              </h4>

              <p className="text-sm text-slate-600">
                {analysisResult.filename}
              </p>

              <p className="text-sm text-slate-500 mt-1">
                {analysisResult.total_characters} characters extracted
              </p>
            </div>
          )}
        </div>
      )}

      {activeTab === "clauses" && (
        <div className="mt-6">
          <h4 className="font-medium text-slate-950 mb-3">
            Clauses detected
          </h4>

          {clauses.length > 0 ? (
            <div className="space-y-3">
              {clauses.map((clause) => (
                <details
                  key={clause.clause_number}
                  className="border border-slate-200 rounded-lg bg-slate-50"
                >
                  <summary className="cursor-pointer list-none p-3">
                    <div className="flex items-center justify-between gap-3">
                      <p className="text-sm font-medium text-slate-950">
                        Clause {clause.clause_number}
                      </p>

                      <span className="text-xs border border-slate-300 rounded-full px-2 py-1 text-slate-700 bg-white">
                        {clause.risk_level}
                      </span>
                    </div>
                  </summary>

                  <div className="px-3 pb-3">
                    <p className="text-xs text-slate-600">
                      {clause.text}
                    </p>

                    {clause.risk_keywords_found?.length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-3">
                        {clause.risk_keywords_found.map((keyword) => (
                          <span
                            key={keyword}
                            className="text-[11px] border border-slate-300 rounded-full px-2 py-0.5 text-slate-600 bg-white"
                          >
                            {keyword}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </details>
              ))}
            </div>
          ) : (
            <p className="text-sm text-slate-500">
              Upload and analyze a contract to see detected clauses.
            </p>
          )}
        </div>
      )}
    </aside>
  );
}

export default SummaryPanel;
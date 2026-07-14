import { useState } from "react";
import Header from "../components/Header";
import ChatWindow from "../components/ChatWindow";
import ReviewPage from "./ReviewPage";
import ClausesPage from "./ClausesPage";
import RiskPage from "./RiskPage";

function Dashboard() {
  const [analysisResult, setAnalysisResult] = useState(null);
  const [activePage, setActivePage] = useState("home");

  return (
    <div className="min-h-screen bg-[#f7f5f0] text-slate-950">
      <Header />

      {activePage === "home" && (
        <main className="max-w-6xl mx-auto px-6 py-10">
          <section className="text-center mb-10">
            <p className="text-sm font-medium text-slate-500 mb-3">
              ML-powered legal contract workspace
            </p>

            <h1 className="text-5xl font-semibold tracking-tight text-slate-950">
              Review contracts with clarity.
            </h1>

            <p className="max-w-2xl mx-auto text-slate-600 mt-4 text-lg">
              Upload a legal document, extract clauses, detect basic risk signals,
              and ask questions from one focused workspace.
            </p>
          </section>

          <ChatWindow
            analysisResult={analysisResult}
            onAnalysisComplete={setAnalysisResult}
          />

          <section className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-8">
            <button
              onClick={() => setActivePage("review")}
              className="text-left bg-white border border-slate-200 rounded-2xl p-6 hover:border-slate-400 hover:shadow-sm transition"
            >
              <p className="text-sm text-slate-500 mb-2">01</p>
              <h3 className="text-xl font-semibold">Review</h3>
              <p className="text-slate-600 mt-2">
                View file details, extracted text preview, and analysis summary.
              </p>
            </button>

            <button
              onClick={() => setActivePage("clauses")}
              className="text-left bg-white border border-slate-200 rounded-2xl p-6 hover:border-slate-400 hover:shadow-sm transition"
            >
              <p className="text-sm text-slate-500 mb-2">02</p>
              <h3 className="text-xl font-semibold">Clauses</h3>
              <p className="text-slate-600 mt-2">
                Open detected clauses in a dedicated full-page view.
              </p>
            </button>

            <button
              onClick={() => setActivePage("risk")}
              className="text-left bg-white border border-slate-200 rounded-2xl p-6 hover:border-slate-400 hover:shadow-sm transition"
            >
              <p className="text-sm text-slate-500 mb-2">03</p>
              <h3 className="text-xl font-semibold">Risk Analysis</h3>
              <p className="text-slate-600 mt-2">
                See risk level, detected risk keywords, and clause warnings.
              </p>
            </button>
          </section>
        </main>
      )}

      {activePage === "review" && (
        <ReviewPage
          analysisResult={analysisResult}
          onBack={() => setActivePage("home")}
        />
      )}

      {activePage === "clauses" && (
        <ClausesPage
          analysisResult={analysisResult}
          onBack={() => setActivePage("home")}
        />
      )}

      {activePage === "risk" && (
        <RiskPage
          analysisResult={analysisResult}
          onBack={() => setActivePage("home")}
        />
      )}
    </div>
  );
}

export default Dashboard;
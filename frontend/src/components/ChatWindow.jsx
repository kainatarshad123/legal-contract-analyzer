import { useState } from "react";
import UploadBox from "./UploadBox";

function ChatWindow({ analysisResult, onAnalysisComplete }) {
  const [question, setQuestion] = useState("");
  const [chatAnswer, setChatAnswer] = useState("");
  const [isAsking, setIsAsking] = useState(false);
  const [chatError, setChatError] = useState("");

  async function handleAsk() {
    if (!analysisResult?.contract_id) {
      setChatError("Please upload and analyze a contract first.");
      return;
    }

    if (!question.trim()) {
      setChatError("Please type a question first.");
      return;
    }

    setIsAsking(true);
    setChatError("");
    setChatAnswer("");

    try {
      const response = await fetch("http://127.0.0.1:8000/ask-contract", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          contract_id: analysisResult.contract_id,
        }),
      });

      if (!response.ok) {
        throw new Error("Question request failed.");
      }

      const data = await response.json();
      setChatAnswer(data.answer);
    } catch (error) {
      setChatError("Could not get answer from backend. Make sure FastAPI is running.");
    } finally {
      setIsAsking(false);
    }
  }

  return (
    <section className="bg-white border border-slate-200 rounded-[28px] p-6 shadow-sm">
      <UploadBox onAnalysisComplete={onAnalysisComplete} />

      <div className="mt-6 bg-[#f7f5f0] rounded-2xl p-5">
        <p className="text-sm font-medium text-slate-500 mb-3">
          Ask from this contract
        </p>

        <div className="flex flex-col md:flex-row gap-3">
          <input
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            className="flex-1 bg-white border border-slate-200 rounded-full px-5 py-4 outline-none focus:ring-2 focus:ring-slate-900"
            placeholder="Ask about termination, payment terms, risks, or summary..."
          />

          <button
            onClick={handleAsk}
            disabled={isAsking}
            className="bg-slate-950 text-white px-8 py-4 rounded-full font-medium hover:bg-slate-800 disabled:bg-slate-400"
          >
            {isAsking ? "Asking..." : "Ask"}
          </button>
        </div>

        <div className="flex flex-wrap gap-2 mt-4">
          <button
            onClick={() => setQuestion("Summarize this contract")}
            className="px-4 py-2 rounded-full bg-white border border-slate-200 text-sm text-slate-600 hover:border-slate-400"
          >
            Summarize
          </button>

          <button
            onClick={() => setQuestion("What are the risky clauses?")}
            className="px-4 py-2 rounded-full bg-white border border-slate-200 text-sm text-slate-600 hover:border-slate-400"
          >
            Risky clauses
          </button>

          <button
            onClick={() => setQuestion("Explain the termination clause")}
            className="px-4 py-2 rounded-full bg-white border border-slate-200 text-sm text-slate-600 hover:border-slate-400"
          >
            Termination
          </button>

          <button
            onClick={() => setQuestion("What are the payment obligations?")}
            className="px-4 py-2 rounded-full bg-white border border-slate-200 text-sm text-slate-600 hover:border-slate-400"
          >
            Payment
          </button>
        </div>

        {chatError && (
          <p className="text-sm text-red-600 mt-4">
            {chatError}
          </p>
        )}

        {chatAnswer && (
          <div className="mt-5 bg-white border border-slate-200 rounded-2xl p-5">
            <p className="text-sm font-semibold text-slate-950 mb-2">
              Contract Agent Response
            </p>

            <p className="text-sm leading-6 text-slate-700 whitespace-pre-line">
              {chatAnswer}
            </p>
          </div>
        )}
      </div>
    </section>
  );
}

export default ChatWindow;

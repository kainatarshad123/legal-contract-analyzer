import { useState, useEffect, useRef } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:8000";

/* ------------------------------------------------------------------ */
/* Icons — small hand-drawn line icons, no external icon dependency    */
/* ------------------------------------------------------------------ */

function BrandMark({ size = 26 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24">
      <rect x="0.5" y="0.5" width="23" height="23" rx="7" fill="var(--brass)" />
      <text
        x="12"
        y="16.6"
        textAnchor="middle"
        fontFamily="'Source Serif 4', serif"
        fontSize="13"
        fontWeight="700"
        fill="var(--ink)"
      >
        §
      </text>
    </svg>
  );
}

function IconBase({ children, size = 16, ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      {...rest}
    >
      {children}
    </svg>
  );
}

const IconDoc = (p) => (
  <IconBase {...p}>
    <path d="M7 3h7l4 4v14a1 1 0 0 1-1 1H7a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1Z" />
    <path d="M14 3v4h4" />
    <path d="M9 13h6M9 17h6M9 9h2" />
  </IconBase>
);

const IconLayers = (p) => (
  <IconBase {...p}>
    <path d="m12 3 8 4.5-8 4.5-8-4.5Z" />
    <path d="m4 12 8 4.5 8-4.5" />
    <path d="m4 16.5 8 4.5 8-4.5" />
  </IconBase>
);

const IconAlert = (p) => (
  <IconBase {...p}>
    <path d="M12 3 2 20h20Z" />
    <path d="M12 10v4" />
    <circle cx="12" cy="17" r="0.4" fill="currentColor" />
  </IconBase>
);

const IconFlag = (p) => (
  <IconBase {...p}>
    <path d="M5 21V4" />
    <path d="M5 4h13l-3 4 3 4H5" />
  </IconBase>
);

const IconClock = (p) => (
  <IconBase {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7v5l3.5 2" />
  </IconBase>
);

const IconPlus = (p) => (
  <IconBase {...p}>
    <path d="M12 5v14M5 12h14" />
  </IconBase>
);

const IconPaperclip = (p) => (
  <IconBase {...p}>
    <path d="M8 12.5 15.3 5a3 3 0 0 1 4.2 4.2l-8.4 8.4a5 5 0 1 1-7-7L12 3" />
  </IconBase>
);

const IconSend = (p) => (
  <IconBase {...p}>
    <path d="m4 12 16-8-6 16-2.5-6.5L4 12Z" />
  </IconBase>
);

const IconCamera = (p) => (
  <IconBase {...p}>
    <path d="M4 8h3l1.5-2h7L17 8h3a1 1 0 0 1 1 1v9a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1V9a1 1 0 0 1 1-1Z" />
    <circle cx="12" cy="13.5" r="3.3" />
  </IconBase>
);

const IconImage = (p) => (
  <IconBase {...p}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <circle cx="8.5" cy="9.5" r="1.5" />
    <path d="m4 17 5-5 4 4 3-3 4 4" />
  </IconBase>
);

const IconTrash = (p) => (
  <IconBase {...p}>
    <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />
  </IconBase>
);

const IconFolderOpen = (p) => (
  <IconBase {...p}>
    <path d="M3 7a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1l-1.5 9a1 1 0 0 1-1 .8H5.4a1 1 0 0 1-1-.85Z" />
  </IconBase>
);

const IconChevron = (p) => (
  <IconBase {...p}>
    <path d="m6 9 6 6 6-6" />
  </IconBase>
);

const IconExpand = (p) => (
  <IconBase {...p}>
    <path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7" />
  </IconBase>
);

const IconX = (p) => (
  <IconBase {...p}>
    <path d="m6 6 12 12M18 6 6 18" />
  </IconBase>
);

const IconSparkle = (p) => (
  <IconBase {...p}>
    <path d="M12 3v4M12 17v4M3 12h4M17 12h4" />
    <path d="m6 6 2.5 2.5M15.5 15.5 18 18M6 18l2.5-2.5M15.5 8.5 18 6" />
  </IconBase>
);

const IconDollar = (p) => (
  <IconBase {...p}>
    <path d="M12 2v20M17 6.5c0-1.9-2.2-3.5-5-3.5s-5 1.6-5 3.5 2.2 3 5 3.5 5 1.6 5 3.5-2.2 3.5-5 3.5-5-1.6-5-3.5" />
  </IconBase>
);

const NAV_ICONS = {
  review: IconDoc,
  clauses: IconLayers,
  risks: IconAlert,
  missing: IconFlag,
  history: IconClock,
};

/* ------------------------------------------------------------------ */
/* App                                                                 */
/* ------------------------------------------------------------------ */

function App() {
  const [savedContracts, setSavedContracts] = useState([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("review");
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedFile, setSelectedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [question, setQuestion] = useState("");
  const [asking, setAsking] = useState(false);
  const [latestStructuredAnswer, setLatestStructuredAnswer] = useState(null);
  const [chatHistory, setChatHistory] = useState([
    {
      role: "agent",
      text: "Upload or open a legal contract, then ask contract-specific questions here.",
    },
  ]);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);
  };

  const buildSummaryAnswer = (data, fallbackSummary, filename) => {
  const contractType = data?.analysis?.contract_type || "Unknown";
  const overallRisk = data?.analysis?.overall_risk || "Unknown";
  const totalClauses = data?.analysis?.total_clauses || 0;
  const riskyClauses = data?.analysis?.risky_clauses || 0;

  return {
    answer_type: "summary",
    title: "Contract Summary",
    risk_level: overallRisk,
    summary:
      fallbackSummary ||
      "The contract was uploaded and analyzed successfully. Review the contract details, missing fields, risky clauses, and extracted clauses below.",
    key_points: [
      `Contract type: ${contractType}`,
      `Overall risk level: ${overallRisk}`,
      `Total clauses detected: ${totalClauses}`,
      `Risky clauses detected: ${riskyClauses}`,
    ],
    missing_fields: data?.analysis?.missing_fields || [],
    related_clauses: [],
    contract_details: {
      contract_type: contractType,
      ...(filename ? { filename } : {}),
    },
    answer: "",
  };
};

  const isContractLegalQuestion = (value) => {
    const text = value.toLowerCase();

    const legalKeywords = [
      "contract", "agreement", "clause", "legal", "law", "rights",
      "obligation", "liability", "risk", "risky", "payment", "rent",
      "termination", "terminate", "missing", "party", "parties",
      "lessor", "lessee", "breach", "default", "indemnity", "indemnify",
      "confidential", "nda", "non-disclosure", "lease", "employment",
      "service", "summarize", "summary", "explain", "review", "analyze",
      "analyse", "compare", "draft", "create agreement", "create contract",
      "arbitration", "dispute", "damages", "penalty", "notice",
      "mean", "meaning", "define", "definition", "illegal", "legally",
      "enforceable", "valid", "void", "sue", "lawsuit", "court",
      "attorney", "lawyer", "evict", "eviction", "landlord", "tenant",
      "sign", "signature", "am i allowed", "can they", "can i", "my rights",
    ];

    return legalKeywords.some((keyword) => text.includes(keyword));
  };

  const makeShortAgentMessage = (data) => {
    if (!data) return "I prepared the result in the main analysis panel.";

    const title = data.title || "Contract analysis";
    const risk = data.risk_level ? ` Risk level: ${data.risk_level}.` : "";
    const summary = data.summary || data.answer || "Review the full structured result in the main response panel.";

    return `${title}.${risk} ${summary}`;
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      alert("Please choose a PDF contract first.");
      return;
    }

    try {
      setUploading(true);

      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(`${API_BASE_URL}/upload-contract`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Upload failed");
      }

      const data = await response.json();

      setAnalysisResult(data);
      const summaryAnswer = buildSummaryAnswer(
        data,
        "The contract was uploaded and analyzed successfully.",
        data?.filename
      );

      setLatestStructuredAnswer(summaryAnswer);
      setChatHistory([
        {
          role: "agent",
          text: "Contract uploaded and analyzed. I prepared the full summary in the main response panel.",
          structured: summaryAnswer,
        },
      ]);
      setActiveTab("review");
      loadContractHistory();
    } catch (error) {
      console.error(error);
      alert("Something went wrong while uploading the contract.");
    } finally {
      setUploading(false);
    }
  };

  const handleAskQuestion = async (directQuestion = null) => {
    if (!analysisResult?.contract_id) {
      alert("Please upload or open a contract first.");
      return;
    }

    const userQuestion = (directQuestion || question).trim();

    if (!userQuestion) {
      return;
    }

    setQuestion("");

    setChatHistory((previous) => [
      ...previous,
      {
        role: "user",
        text: userQuestion,
      },
    ]);

    if (!isContractLegalQuestion(userQuestion)) {
      const legalOnlyAnswer = {
        answer_type: "legal_scope",
        title: "Legal Contract Questions Only",
        risk_level: "Unknown",
        summary:
          "I can only help with legal contract review, agreement analysis, clause risk, missing fields, payment terms, termination, obligations, and related legal-document questions.",
        key_points: [
          "Please ask a question about the uploaded contract or legal agreement.",
          "Examples: summarize this contract, explain the termination clause, identify risky clauses, or review payment terms.",
        ],
        missing_fields: [],
        related_clauses: [],
        contract_details: {},
      };

      setLatestStructuredAnswer(legalOnlyAnswer);
      setChatHistory((previous) => [
        ...previous,
        {
          role: "agent",
          text: "I can only help with legal contract review and related legal-document questions.",
        },
      ]);

      return;
    }

    try {
      setAsking(true);

      const response = await fetch(`${API_BASE_URL}/ask-contract`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: userQuestion,
          contract_id: analysisResult.contract_id,
        }),
      });

      if (!response.ok) {
        throw new Error("Question request failed");
      }

      const data = await response.json();

      setLatestStructuredAnswer(data);
      setChatHistory((previous) => [
        ...previous,
        {
          role: "agent",
          text: makeShortAgentMessage(data),
          structured: data,
        },
      ]);
    } catch (error) {
      console.error(error);

      const errorAnswer = {
        answer_type: "error",
        title: "Contract Agent Error",
        risk_level: "Unknown",
        summary:
          "Something went wrong while asking the contract agent. Please check that the backend is running and try again.",
        key_points: [],
        missing_fields: [],
        related_clauses: [],
        contract_details: {},
      };

      setLatestStructuredAnswer(errorAnswer);
      setChatHistory((previous) => [
        ...previous,
        {
          role: "agent",
          text: "Something went wrong while asking the contract agent.",
        },
      ]);
    } finally {
      setAsking(false);
    }
  };

  const loadContractHistory = async () => {
    try {
      setHistoryLoading(true);

      const response = await fetch(`${API_BASE_URL}/contracts`);

      if (!response.ok) {
        throw new Error("Failed to load contract history");
      }

      const data = await response.json();
      setSavedContracts(data.contracts || []);
    } catch (error) {
      console.error("History loading error:", error);
    } finally {
      setHistoryLoading(false);
    }
  };

  const openSavedContract = async (contractId) => {
    try {
      const response = await fetch(`${API_BASE_URL}/contracts/${contractId}`);

      if (!response.ok) {
        throw new Error("Failed to open saved contract");
      }

      const data = await response.json();

      if (data.error) {
        alert(data.message || "Contract not found.");
        return;
      }

      const loadedContract = {
        contract_id: data.contract_id,
        filename: data.filename,
        analysis: data.analysis || {},
        clauses: data.clauses || [],
      };

      const savedSummaryAnswer = buildSummaryAnswer(
        loadedContract,
        "This saved contract was loaded successfully.",
        data.filename
      );

      setAnalysisResult(loadedContract);
      setLatestStructuredAnswer(savedSummaryAnswer);
      setChatHistory([
        {
          role: "agent",
          text: `Saved contract "${data.filename}" opened. I prepared the full summary in the main response panel.`,
          structured: savedSummaryAnswer,
        },
      ]);
      setSelectedFile(null);
      setActiveTab("review");
    } catch (error) {
      console.error("Open saved contract error:", error);
      alert("Could not open this saved contract.");
    }
  };

  const deleteSavedContract = async (contractId) => {
    const confirmed = window.confirm(
      "Are you sure you want to delete this saved contract from history?"
    );

    if (!confirmed) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/contracts/${contractId}`, {
        method: "DELETE",
      });

      if (!response.ok) {
        throw new Error("Failed to delete contract");
      }

      const data = await response.json();

      if (data.error) {
        alert(data.message || "Contract could not be deleted.");
        return;
      }

      setSavedContracts((previous) =>
        previous.filter((contract) => contract.contract_id !== contractId)
      );

      if (analysisResult?.contract_id === contractId) {
        handleNewReview();
      }
    } catch (error) {
      console.error("Delete contract error:", error);
      alert("Could not delete this contract.");
    }
  };

  useEffect(() => {
    loadContractHistory();
  }, []);

  const handleNewReview = () => {
    setAnalysisResult(null);
    setLatestStructuredAnswer(null);
    setSelectedFile(null);
    setQuestion("");
    setActiveTab("review");
    setChatHistory([
      {
        role: "agent",
        text: "Upload or open a legal contract, then ask contract-specific questions here.",
      },
    ]);
  };

  return (
    <div className="app-shell">
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        loadContractHistory={loadContractHistory}
        handleAskQuestion={handleAskQuestion}
        hasContract={Boolean(analysisResult)}
        savedContracts={savedContracts}
        openSavedContract={openSavedContract}
      />

      <main className="main-workspace">
        <TopBar analysisResult={analysisResult} onNewReview={handleNewReview} />

        {!analysisResult && activeTab !== "history" && (
          <LandingUpload
            selectedFile={selectedFile}
            uploading={uploading}
            handleFileChange={handleFileChange}
            handleUpload={handleUpload}
          />
        )}

        {activeTab === "history" && (
          <HistoryView
            savedContracts={savedContracts}
            historyLoading={historyLoading}
            loadContractHistory={loadContractHistory}
            openSavedContract={openSavedContract}
            deleteSavedContract={deleteSavedContract}
          />
        )}

        {analysisResult && activeTab !== "history" && (
          <>
            {activeTab === "review" && (
              <ReviewWorkspace
                latestStructuredAnswer={latestStructuredAnswer}
                setLatestStructuredAnswer={setLatestStructuredAnswer}
                chatHistory={chatHistory}
                question={question}
                setQuestion={setQuestion}
                handleAskQuestion={handleAskQuestion}
                selectedFile={selectedFile}
                handleFileChange={handleFileChange}
                handleUpload={handleUpload}
                uploading={uploading}
                asking={asking}
              />
            )}

            {activeTab === "clauses" && (
              <ClausesView analysisResult={analysisResult} />
            )}

            {activeTab === "risks" && (
              <RiskView analysisResult={analysisResult} />
            )}

            {activeTab === "missing" && (
              <MissingFieldsView analysisResult={analysisResult} />
            )}
          </>
        )}
      </main>

      <RightPanel analysisResult={analysisResult} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sidebar                                                             */
/* ------------------------------------------------------------------ */

function Sidebar({
  activeTab,
  setActiveTab,
  loadContractHistory,
  handleAskQuestion,
  hasContract,
  savedContracts,
  openSavedContract,
}) {

  const tabs = [
    { id: "review", label: "Review" },
    { id: "clauses", label: "Clauses" },
    { id: "risks", label: "Risks" },
    { id: "missing", label: "Missing info" },
    { id: "history", label: "History" },
  ];

  const legalTools = [
    { label: "Agreement summary", prompt: "Summarize this contract", icon: IconDoc },
    { label: "Payment terms", prompt: "What are the payment terms?", icon: IconDollar },
    { label: "Termination review", prompt: "Explain the termination clause", icon: IconAlert },
    { label: "Risky clauses", prompt: "What are the risky clauses?", icon: IconSparkle },
  ];

  const handleTabClick = (tabId) => {
    const contractRequiredTabs = ["clauses", "risks", "missing"];

    if (contractRequiredTabs.includes(tabId) && !hasContract) {
      alert("Please upload or open a contract first.");
      setActiveTab("review");
      return;
    }

    setActiveTab(tabId);

    if (tabId === "history") {
      loadContractHistory();
    }
  };

  const handleToolClick = (prompt) => {
    if (!hasContract) {
      alert("Please upload or open a contract first.");
      return;
    }

    setActiveTab("review");
    handleAskQuestion(prompt);
  };

  return (
    <aside className="sidebar">
      <div className="brand">
        <BrandMark />
        <div>
          <h1>LegalContract AI</h1>
          <p>ML contract review</p>
        </div>
      </div>

      <nav className="sidebar-nav">
        {tabs.map((tab) => {
          const Icon = NAV_ICONS[tab.id];
          return (
            <button
              key={tab.id}
              className={`sidebar-link ${activeTab === tab.id ? "active" : ""}`}
              onClick={() => handleTabClick(tab.id)}
            >
              <Icon size={16} />
              {tab.label}
            </button>
          );
        })}
      </nav>

      <div className="sidebar-section">
        <p className="sidebar-section-label">Legal tools</p>

        {legalTools.map((tool) => (
          <button
            key={tool.label}
            type="button"
            className="sidebar-tool-btn"
            onClick={() => handleToolClick(tool.prompt)}
          >
            <tool.icon size={15} />
            {tool.label}
          </button>
        ))}
      </div>

      <div className="sidebar-section sidebar-recent">
  <p className="sidebar-section-label">Recent contracts</p>

  <div className="sidebar-chat-list">
    {savedContracts.slice(0, 6).map((contract) => (
      <button
        key={contract.contract_id}
        type="button"
        className="sidebar-chat-item"
        onClick={() => openSavedContract(contract.contract_id)}
        title={contract.filename}
      >
        <IconDoc size={14} />
        <span>{contract.filename}</span>
      </button>
    ))}

    {savedContracts.length === 0 && (
      <p className="sidebar-empty-note">
        Uploaded contracts appear here.
      </p>
    )}
  </div>
</div>

      <div className="sidebar-footer">
        <span className="status-dot live" />
        <div>
          <p>Prototype workspace</p>
          <span>Manual ML dataset active</span>
        </div>
      </div>
    </aside>
  );
}

/* ------------------------------------------------------------------ */
/* Top bar                                                             */
/* ------------------------------------------------------------------ */

function TopBar({ analysisResult, onNewReview }) {
  return (
    <header className="topbar">
      <div>
        <p className="eyebrow">Legal contract ML agent · SaaS</p>
        <h2>
          {analysisResult
            ? analysisResult.filename
            : "Upload and review legal contracts"}
        </h2>
      </div>

      <div className="topbar-right">
        <div className="topbar-status">
          <span className={analysisResult ? "status-dot live" : "status-dot"} />
          {analysisResult ? "Contract loaded" : "Waiting for upload"}
        </div>

        {analysisResult && (
          <button type="button" className="topbar-new-btn" onClick={onNewReview}>
            <IconPlus size={14} />
            New
          </button>
        )}
      </div>
    </header>
  );
}

/* ------------------------------------------------------------------ */
/* Image -> single-page PDF conversion (no external libraries)         */
/*                                                                       */
/* The backend's /upload-contract endpoint only accepts real PDF bytes  */
/* (PyMuPDF, filetype="pdf"). To let people attach a photo instead of a */
/* PDF without breaking that endpoint, we wrap the photo into a minimal */
/* valid one-page PDF, client-side, before uploading it.                */
/* NOTE: this does not add OCR — a scanned photo still has no text      */
/* layer, so analysis results will be sparse until OCR is added on the  */
/* backend. See the chat for details.                                   */
/* ------------------------------------------------------------------ */

async function imageFileToPdfFile(imageFile) {
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(imageFile);
  });

  const image = await new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = dataUrl;
  });

  const maxDimension = 1600;
  const scale = Math.min(1, maxDimension / Math.max(image.width, image.height));
  const width = Math.round(image.width * scale);
  const height = Math.round(image.height * scale);

  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  canvas.getContext("2d").drawImage(image, 0, 0, width, height);

  const jpegDataUrl = canvas.toDataURL("image/jpeg", 0.82);
  const base64 = jpegDataUrl.split(",")[1];
  const binary = atob(base64);
  const jpegBytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i += 1) {
    jpegBytes[i] = binary.charCodeAt(i);
  }

  const pdfBytes = buildSinglePageImagePdf(jpegBytes, width, height);
  const pdfName = imageFile.name.replace(/\.[^.]+$/, "") + "-photo.pdf";

  return new File([pdfBytes], pdfName, { type: "application/pdf" });
}

function buildSinglePageImagePdf(jpegBytes, width, height) {
  const encoder = new TextEncoder();
  const parts = [];
  let offset = 0;
  const objectOffsets = {};

  const push = (data) => {
    const bytes = typeof data === "string" ? encoder.encode(data) : data;
    parts.push(bytes);
    offset += bytes.length;
  };

  push("%PDF-1.4\n");

  objectOffsets[1] = offset;
  push("1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n");

  objectOffsets[2] = offset;
  push("2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n");

  objectOffsets[3] = offset;
  push(
    `3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${width} ${height}] ` +
      `/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>\nendobj\n`
  );

  objectOffsets[4] = offset;
  push(
    `4 0 obj\n<< /Type /XObject /Subtype /Image /Width ${width} /Height ${height} ` +
      `/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length ${jpegBytes.length} >>\nstream\n`
  );
  push(jpegBytes);
  push("\nendstream\nendobj\n");

  const contentStream = `q ${width} 0 0 ${height} 0 0 cm /Im0 Do Q`;
  objectOffsets[5] = offset;
  push(`5 0 obj\n<< /Length ${contentStream.length} >>\nstream\n${contentStream}\nendstream\nendobj\n`);

  const xrefOffset = offset;
  let xref = "xref\n0 6\n0000000000 65535 f \n";
  for (let i = 1; i <= 5; i += 1) {
    xref += `${String(objectOffsets[i]).padStart(10, "0")} 00000 n \n`;
  }
  push(xref);
  push(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xrefOffset}\n%%EOF`);

  const totalLength = parts.reduce((sum, part) => sum + part.length, 0);
  const result = new Uint8Array(totalLength);
  let position = 0;
  for (const part of parts) {
    result.set(part, position);
    position += part.length;
  }

  return result;
}

/* ------------------------------------------------------------------ */
/* Attach menu — PDF / camera / photo library                          */
/* ------------------------------------------------------------------ */

function AttachMenu({ onPdfSelected, onImageSelected, disabled }) {
  const [open, setOpen] = useState(false);
  const pdfInputRef = useRef(null);
  const cameraInputRef = useRef(null);
  const galleryInputRef = useRef(null);

  const closeOnBlur = (event) => {
    if (!event.currentTarget.contains(event.relatedTarget)) {
      setOpen(false);
    }
  };

  return (
    <div className="attach-menu" onBlur={closeOnBlur}>
      <button
        type="button"
        className="bar-icon-btn"
        onClick={() => setOpen((value) => !value)}
        aria-label="Attach a contract"
        disabled={disabled}
      >
        <IconPaperclip size={17} />
      </button>

      {open && (
        <div className="attach-menu-popover">
          <button
            type="button"
            onClick={() => {
              pdfInputRef.current?.click();
              setOpen(false);
            }}
          >
            <IconDoc size={15} />
            Upload PDF
          </button>

          <button
            type="button"
            onClick={() => {
              cameraInputRef.current?.click();
              setOpen(false);
            }}
          >
            <IconCamera size={15} />
            Take photo
          </button>

          <button
            type="button"
            onClick={() => {
              galleryInputRef.current?.click();
              setOpen(false);
            }}
          >
            <IconImage size={15} />
            Choose photo
          </button>
        </div>
      )}

      <input
        ref={pdfInputRef}
        type="file"
        accept="application/pdf"
        onChange={(event) => onPdfSelected(event.target.files[0])}
        hidden
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={(event) => onImageSelected(event.target.files[0])}
        hidden
      />
      <input
        ref={galleryInputRef}
        type="file"
        accept="image/*"
        onChange={(event) => onImageSelected(event.target.files[0])}
        hidden
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Landing / upload hero                                               */
/* ------------------------------------------------------------------ */

function LandingUpload({ selectedFile, uploading, handleFileChange, handleUpload }) {
  const [converting, setConverting] = useState(false);
  const [note, setNote] = useState("");
  const promptFileInputRef = useRef(null);

  const applyFile = (file) => {
    handleFileChange({ target: { files: [file] } });
  };

  const handlePdfSelected = (file) => {
    if (!file) return;
    setNote("");
    applyFile(file);
  };

  const handleImageSelected = async (file) => {
    if (!file) return;

    try {
      setConverting(true);
      const pdfFile = await imageFileToPdfFile(file);
      applyFile(pdfFile);
      setNote(
        "Photo converted to a PDF for upload. Scanned photos have no text layer yet (OCR isn't enabled on the backend), so results may be limited until that's added."
      );
    } catch (error) {
      console.error(error);
      alert("Could not process that photo. Please try a PDF instead.");
    } finally {
      setConverting(false);
    }
  };

  const prompts = [
    { icon: IconDoc, label: "Summarize a lease" },
    { icon: IconAlert, label: "Spot risky clauses" },
    { icon: IconFlag, label: "Find missing terms" },
    { icon: IconClock, label: "Review termination rights" },
  ];

  return (
    <section className="landing">
      <div className="landing-hero">
        <span className="landing-eyebrow">ML-powered legal document review</span>
        <h1>What contract should we review today?</h1>
        <p>
          Upload a contract PDF, or attach a photo of a printed contract. The
          agent extracts text, splits clauses, predicts risk levels, detects
          missing fields, and answers contract-specific questions.
        </p>
      </div>

      <div className="landing-prompts">
        {prompts.map((prompt) => (
          <button
            key={prompt.label}
            type="button"
            className="prompt-pill"
            onClick={() => promptFileInputRef.current?.click()}
          >
            <prompt.icon size={14} />
            {prompt.label}
          </button>
        ))}
      </div>

      <input
        ref={promptFileInputRef}
        type="file"
        accept="application/pdf"
        onChange={(event) => handlePdfSelected(event.target.files[0])}
        hidden
      />

      <div className="landing-input-bar">
        <AttachMenu
          onPdfSelected={handlePdfSelected}
          onImageSelected={handleImageSelected}
          disabled={converting}
        />

        <span className={`bar-filename ${selectedFile ? "" : "placeholder"}`}>
          {converting
            ? "Converting photo..."
            : selectedFile
            ? selectedFile.name
            : "Choose a contract PDF or photo to begin..."}
        </span>

        <button
          type="button"
          className="bar-send-btn"
          onClick={handleUpload}
          disabled={uploading || converting || !selectedFile}
        >
          {uploading ? <span className="spinner" /> : <IconSend size={16} />}
        </button>
      </div>

      {note && <p className="landing-note">{note}</p>}

      <div className="feature-grid">
        <FeatureCard
          icon={IconAlert}
          title="Clause risk detection"
          text="Classifies clauses as low, medium, or high risk using ML and legal keyword validation."
        />
        <FeatureCard
          icon={IconSparkle}
          title="Contract Q&A"
          text="Ask about summary, rent, termination, default, risky clauses, or missing fields."
        />
        <FeatureCard
          icon={IconFlag}
          title="Missing field review"
          text="Detects blank or incomplete areas such as dates, rent amount, parties, and lease duration."
        />
      </div>
    </section>
  );
}

function FeatureCard({ title, text, icon: Icon }) {
  return (
    <div className="feature-card">
      <div className="feature-card-icon">
        <Icon size={16} />
      </div>
      <h3>{title}</h3>
      <p>{text}</p>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* History                                                             */
/* ------------------------------------------------------------------ */

function HistoryView({
  savedContracts,
  historyLoading,
  loadContractHistory,
  openSavedContract,
  deleteSavedContract,
}) {
  return (
    <section className="page-card history-page">
      <div className="section-header">
        <div>
          <p className="eyebrow">Saved contracts</p>
          <h2>Contract history</h2>
          <p className="muted">Open previously uploaded contracts from your database.</p>
        </div>

        <button className="secondary-btn" onClick={loadContractHistory}>
          Refresh history
        </button>
      </div>

      {historyLoading && <p className="muted-text">Loading saved contracts...</p>}

      {!historyLoading && savedContracts.length === 0 && (
        <div className="empty-state">
          <h3>No saved contracts found</h3>
          <p>Upload a contract first, then it will appear here.</p>
        </div>
      )}

      {!historyLoading && savedContracts.length > 0 && (
        <div className="history-list">
          {savedContracts.map((contract) => (
            <div key={contract.contract_id} className="history-card">
              <div className="history-card-icon">
                <IconDoc size={18} />
              </div>

              <div className="history-card-body">
                <h3>{contract.filename}</h3>

                <p>
                  Uploaded:{" "}
                  {contract.created_at
                    ? new Date(contract.created_at).toLocaleString()
                    : "Unknown date"}
                </p>

                <p className="contract-id-text">ID: {contract.contract_id}</p>
              </div>

              <div className="history-actions">
                <button
                  className="primary-btn"
                  onClick={() => openSavedContract(contract.contract_id)}
                >
                  <IconFolderOpen size={14} />
                  Open
                </button>

                <button
                  className="danger-btn"
                  onClick={() => deleteSavedContract(contract.contract_id)}
                >
                  <IconTrash size={14} />
                  Delete
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

/* ------------------------------------------------------------------ */
/* Review workspace                                                    */
/* ------------------------------------------------------------------ */

function ReviewWorkspace({
  latestStructuredAnswer,
  setLatestStructuredAnswer,
  chatHistory,
  question,
  setQuestion,
  handleAskQuestion,
  selectedFile,
  handleFileChange,
  handleUpload,
  uploading,
  asking,
}) {
  const [answerCollapsed, setAnswerCollapsed] = useState(false);
  const [focusMode, setFocusMode] = useState(false);

  const quickPrompts = ["Summarize", "Risky clauses", "Payment terms", "Termination", "Missing info"];

  const promptMap = {
    Summarize: "Summarize this contract",
    "Risky clauses": "What are the risky clauses?",
    "Payment terms": "What are the payment terms?",
    Termination: "Explain the termination clause",
    "Missing info": "What information is missing?",
  };

  const hasAnswer = Boolean(latestStructuredAnswer);

  return (
    <section className="review-workspace-integrated">
      <div className="summary-hero compact-hero">
        <div>
          <p className="eyebrow">Contract review</p>
          <h1>Contract analysis workspace</h1>
          <p>
            Review risks, payment terms, termination clauses, and missing
            information from the uploaded contract.
          </p>
        </div>

        <div className="summary-actions">
          <label className="file-picker">
            <input type="file" accept="application/pdf" onChange={handleFileChange} />
            <IconPaperclip size={14} />
            {selectedFile ? selectedFile.name : "Replace PDF"}
          </label>

          <button onClick={handleUpload} disabled={uploading}>
            {uploading ? "Analyzing..." : "Re-analyze"}
          </button>
        </div>
      </div>

      <div className={`answer-card integrated-answer-card ${answerCollapsed ? "answer-collapsed" : ""}`}>
        <div className="answer-card-header integrated-answer-header">
          <div>
            <p className="eyebrow">Latest agent response</p>
            <h2>Contract agent response</h2>
          </div>

          <div className="answer-view-actions">
            <button
              type="button"
              className="small-soft-btn"
              onClick={() => setAnswerCollapsed(!answerCollapsed)}
              disabled={!hasAnswer}
            >
              <IconChevron size={14} style={{ transform: answerCollapsed ? "rotate(-90deg)" : "none" }} />
              {answerCollapsed ? "Expand" : "Collapse"}
            </button>

            <button
              type="button"
              className="small-primary-btn"
              onClick={() => setFocusMode(true)}
              disabled={!hasAnswer}
            >
              <IconExpand size={13} />
              Focus view
            </button>
          </div>
        </div>

        {!answerCollapsed && (
          <div className="answer-content integrated-answer-content">
            {latestStructuredAnswer ? (
              <StructuredAnswer data={latestStructuredAnswer} />
            ) : (
              <p>No response yet. Ask a contract-specific question below.</p>
            )}
          </div>
        )}

        {answerCollapsed && (
          <div className="collapsed-answer-preview">
            <p>
              Response is collapsed. Click <strong>Expand</strong> or{" "}
              <strong>Focus view</strong> to review the full analysis.
            </p>
          </div>
        )}

        <AskAgentBox
          question={question}
          setQuestion={setQuestion}
          handleAskQuestion={handleAskQuestion}
          asking={asking}
          quickPrompts={quickPrompts}
          promptMap={promptMap}
        />
      </div>

      {focusMode && (
        <div className="answer-focus-overlay">
          <div className="answer-focus-panel focus-with-chat">
            <div className="answer-focus-header">
              <div>
                <p className="eyebrow">Focused review</p>
                <h2>Contract agent workspace</h2>
              </div>

              <button type="button" className="chat-close-btn" onClick={() => setFocusMode(false)}>
                <IconX size={16} />
              </button>
            </div>

            <div className="focus-workspace-body">
              <div className="answer-focus-content focus-report-content">
                {latestStructuredAnswer ? (
                  <StructuredAnswer data={latestStructuredAnswer} />
                ) : (
                  <p>No response yet. Ask a contract-specific question first.</p>
                )}
              </div>

              <div className="focus-chat-panel">
                <div className="focus-chat-header">
                  <h3>Analysis chat</h3>
                  <p>Your previous questions stay here while the latest detailed answer appears on the left.</p>
                </div>

                <div className="focus-chat-history">
                  {chatHistory.map((message, index) => (
                    <ChatMessage
                      key={`${message.role}-${index}`}
                      message={message}
                      isActive={Boolean(message.structured) && message.structured === latestStructuredAnswer}
                      onSelect={() => message.structured && setLatestStructuredAnswer(message.structured)}
                    />
                  ))}
                </div>

                <AskAgentBox
                  question={question}
                  setQuestion={setQuestion}
                  handleAskQuestion={handleAskQuestion}
                  asking={asking}
                  quickPrompts={quickPrompts}
                  promptMap={promptMap}
                  compact
                />
              </div>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

function AskAgentBox({
  question,
  setQuestion,
  handleAskQuestion,
  asking,
  quickPrompts,
  promptMap,
  compact = false,
}) {
  return (
    <div className={`integrated-agent-box ${compact ? "compact-agent-box" : ""}`}>
      <div className="integrated-agent-heading">
        <div>
          <h3>Ask contract agent</h3>
          <p>Ask contract-specific questions. The detailed result appears in the analysis panel.</p>
        </div>
      </div>

      <div className="integrated-quick-prompts">
        {quickPrompts.map((prompt) => (
          <button
            key={prompt}
            type="button"
            onClick={() => handleAskQuestion(promptMap[prompt])}
            disabled={asking}
          >
            {prompt}
          </button>
        ))}
      </div>

      <div className="integrated-question-box">
        <input
          value={question}
          onChange={(event) => setQuestion(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === "Enter") {
              handleAskQuestion();
            }
          }}
          placeholder="Ask about risk, rent, termination, missing fields..."
          disabled={asking}
        />

        <button onClick={() => handleAskQuestion()} disabled={asking}>
          {asking ? <span className="spinner" /> : <IconSend size={15} />}
        </button>
      </div>
    </div>
  );
}

function ChatMessage({ message, isActive = false, onSelect }) {
  const clickable = Boolean(message.structured);
  const structured = message.structured;
  const previewText = structured?.summary || structured?.answer || message.text;
  const previewPoints = structured?.key_points?.slice(0, 2) || [];

  return (
    <button
      type="button"
      className={`focus-chat-message ${message.role} ${clickable ? "clickable" : ""} ${
        isActive ? "active" : ""
      }`}
      onClick={onSelect}
      disabled={!clickable}
    >
      <div className="focus-chat-label">
        {message.role === "user" ? "You" : "Contract Agent"}
        {structured?.risk_level && structured.risk_level !== "Unknown" && (
          <span className={`risk-pill ${structured.risk_level.toLowerCase()}`}>
            {structured.risk_level}
          </span>
        )}
      </div>

      {structured?.title && <strong className="focus-chat-title">{structured.title}</strong>}

      <p>{previewText}</p>

      {previewPoints.length > 0 && (
        <ul className="focus-chat-points">
          {previewPoints.map((point, index) => (
            <li key={index}>{point}</li>
          ))}
        </ul>
      )}

      {clickable && (
        <span className="focus-chat-hint">{isActive ? "Currently shown" : "View full answer →"}</span>
      )}
    </button>
  );
}

/* ------------------------------------------------------------------ */
/* Structured answer                                                   */
/* ------------------------------------------------------------------ */

function StructuredAnswer({ data }) {
  if (!data) return null;

  return (
    <div className="structured-answer">
      <div className="structured-top">
        <div>
          <p className="eyebrow">{data.answer_type || "Contract Review"}</p>
          <h2>{data.title || "Contract Agent Response"}</h2>
        </div>

        <RiskBadge risk={data.risk_level} />
      </div>

      {data.summary && (
        <div className="structured-card main-summary-card">
          <h3>Summary</h3>
          <p>{data.summary}</p>
        </div>
      )}

      {data.contract_details && Object.keys(data.contract_details).length > 0 && (
        <div className="structured-card">
          <h3>Contract details</h3>
          <div className="details-grid">
            {Object.entries(data.contract_details).map(([key, value]) => (
              <div key={key} className="detail-item">
                <span>{formatLabel(key)}</span>
                <strong>{Array.isArray(value) ? value.join(", ") : value || "Not specified"}</strong>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.key_points?.length > 0 && (
        <div className="structured-card">
          <h3>Key points</h3>
          <div className="structured-list">
            {data.key_points.map((point, index) => (
              <div key={index} className="structured-list-item">
                <span>✓</span>
                <p>{point}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {data.missing_fields?.length > 0 && (
        <div className="structured-card warning-card">
          <h3>Missing / incomplete information</h3>
          <div className="missing-chip-list">
            {data.missing_fields.map((field, index) => (
              <span key={index}>{field}</span>
            ))}
          </div>
        </div>
      )}

      {data.related_clauses?.length > 0 && (
        <div className="structured-card">
          <h3>Related clauses</h3>

          <div className="related-clause-list">
            {data.related_clauses.map((clause) => (
              <div key={clause.clause_number} className="related-clause-card">
                <div>
                  <strong>Clause {clause.clause_number}</strong>
                  <RiskBadge risk={clause.risk_level} small />
                </div>

                <p>{clause.preview}</p>

                {clause.risk_signals?.length > 0 && (
                  <small>Signals: {formatRiskSignals(clause.risk_signals)}</small>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="prototype-note">{data.disclaimer || "Prototype ML-based review. Not legal advice."}</div>
    </div>
  );
}

function RiskBadge({ risk, small = false }) {
  const safeRisk = risk || "Unknown";

  return (
    <span className={`structured-risk-badge ${safeRisk.toLowerCase()} ${small ? "small" : ""}`}>
      {safeRisk}
    </span>
  );
}

function formatLabel(value) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function formatRiskSignals(signals) {
  return signals
    .map((signal) => {
      if (typeof signal === "string") return signal;
      return signal.keyword || signal.reason || "Risk signal";
    })
    .join(", ");
}

/* ------------------------------------------------------------------ */
/* Right panel                                                         */
/* ------------------------------------------------------------------ */

function RightPanel({ analysisResult }) {
  const analysis = analysisResult?.analysis;
  const clauses = analysisResult?.clauses || [];

  const mediumCount = clauses.filter((clause) => clause.risk_level === "Medium").length;
  const highCount = clauses.filter((clause) => clause.risk_level === "High").length;
  const lowCount = clauses.filter((clause) => clause.risk_level === "Low").length;

  const riskyClauses = clauses.filter(
    (clause) => clause.risk_level === "High" || clause.risk_level === "Medium"
  );

  return (
    <aside className="right-panel">
      <h3>Contract intelligence</h3>

      {!analysisResult && (
        <div className="empty-panel">
          <BrandMark size={30} />
          <p>No contract uploaded yet.</p>
          <span>Upload a PDF to see risk score, clause count, and missing fields.</span>
        </div>
      )}

      {analysisResult && (
        <>
          <div className="risk-score-card">
            <span>Overall risk</span>
            <strong className={`risk-text ${analysis?.overall_risk?.toLowerCase()}`}>
              {analysis?.overall_risk || "Unknown"}
            </strong>
          </div>

          <div className="stats-grid">
            <StatCard label="Total clauses" value={analysis?.total_clauses || 0} />
            <StatCard label="Risky clauses" value={analysis?.risky_clauses || 0} />
            <StatCard label="High risk" value={highCount} />
            <StatCard label="Medium risk" value={mediumCount} />
            <StatCard label="Low risk" value={lowCount} />
            <StatCard label="Missing fields" value={analysis?.missing_fields?.length || 0} />
          </div>

          <div className="panel-section">
            <h4>Missing information</h4>

            {analysis?.missing_fields?.length ? (
              analysis.missing_fields.slice(0, 6).map((field, index) => (
                <div key={index} className="mini-warning">
                  {field}
                </div>
              ))
            ) : (
              <p className="muted">No major missing fields detected.</p>
            )}
          </div>

          <div className="panel-section">
            <h4>Top risk clauses</h4>

            {riskyClauses.slice(0, 4).map((clause) => (
              <div key={clause.clause_number} className="mini-clause">
                <div>
                  <strong>Clause {clause.clause_number}</strong>
                  <span className={`risk-pill ${clause.risk_level.toLowerCase()}`}>
                    {clause.risk_level}
                  </span>
                </div>

                <p>{clause.preview}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </aside>
  );
}

function StatCard({ label, value }) {
  return (
    <div className="stat-card">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Clause / risk / missing views                                       */
/* ------------------------------------------------------------------ */

function ClausesView({ analysisResult }) {
  const clauses = analysisResult?.clauses || [];

  return (
    <section className="page-card">
      <div className="section-header">
        <div>
          <p className="eyebrow">Clause library</p>
          <h2>Extracted contract clauses</h2>
        </div>

        <span>{clauses.length} clauses</span>
      </div>

      <div className="clause-list">
        {clauses.map((clause) => (
          <ClauseCard key={clause.clause_number} clause={clause} />
        ))}
      </div>
    </section>
  );
}

function RiskView({ analysisResult }) {
  const clauses = analysisResult?.clauses || [];

  const riskyClauses = clauses.filter(
    (clause) => clause.risk_level === "Medium" || clause.risk_level === "High"
  );

  return (
    <section className="page-card">
      <div className="section-header">
        <div>
          <p className="eyebrow">Risk analysis</p>
          <h2>Risky clauses</h2>
        </div>

        <span>{riskyClauses.length} risky clauses</span>
      </div>

      <div className="clause-list">
        {riskyClauses.map((clause) => (
          <ClauseCard key={clause.clause_number} clause={clause} />
        ))}
      </div>
    </section>
  );
}

function MissingFieldsView({ analysisResult }) {
  const missingFields = analysisResult?.analysis?.missing_fields || [];

  return (
    <section className="page-card">
      <div className="section-header">
        <div>
          <p className="eyebrow">Completeness review</p>
          <h2>Missing or incomplete information</h2>
        </div>

        <span>{missingFields.length} items</span>
      </div>

      <div className="missing-grid">
        {missingFields.length ? (
          missingFields.map((field, index) => (
            <div key={index} className="missing-card">
              <span>Needs review</span>
              <strong>{field}</strong>
              <p>This field appears blank, incomplete, or not clearly stated in the uploaded contract.</p>
            </div>
          ))
        ) : (
          <div className="empty-state">No major missing fields were detected by the prototype model.</div>
        )}
      </div>
    </section>
  );
}

function ClauseCard({ clause }) {
  const reason =
    clause.risk_reason ||
    (clause.risk_signals?.length > 0
      ? clause.risk_signals
          .map((signal) => {
            if (typeof signal === "string") return signal;
            return signal.keyword || signal.reason || "Risk signal";
          })
          .join(", ")
      : "No major risk keyword detected");

  return (
    <article className="clause-card">
      <div className="clause-card-header">
        <div>
          <h3>
            Clause {clause.clause_number}
            {clause.clause_type ? ` — ${clause.clause_type}` : ""}
          </h3>
          <p>Reason: {reason}</p>
        </div>

        <span className={`risk-pill ${clause.risk_level?.toLowerCase()}`}>
          {clause.risk_level}
        </span>
      </div>

      <p className="clause-preview">{clause.preview}</p>

      <div className="clause-intelligence-grid">
        <div>
          <span>Clause type</span>
          <strong>{clause.clause_type || "Not classified"}</strong>
        </div>

        <div>
          <span>Party affected</span>
          <strong>{clause.party_affected || "Not clearly specified"}</strong>
        </div>

        <div>
          <span>ML prediction</span>
          <strong>{clause.ml_prediction || "Unknown"}</strong>
        </div>

        <div>
          <span>Risk level</span>
          <strong>{clause.risk_level || "Unknown"}</strong>
        </div>
      </div>

      {clause.recommended_action && (
        <div className="recommended-action-box">
          <span>Recommended action</span>
          <p>{clause.recommended_action}</p>
        </div>
      )}
    </article>
  );
}

export default App;

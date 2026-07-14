from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import fitz  # PyMuPDF
import re
import uuid
import os
import joblib
import sqlite3
import json
from datetime import datetime
from typing import Any

try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None  # type: ignore[assignment]

load_dotenv()

# ============================================================
# APP SETUP
# ============================================================

app = FastAPI(title="Legal Contract ML Agent SaaS")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# TEMPORARY MEMORY STORE
# ============================================================

CONTRACT_STORE: dict[str, dict[str, Any]] = {}

# ============================================================
# SQLITE DATABASE SETUP
# ============================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "contracts.db")


def get_db_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS contracts (
            contract_id TEXT PRIMARY KEY,
            filename TEXT,
            content_type TEXT,
            contract_text TEXT,
            clauses_json TEXT,
            analysis_json TEXT,
            created_at TEXT
        )
    """)

    conn.commit()
    conn.close()


def save_contract_to_db(
    contract_id: str,
    filename: str | None,
    content_type: str | None,
    contract_text: str,
    clauses: list[dict[str, Any]],
    analysis: dict[str, Any],
) -> None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO contracts (
            contract_id,
            filename,
            content_type,
            contract_text,
            clauses_json,
            analysis_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        contract_id,
        filename,
        content_type,
        contract_text,
        json.dumps(clauses),
        json.dumps(analysis),
        datetime.now().isoformat()
    ))

    conn.commit()
    conn.close()


def load_contract_from_db(contract_id: str) -> dict[str, Any] | None:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT * FROM contracts
        WHERE contract_id = ?
    """, (contract_id,))

    row = cursor.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "filename": row["filename"],
        "text": row["contract_text"],
        "clauses": json.loads(row["clauses_json"]),
        "analysis": json.loads(row["analysis_json"]),
    }

def delete_contract_from_db(contract_id: str) -> bool:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM contracts WHERE contract_id = ?",
        (contract_id,)
    )

    deleted_count = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted_count > 0

def list_contracts_from_db() -> list[dict[str, Any]]:
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT contract_id, filename, content_type, created_at
        FROM contracts
        ORDER BY created_at DESC
    """)

    rows = cursor.fetchall()
    conn.close()

    return [
        {
            "contract_id": row["contract_id"],
            "filename": row["filename"],
            "content_type": row["content_type"],
            "created_at": row["created_at"],
        }
        for row in rows
    ]


init_db()

@app.delete("/contracts/{contract_id}")
def delete_contract(contract_id: str) -> dict[str, Any]:
    deleted = delete_contract_from_db(contract_id)

    if not deleted:
        return {
            "error": True,
            "message": "Contract not found.",
            "contract_id": contract_id,
        }

    if contract_id in CONTRACT_STORE:
        del CONTRACT_STORE[contract_id]

    return {
        "error": False,
        "message": "Contract deleted successfully.",
        "contract_id": contract_id,
    }

# ============================================================
# ML MODEL LOADING
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Risk-classification model files
MODEL_PATH = os.path.join(BASE_DIR, "ml_model", "risk_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "ml_model", "vectorizer.pkl")

# Clause-type classification model files
CLAUSE_TYPE_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml_model",
    "clause_type_model.pkl",
)
CLAUSE_TYPE_VECTORIZER_PATH = os.path.join(
    BASE_DIR,
    "ml_model",
    "clause_type_vectorizer.pkl",
)

risk_model = None
vectorizer = None
clause_type_model = None
clause_type_vectorizer = None

try:
    risk_model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print("Risk model loaded successfully.")
except Exception as error:
    print("Risk model not loaded. Using rule-based risk fallback only.")
    print("Risk model error:", error)

try:
    clause_type_model = joblib.load(CLAUSE_TYPE_MODEL_PATH)
    clause_type_vectorizer = joblib.load(CLAUSE_TYPE_VECTORIZER_PATH)
    print("Clause-type model loaded successfully.")
except Exception as error:
    print("Clause-type model not loaded. Using rule-based clause-type fallback.")
    print("Clause-type model error:", error)


# ============================================================
# GEMINI-BACKED GENERAL LEGAL Q&A
# ============================================================
# This backend no longer uses Ollama.
# Set your Gemini key in CMD before starting the backend:
# set GEMINI_API_KEY=YOUR_REAL_GEMINI_KEY

GEMINI_MODEL = os.getenv("GEMINI_MODEL",  "gemini-2.5-flash-lite")

LEGAL_ASSISTANT_SYSTEM_PROMPT = """You are a legal-contract assistant embedded in a contract review tool.

Rules you must follow:
- Answer using the provided contract text whenever the question is about this specific contract.
- You may also answer general legal knowledge questions, such as what a legal term means, how a type of clause usually works, or what rights a party typically has.
- If the question has nothing to do with law, contracts, leases, clauses, business agreements, or legal risk, say plainly that you can only help with legal and contract-related questions.
- Do not invent or assume contract terms that are not present in the provided text.
- If the contract is silent on something, say so plainly instead of guessing.
- Be clear and concise.
- Use plain language.
- Always make clear that this is general information, not legal advice.
"""


def call_gemini(prompt: str) -> str:
    """Calls Gemini using GEMINI_API_KEY from the environment.

    The function supports the newer google-genai package and also falls back
    to google-generativeai if that is the package installed in your venv.
    """

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise ValueError(
            "Gemini API key is missing. In CMD, run: "
            "set GEMINI_API_KEY=YOUR_REAL_KEY, then restart uvicorn."
        )

    # New SDK: pip install google-genai
    try:
        from google import genai

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        answer_text = (getattr(response, "text", None) or "").strip()

        if answer_text:
            return answer_text

        raise ValueError("Empty response from Gemini.")

    except ImportError:
        pass

    # Old SDK fallback: pip install google-generativeai
    try:
        import google.generativeai as generativeai

        generativeai.configure(api_key=api_key)
        model = generativeai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)

        answer_text = (getattr(response, "text", None) or "").strip()

        if answer_text:
            return answer_text

        raise ValueError("Empty response from Gemini.")

    except ImportError as import_error:
        raise ImportError(
            "Gemini SDK is not installed. Run: "
            "pip install google-genai"
        ) from import_error


def build_llm_general_answer(
    contract_data: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    contract_text = (contract_data.get("text") or "").strip()
    truncated_text = contract_text[:12000]

    prompt = f"""
{LEGAL_ASSISTANT_SYSTEM_PROMPT}

Contract text may be truncated for length:
---
{truncated_text}
---

User question:
{question}

Answer:
"""

    try:
        answer_text = call_gemini(prompt)

    except Exception as error:
        print("Gemini general-answer error:", error)
        fallback = build_structured_general(contract_data, question)
        fallback["summary"] = (
            "Gemini is temporarily unavailable or the API key is not configured correctly. "
            "I am showing the standard contract response instead. Check GEMINI_API_KEY, "
            "make sure the key starts with AIza, and restart the backend."
        )
        fallback["answer"] = fallback["summary"]
        return fallback

    return {
        "answer_type": "general_legal",
        "title": "Legal Q&A",
        "risk_level": get_overall_risk(contract_data["clauses"]),
        "summary": answer_text,
        "key_points": [],
        "missing_fields": [],
        "related_clauses": [],
        "contract_details": {},
        "answer": answer_text,
        "disclaimer": (
            "This is general information, not legal advice. Consult a licensed "
            "attorney for advice about your specific situation."
        ),
    }


# ============================================================
# REQUEST MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str
    contract_id: str


# ============================================================
# CLEANING FUNCTIONS
# ============================================================

def extract_pdf_content(file_bytes: bytes) -> dict[str, Any]:
    # Extract selectable text first, then OCR pages with little or no text.
    pdf = fitz.open(stream=file_bytes, filetype="pdf")

    extracted_pages = []
    text_pages = 0
    ocr_pages = 0
    failed_ocr_pages = []

    for page_number, page in enumerate(pdf, start=1):
        page_text = (page.get_text("text") or "").strip()
        compact_text = re.sub(r"\s+", "", page_text)

        if len(compact_text) >= 40:
            extracted_pages.append(page_text)
            text_pages += 1
            continue

        if pytesseract is None or Image is None:
            failed_ocr_pages.append(page_number)
            extracted_pages.append(page_text)
            continue

        try:
            pixmap = page.get_pixmap(
                matrix=fitz.Matrix(2.0, 2.0),
                alpha=False,
            )

            image = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

            ocr_text = pytesseract.image_to_string(
                image,
                config="--oem 3 --psm 6",
            ).strip()

            extracted_pages.append(ocr_text)
            ocr_pages += 1

        except Exception as error:
            print(f"OCR failed on page {page_number}: {error}")
            failed_ocr_pages.append(page_number)
            extracted_pages.append(page_text)

    page_count = len(extracted_pages)
    pdf.close()

    if text_pages > 0 and ocr_pages > 0:
        extraction_method = "mixed"
    elif ocr_pages > 0:
        extraction_method = "ocr"
    else:
        extraction_method = "text"

    return {
        "text": "\n\n".join(extracted_pages).strip(),
        "page_count": page_count,
        "text_pages": text_pages,
        "ocr_pages": ocr_pages,
        "failed_ocr_pages": failed_ocr_pages,
        "extraction_method": extraction_method,
    }


def extract_text_from_pdf(file_bytes: bytes) -> str:
    extracted_text = extract_pdf_content(file_bytes).get("text", "")
    return str(extracted_text)


def normalize_raw_text(text: str) -> str:
    if not text:
        return ""

    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_irrelevant_sections(text: str) -> str:
    """
    Removes legal website/template explanation sections and keeps the actual contract draft.
    """

    if not text:
        return ""

    start_patterns = [
        r"Draft of Deed of Lease",
        r"This Deed of Lease is made",
        r"THIS DEED OF LEASE",
        r"NOW This Deed Witnesseth",
        r"NOW THIS DEED WITNESSETH",
    ]

    start_index = None

    for pattern in start_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            start_index = match.start()
            break

    if start_index is not None:
        text = text[start_index:]

    stop_patterns = [
        r"Documents Required",
        r"Procedure for",
        r"Legal Considerations",
        r"How can a lawyer help",
        r"What is Deed of Family Trust",
        r"What is Lease Deed",
        r"Why is Lease Deed",
    ]

    stop_indexes = []

    for pattern in stop_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            stop_indexes.append(match.start())

    if stop_indexes:
        text = text[:min(stop_indexes)]

    return text.strip()


def clean_display_text(text: str) -> str:
    """
    Cleans text for frontend display.
    This improves readability without changing legal meaning.
    """

    if not text:
        return ""

    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)

    replacements = {
        "convenants": "covenants",
        "If the-ground": "If the ground",
        "if the-ground": "if the ground",
        "an arrears": "in arrears",
        "reas...": "reason",
    }

    for wrong, right in replacements.items():
        text = text.replace(wrong, right)

    # Replace visible blanks with cleaner wording
    text = re.sub(r"\.{3,}", " [not filled in] ", text)
    text = re.sub(r"…+", " [not filled in] ", text)
    text = text.replace("20___", "[year not filled in]")
    text = re.sub(r"_{2,}", " [not filled in] ", text)

    # Clean duplicate placeholder wording
    text = re.sub(r"(\[not filled in\]\s*){2,}", "[not filled in] ", text)
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def smart_preview(text: str, max_length: int = 420) -> str:
    """
    Creates a clean frontend preview without cutting words badly.
    """

    text = clean_display_text(text)

    if len(text) <= max_length:
        return text

    cut = text[:max_length]

    # Prefer ending at a sentence boundary.
    sentence_end = max(
        cut.rfind(". "),
        cut.rfind("; "),
        cut.rfind(": ")
    )

    if sentence_end > 180:
        return cut[:sentence_end + 1].strip()

    # Otherwise end at a comma if available.
    comma_end = cut.rfind(", ")

    if comma_end > 220:
        return cut[:comma_end + 1].strip()

    # Otherwise end at the last complete word.
    last_space = cut.rfind(" ")

    if last_space > 180:
        return cut[:last_space].strip() + "..."

    return cut.strip() + "..."


def has_blank_or_placeholder(text: str) -> bool:
    if not text:
        return True

    patterns = [
        "[not filled in]",
        "[year not filled in]",
        "[blank]",
        "[year blank]",
        "....",
        "...",
        "___",
        "_____",
        "20___",
    ]

    lower_text = text.lower()

    return any(pattern.lower() in lower_text for pattern in patterns)


def value_or_not_specified(value: str) -> str:
    if not value or has_blank_or_placeholder(value):
        return "Not specified in the document."
    return clean_display_text(value)


# ============================================================
# CLAUSE SPLITTING
# ============================================================

def split_into_clauses(text: str) -> list[str]:
    """
    Splits contract text into meaningful legal clauses/sections.

    This version avoids splitting on every 'The Lessor' or 'The Lessee'
    because that was creating broken fragments.
    """

    text = normalize_raw_text(text)

    # Convert excessive newlines into single spaces first.
    text = re.sub(r"\s+", " ", text).strip()

    # Add split markers before major numbered clauses only.
    # Example: 1. In pursuance..., 2. The Lessee..., 3. The Lessor...
    text = re.sub(
        r"\s(?=(?:\d+)\.\s)",
        "\n",
        text
    )

    # Add split markers before major legal sections.
    major_section_patterns = [
        r"NOW\s+This\s+Deed\s+Witnesseth",
        r"NOW\s+THIS\s+DEED\s+WITNESSETH",
        r"IN\s+WITNESS\s+WHEREOF",
        r"In\s+witness\s+whereof",
        r"THE\s+SCHEDULE\s+ABOVE\s+REFERRED\s+TO",
        r"The\s+Schedule\s+Above\s+Referred\s+To",
    ]

    for pattern in major_section_patterns:
        text = re.sub(
            rf"\s(?=({pattern}))",
            "\n",
            text,
            flags=re.IGNORECASE
        )

    raw_sections = [section.strip() for section in text.split("\n") if section.strip()]

    clauses = []

    for section in raw_sections:
        section = clean_display_text(section)

        if len(section) < 60:
            continue

        # If a numbered section contains lettered subclauses, keep them together
        # unless the section is extremely long.
        if len(section) > 2500:
            subclauses = split_large_clause_by_lettered_parts(section)
            clauses.extend(subclauses)
        else:
            clauses.append(section)

    # Final cleanup: remove duplicates and tiny broken fragments
    cleaned_clauses = []
    seen = set()

    for clause in clauses:
        clause = clean_display_text(clause)

        if len(clause) < 60:
            continue

        # Avoid fragments that are clearly not useful standalone clauses
        bad_endings = [
            "hereinafter called '",
            "in the manner hereinafter appearin",
            "regularl",
            "notic",
            "withhel",
        ]

        if any(clause.lower().endswith(item.lower()) for item in bad_endings):
            continue

        key = clause[:120].lower()

        if key not in seen:
            cleaned_clauses.append(clause)
            seen.add(key)

    return cleaned_clauses

def split_large_clause_by_lettered_parts(section: str) -> list[str]:
    """
    Splits very large numbered clauses into lettered subclauses only when needed.
    Example: a. To pay rent...
             b. To bear taxes...
             c. To maintain premises...
    """

    section = clean_display_text(section)

    # Split only before real lettered subclauses like "a. ", "b. ", "c. "
    parts = re.split(r"(?=\s[a-h]\.\s)", section)

    cleaned_parts = []

    for part in parts:
        part = clean_display_text(part)

        if len(part) < 80:
            continue

        cleaned_parts.append(part)

    if cleaned_parts:
        return cleaned_parts

    return [section]


# ============================================================
# RISK ANALYSIS
# ============================================================

RISK_KEYWORDS = {
        "arrears": "Rent arrears/default wording detected.",
    "re-entry": "Re-entry remedy wording detected.",
    "reentry": "Re-entry remedy wording detected.",
    "without prejudice": "Without-prejudice rights wording detected.",
    "not entitled": "Restriction or limitation wording detected.",
    "shall not be entitled": "Restriction or limitation wording detected.",
    "without payment of any compensation": "No-compensation wording detected.",
    "assign": "Assignment restriction wording detected.",
    "sublet": "Subletting restriction wording detected.",
    "mortgage": "Mortgage/transfer restriction wording detected.",
    "termination": "Termination wording detected.",
    "earlier determination": "Early termination wording detected.",
    "default": "Default wording detected.",
    "breach": "Breach wording detected.",
    "penalty": "Penalty wording detected.",
    "liability": "Liability wording detected.",
    "indemnify": "Indemnity wording detected.",
    "indemnification": "Indemnity wording detected.",
    "indemnified": "Indemnity wording detected.",
    "damages": "Damages wording detected.",
    "late payment": "Late payment wording detected.",
    "dispute": "Dispute wording detected.",
    "waiver": "Waiver wording detected.",
    "waive": "Waiver wording detected.",
    "without notice": "No-notice wording detected.",
    "liquidated damages": "Liquidated damages wording detected.",
    "legal costs": "Legal cost wording detected.",
    "hold harmless": "Hold harmless wording detected.",
    "arbitration": "Arbitration wording detected.",
    "confidential": "Confidentiality wording detected.",
    "non-compete": "Non-compete wording detected.",
}


PAYMENT_KEYWORDS = [
    "rent",
    "monthly rent",
    "ground rent",
    "payment",
    "pay",
    "paid",
    "payable",
    "due",
    "interest",
    "deposit",
    "fee",
    "charges",
    "deductions",
    "arrears",
]

TERMINATION_KEYWORDS = [
    "termination",
    "terminate",
    "earlier determination",
    "expiration",
    "default",
    "breach",
    "arrears",
    "re-entry",
    "reentry",
    "not paid",
    "failure to pay",
    "failed to comply",
    "covenants",
    "demise shall absolutely determine",
    "notice in writing",
]


def predict_ml_risk(clause_text: str) -> str:
    if risk_model is None or vectorizer is None:
        return "Unknown"

    try:
        transformed = vectorizer.transform([clause_text])
        prediction = risk_model.predict(transformed)[0]
        return str(prediction)
    except Exception:
        return "Unknown"


def detect_risk_signals(clause_text: str) -> list[dict[str, str]]:
    found = []

    lower_text = clause_text.lower()

    for keyword, reason in RISK_KEYWORDS.items():
        if keyword in lower_text:
            found.append({
                "keyword": keyword,
                "reason": reason,
            })

    return found


def final_risk_level(ml_prediction: str, risk_signals: list, clause_text: str) -> str:
    lower_text = clause_text.lower()

    high_risk_terms = [
        "unlimited liability",
        "without notice",
        "liquidated damages",
        "hold harmless",
        "indemnify",
        "indemnification",
        "indemnified",
        "penalty",
    ]

    medium_risk_terms = [
        "termination",
        "earlier determination",
        "default",
        "breach",
        "damages",
        "late payment",
        "dispute",
        "waiver",
        "legal costs",
    ]

    if any(term in lower_text for term in high_risk_terms):
        return "High"

    if any(term in lower_text for term in medium_risk_terms):
        return "Medium"

    if ml_prediction in ["High", "Medium"] and risk_signals:
        return ml_prediction

    return "Low"


def predict_clause_type_ml(clause_text: str) -> dict:
    """
    Predicts a clause category using the trained TF-IDF vectorizer and
    Logistic Regression clause-type classifier.

    The model and vectorizer were saved separately, so the clause text
    must be transformed before prediction.
    """

    fallback = {
        "clause_type": "General Contract Clause",
        "clause_type_confidence": 0.0,
        "clause_type_source": "fallback",
    }

    if not isinstance(clause_text, str):
        return fallback

    cleaned_text = clause_text.strip()

    if not cleaned_text:
        return fallback

    if clause_type_model is None or clause_type_vectorizer is None:
        return fallback

    try:
        transformed = clause_type_vectorizer.transform([cleaned_text])
        prediction = clause_type_model.predict(transformed)[0]
        probabilities = clause_type_model.predict_proba(transformed)[0]
        confidence = float(probabilities.max())

        return {
            "clause_type": str(prediction),
            "clause_type_confidence": round(confidence, 4),
            "clause_type_source": "ml",
        }

    except Exception as error:
        print("Clause-type prediction failed:", error)
        return fallback



# ============================================================
# PHASE 6 — VALIDATED HYBRID CLAUSE-TYPE CLASSIFIER
# ============================================================

HYBRID_ML_CONFIDENCE_THRESHOLD = 0.40

HYBRID_DISTINCTIVE_RULE_LABELS = {
    "Signature / Execution",
    "Quiet Enjoyment",
    "Assignment / Subletting",
    "Insurance",
    "Liability / Indemnity",
    "Governing Law",
    "Dispute Resolution",
    "Confidentiality",
}

HYBRID_ALLOWED_LABELS = {
    "Alterations / Improvements",
    "Assignment / Subletting",
    "Confidentiality",
    "Dispute Resolution",
    "General Obligation",
    "Governing Law",
    "Insurance",
    "Lease Grant",
    "Liability / Indemnity",
    "Notice",
    "Other / Unknown",
    "Payment / Rent",
    "Possession / Surrender",
    "Property / Premises",
    "Quiet Enjoyment",
    "Repairs / Maintenance",
    "Signature / Execution",
    "Taxes / Utilities",
    "Term and Renewal",
    "Termination / Default",
    "Use of Premises",
    "Warranties / Representations",
}


def detect_high_precision_clause_rule(clause_text: str) -> dict:
    if not isinstance(clause_text, str) or not clause_text.strip():
        return {"rule_label": None, "rule_matches": []}

    text = clause_text.lower()

    rule_patterns = [
        ("Signature / Execution", [
            r"\bin witness whereof\b",
            r"\bsigned and delivered\b",
            r"\bexecuted by\b",
            r"\belectronic signatures?\b",
            r"\bcounterparts?\b",
            r"\bwitnessed by\b",
            r"\bin the presence of\b",
        ]),
        ("Quiet Enjoyment", [
            r"\bquiet enjoyment\b",
            r"\bpeaceably and quietly\b",
            r"\bquietly hold\b",
            r"\bquietly enjoy\b",
            r"\bwithout interruption\b",
            r"\bwithout disturbance\b",
        ]),
        ("Assignment / Subletting", [
            r"\bshall not assign\b",
            r"\bnot to assign\b",
            r"\bassignment\b",
            r"\bsublet(?:ting)?\b",
            r"\bunderlet\b",
            r"\bpart with possession\b",
        ]),
        ("Insurance", [
            r"\bcommercial general liability insurance\b",
            r"\brenter'?s insurance\b",
            r"\bproperty insurance\b",
            r"\badditional insured\b",
            r"\binsured risks?\b",
            r"\binsurance premium\b",
        ]),
        ("Liability / Indemnity", [
            r"\bindemnif(?:y|ied|ication)\b",
            r"\bhold harmless\b",
            r"\blimitation of liability\b",
            r"\bunlimited liability\b",
        ]),
        ("Termination / Default", [
            r"\bevent of default\b",
            r"\bright of re-entry\b",
            r"\bright of reentry\b",
            r"\bre-entry\b",
            r"\breentry\b",
            r"\bearlier determination\b",
            r"\bterminate this agreement\b",
            r"\btermination of this lease\b",
        ]),
        ("Possession / Surrender", [
            r"\bsurrender the premises\b",
            r"\byield up the premises\b",
            r"\bvacant possession\b",
            r"\bdeliver possession\b",
            r"\breturn possession\b",
            r"\bholding over\b",
        ]),
        ("Alterations / Improvements", [
            r"\bstructural alteration\b",
            r"\btenant improvements?\b",
            r"\badditions or alterations\b",
            r"\binstall fixtures?\b",
            r"\bapproved plans\b",
        ]),
        ("Repairs / Maintenance", [
            r"\bgood and substantial repair\b",
            r"\btenantable repair\b",
            r"\bstructural repairs?\b",
            r"\bkeep the premises clean\b",
            r"\bmaintain the premises\b",
        ]),
        ("Taxes / Utilities", [
            r"\bproperty taxes?\b",
            r"\breal estate taxes?\b",
            r"\brates and taxes\b",
            r"\butility charges?\b",
            r"\bwater charges?\b",
            r"\belectricity charges?\b",
            r"\bgas charges?\b",
        ]),
        ("Governing Law", [
            r"\bgoverned by the laws?\b",
            r"\bgoverning law\b",
            r"\bchoice of law\b",
        ]),
        ("Dispute Resolution", [
            r"\barbitration\b",
            r"\bmediation\b",
            r"\bwaiver of jury trial\b",
        ]),
        ("Confidentiality", [
            r"\bconfidential information\b",
            r"\bnon-disclosure\b",
            r"\bnondisclosure\b",
            r"\bproprietary information\b",
        ]),
        ("Notice", [
            r"\bnotice shall be in writing\b",
            r"\bcertified mail\b",
            r"\bregistered post\b",
            r"\bformal notice\b",
        ]),
        ("Lease Grant", [
            r"\bhereby leases\b",
            r"\bhereby lets\b",
            r"\bhereby demises\b",
            r"\bto hold the demised premises\b",
            r"\baccepts the lease\b",
        ]),
        ("Term and Renewal", [
            r"\boption to renew\b",
            r"\brenewal term\b",
            r"\binitial term\b",
            r"\bfixed term\b",
        ]),
        ("Payment / Rent", [
            r"\bmonthly rent\b",
            r"\bannual rent\b",
            r"\bbase rent\b",
            r"\bsecurity deposit\b",
            r"\badditional rent\b",
        ]),
        ("Use of Premises", [
            r"\bused only for\b",
            r"\bpermitted use\b",
            r"\bresidential purposes only\b",
            r"\bbusiness purposes only\b",
        ]),
    ]

    matched_labels = []

    for label, patterns in rule_patterns:
        if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in patterns):
            matched_labels.append(label)

    unique_matches = list(dict.fromkeys(matched_labels))

    return {
        "rule_label": unique_matches[0] if len(unique_matches) == 1 else None,
        "rule_matches": unique_matches,
    }


def predict_clause_type_hybrid(clause_text: str) -> dict:
    ml_result = predict_clause_type_ml(clause_text)
    rule_result = detect_high_precision_clause_rule(clause_text)

    ml_label = str(ml_result.get("clause_type", "")).strip()
    ml_confidence = float(
        ml_result.get("clause_type_confidence", 0.0) or 0.0
    )

    rule_label = rule_result["rule_label"]
    rule_matches = rule_result["rule_matches"]

    ml_is_valid = ml_label in HYBRID_ALLOWED_LABELS
    rule_is_valid = rule_label in HYBRID_ALLOWED_LABELS

    use_rule = (
        rule_is_valid
        and (
            not ml_is_valid
            or ml_confidence < HYBRID_ML_CONFIDENCE_THRESHOLD
            or rule_label in HYBRID_DISTINCTIVE_RULE_LABELS
        )
    )

    if use_rule:
        final_label = rule_label
        source = "hybrid_rule"
    elif ml_is_valid:
        final_label = ml_label
        source = "hybrid_ml"
    elif rule_is_valid:
        final_label = rule_label
        source = "hybrid_rule_fallback"
    else:
        final_label = "Other / Unknown"
        source = "hybrid_fallback"

    return {
        "clause_type": final_label,
        "clause_type_confidence": round(ml_confidence, 4),
        "clause_type_source": source,
        "ml_clause_type": ml_label if ml_is_valid else None,
        "rule_clause_type": rule_label,
        "rule_matches": rule_matches,
        "hybrid_threshold": HYBRID_ML_CONFIDENCE_THRESHOLD,
    }

def detect_clause_type(clause_text: str) -> str:
    """
    Detect a clause type with lease-aware rule priority.

    Specific legal concepts are checked before broad terms such as
    "premises", "building", "rent", "lessor", and "lessee".
    This prevents most lease clauses from collapsing into
    Property / Premises or Parties / Identity.
    """

    text = clause_text.lower()

    # 1. Signature and execution language
    if any(term in text for term in [
        "in witness whereof",
        "signed and delivered",
        "signature",
        "executed by",
        "counterpart",
        "witnessed by",
        "in the presence of",
    ]):
        return "Signature / Execution"

    # 2. Quiet enjoyment
    if any(term in text for term in [
        "quiet enjoyment",
        "peaceably and quietly",
        "quietly hold",
        "quietly enjoy",
        "without interruption",
        "without disturbance",
        "without eviction",
        "peaceably hold",
    ]):
        return "Quiet Enjoyment"

    # 3. Assignment and subletting restrictions
    if any(term in text for term in [
        "shall not assign",
        "not to assign",
        "assign mortgage",
        "assign, mortgage",
        "assignment",
        "sublet",
        "subletting",
        "part with possession",
        "transfer this agreement",
        "transfer the lease",
    ]):
        return "Assignment / Subletting"

    # 4. Repairs and maintenance
    # Repair duties are checked before alteration permissions so a mixed clause
    # beginning with a maintenance obligation keeps Repairs / Maintenance as
    # its primary label. Alteration language can still appear as a secondary label.
    if any(term in text for term in [
        "good repair",
        "tenantable repair",
        "tenantable repairs",
        "repair and condition",
        "structural repairs",
        "maintenance",
        "maintain the premises",
        "keep the premises clean",
        "damage requiring",
        "service the equipment",
    ]):
        return "Repairs / Maintenance"

    # 5. Alterations and improvements
    if any(term in text for term in [
        "alteration",
        "alterations",
        "addition or alteration",
        "additions or alterations",
        "structural alteration",
        "improvement",
        "improvements",
        "construct any new building",
        "construct any new buildings",
        "demolish any existing building",
        "approved plans",
        "install fixtures",
    ]):
        return "Alterations / Improvements"

    # 6. Use of premises
    if any(term in text for term in [
        "used only for",
        "use the premises",
        "use or permit to be used",
        "lawful purposes",
        "permitted use",
        "illegal or hazardous activity",
        "solely as an office",
        "residential purposes",
        "business purposes",
        "change the permitted use",
    ]):
        return "Use of Premises"

    # 7. Possession and surrender
    if any(term in text for term in [
        "surrender the premises",
        "return possession",
        "vacant possession",
        "deliver possession",
        "yield up the premises",
        "remove its personal property",
        "all buildings and structures shall vest",
        "shall automatically vest in the lessor",
        "holding over",
        "hand over all keys",
    ]):
        return "Possession / Surrender"

    # 8. Taxes and utilities
    if any(term in text for term in [
        "property taxes",
        "municipal taxes",
        "real property taxes",
        "rates and taxes",
        "assessment duties",
        "business rates",
        "electricity charges",
        "water charges",
        "gas charges",
        "utility charges",
        "utilities",
        "sewer charges",
        "refuse charges",
    ]):
        return "Taxes / Utilities"

    # 9. Term and renewal
    if any(term in text for term in [
        "initial term",
        "renewal term",
        "option to renew",
        "automatically renew",
        "renew the lease",
        "successive one-year periods",
        "fixed term",
        "extension of the term",
        "holding over after expiration",
    ]):
        return "Term and Renewal"

    # 10. Lease grant
    if any(term in text for term in [
        "hereby leases",
        "hereby lets",
        "hereby demises",
        "grants the lessee",
        "grants the tenant",
        "leasehold interest",
        "accepts the lease",
        "takes the premises",
        "to hold the demised premises",
    ]):
        return "Lease Grant"

    # 11. Default, forfeiture, re-entry and termination
    if any(term in text for term in [
        "termination",
        "terminate",
        "earlier determination",
        "absolutely determine",
        "right of re-entry",
        "right of reentry",
        "re-entry",
        "reentry",
        "event of default",
        "in default",
        "default period",
        "breach of the covenants",
        "failed to comply",
        "expiration of the term",
    ]):
        return "Termination / Default"

    # 12. Liability and indemnity
    if any(term in text for term in [
        "indemnify",
        "indemnified",
        "indemnification",
        "indemnity",
        "hold harmless",
        "unlimited liability",
        "limitation of liability",
        "liable for",
    ]):
        return "Liability / Indemnity"

    # 13. Dispute resolution and governing law
    if any(term in text for term in [
        "arbitration",
        "dispute resolution",
        "waiver of jury",
        "exclusive jurisdiction",
        "submission to jurisdiction",
        "venue",
    ]):
        return "Dispute Resolution"

    if any(term in text for term in [
        "governing law",
        "governed by the laws",
        "applicable law",
        "choice of law",
    ]):
        return "Governing Law"

    # 14. Confidentiality and insurance
    if any(term in text for term in [
        "confidential information",
        "confidentiality",
        "non-disclosure",
        "nondisclosure",
        "keep confidential",
    ]):
        return "Confidentiality"

    if any(term in text for term in [
        "insurance policy",
        "maintain insurance",
        "insurance coverage",
        "insured against",
    ]):
        return "Insurance"

    # 15. Notice
    if any(term in text for term in [
        "notice in writing",
        "written notice",
        "notice shall be",
        "notice must be",
        "service of notice",
    ]):
        return "Notice"

    # 16. Payment and rent
    payment_score = sum(
        term in text
        for term in [
            "monthly rent",
            "ground rent",
            "rent amount",
            "shall pay",
            "to pay",
            "payment due",
            "due date",
            "interest thereon",
            "security deposit",
            "fees payable",
            "charges payable",
        ]
    )
    if payment_score >= 1:
        return "Payment / Rent"

    # 17. Parties and identity
    if (
        "between" in text
        and ("hereinafter called" in text or "party of the" in text)
    ):
        return "Parties / Identity"

    # 18. Property description is intentionally near the end because
    # premises/building/land occur in almost every lease clause.
    property_score = sum(
        term in text
        for term in [
            "described in schedule",
            "identified in the attached plan",
            "boundaries of the premises",
            "leased premises include",
            "demised premises include",
            "property address",
            "parking spaces",
            "common areas",
            "floor area",
            "storage area",
            "land and premises",
        ]
    )
    if property_score >= 1:
        return "Property / Premises"

    # 19. General obligations and unknown clauses
    if any(term in text for term in [
        "compliance with laws",
        "further assurances",
        "cooperate in good faith",
        "general obligations",
        "covenants with the lessor as follows",
        "covenants with the landlord as follows",
    ]):
        return "General Obligation"

    if any(term in text for term in [
        "entire agreement",
        "entire understanding",
        "headings are included",
        "severability",
        "remaining provisions shall remain effective",
    ]):
        return "Other / Unknown"

    return "General Contract Clause"


def detect_secondary_clause_types(clause_text: str, primary_clause_type: str) -> list[str]:
    """Return additional legal concepts found in a mixed-purpose clause.

    The existing single-label classifier remains the source of the primary label.
    These rules add distinct secondary labels without duplicating the primary one.
    """

    if not isinstance(clause_text, str) or not clause_text.strip():
        return []

    text = clause_text.lower()

    category_terms = {
        "Payment / Rent": [
            "monthly rent", "ground rent", "rent shall be paid", "shall pay",
            "payment due", "due date", "interest thereon", "security deposit",
        ],
        "Termination / Default": [
            "termination", "terminate", "earlier determination",
            "absolutely determine", "re-entry", "reentry", "event of default",
            "in default", "arrears", "failed to comply", "breach of the covenants",
        ],
        "Assignment / Subletting": [
            "shall not assign", "not to assign", "assignment", "sublet",
            "subletting", "part with possession", "assign mortgage",
        ],
        "Liability / Indemnity": [
            "indemnify", "indemnified", "indemnification", "indemnity",
            "hold harmless", "liable for", "limitation of liability",
        ],
        "Repairs / Maintenance": [
            "good repair", "tenantable repair", "tenantable repairs",
            "structural repairs", "maintenance", "maintain the premises",
            "keep the premises clean",
        ],
        "Alterations / Improvements": [
            "alteration", "alterations", "improvement", "improvements",
            "construct any new building", "construct any new buildings",
            "demolish any existing building", "install fixtures", "approved plans",
        ],
        "Use of Premises": [
            "used only for", "use the premises", "use or permit to be used",
            "lawful purposes", "permitted use", "residential purposes",
            "business purposes",
        ],
        "Quiet Enjoyment": [
            "quiet enjoyment", "peaceably and quietly", "quietly hold",
            "quietly enjoy", "without interruption", "without disturbance",
            "without eviction",
        ],
        "Possession / Surrender": [
            "surrender the premises", "return possession", "vacant possession",
            "deliver possession", "yield up the premises", "holding over",
            "shall automatically vest in the lessor", "hand over all keys",
        ],
        "Taxes / Utilities": [
            "property taxes", "municipal taxes", "real property taxes",
            "rates and taxes", "assessment duties", "business rates",
            "electricity charges", "water charges", "utility charges", "utilities",
        ],
        "Term and Renewal": [
            "initial term", "renewal term", "option to renew", "automatically renew",
            "renew the lease", "fixed term", "extension of the term",
            "expiration of the term",
        ],
        "Lease Grant": [
            "hereby leases", "hereby lets", "hereby demises", "grants the lessee",
            "grants the tenant", "leasehold interest", "to hold the demised premises",
        ],
        "Notice": [
            "notice in writing", "written notice", "notice shall be",
            "notice must be", "service of notice",
        ],
        "Insurance": [
            "insurance policy", "maintain insurance", "insurance coverage",
            "insured against",
        ],
        "Dispute Resolution": [
            "arbitration", "dispute resolution", "waiver of jury",
            "exclusive jurisdiction", "submission to jurisdiction", "venue",
        ],
        "Governing Law": [
            "governing law", "governed by the laws", "choice of law",
        ],
        "Confidentiality": [
            "confidential information", "confidentiality", "non-disclosure",
            "nondisclosure", "keep confidential",
        ],
        "Signature / Execution": [
            "in witness whereof", "signed and delivered", "executed by",
            "counterpart", "witnessed by",
        ],
    }

    secondary_types = []

    for category, terms in category_terms.items():
        if category == primary_clause_type:
            continue

        if any(term in text for term in terms):
            secondary_types.append(category)

    # Keep the output compact and deterministic.
    return secondary_types[:4]

def detect_party_affected(clause_text: str) -> str:
    """
    Detects which party is most affected by the clause.
    This is rule-based for now and can later become an ML label.
    """

    text = clause_text.lower()

    lessee_terms = ["lessee", "tenant", "renter"]
    lessor_terms = ["lessor", "landlord", "owner"]

    lessee_count = sum(1 for term in lessee_terms if term in text)
    lessor_count = sum(1 for term in lessor_terms if term in text)

    if lessee_count > lessor_count:
        return "Lessee / Tenant"

    if lessor_count > lessee_count:
        return "Lessor / Landlord"

    if lessee_count and lessor_count:
        return "Both parties"

    return "Not clearly specified"


def generate_risk_reason(clause_text: str, risk_level: str, clause_type: str, risk_signals: list) -> str:
    """
    Creates a plain-language risk reason for the frontend clause card.
    """

    text = clause_text.lower()

    if risk_signals:
        reasons = [signal.get("reason", "") for signal in risk_signals[:2] if signal.get("reason")]
        if reasons:
            return " ".join(reasons)

    if clause_type == "Payment / Rent":
        if has_blank_or_placeholder(clause_text):
            return "This payment/rent clause appears to contain blank or incomplete commercial terms."
        return "This clause creates a payment obligation and should be checked for amount, due date, late fees, and consequences of non-payment."

    if clause_type == "Termination / Default":
        return "This clause may affect how and when the contract can end, especially after default or breach."

    if clause_type == "Liability / Indemnity":
        return "This clause may shift financial responsibility, liability, damages, or indemnity obligations to one party."

    if clause_type == "Assignment / Subletting":
        return "This clause may restrict transfer, assignment, mortgage, or subletting rights."

    if clause_type == "Dispute Resolution":
        return "This clause may affect how disputes are handled and where claims must be brought."

    if clause_type == "Notice":
        return "This clause may affect whether formal written notice is required before a party can act."

    if clause_type == "Repairs / Maintenance":
        return "This clause allocates responsibility for repairs, maintenance, damage, or the condition of the premises."

    if clause_type == "Alterations / Improvements":
        return "This clause controls alterations, construction work, improvements, consent, or reinstatement obligations."

    if clause_type == "Use of Premises":
        return "This clause limits how the premises may be occupied or used and may require legal or regulatory compliance."

    if clause_type == "Quiet Enjoyment":
        return "This clause concerns the tenant's right to occupy and use the premises without unlawful interference."

    if clause_type == "Taxes / Utilities":
        return "This clause allocates responsibility for taxes, rates, utilities, or other property-related charges."

    if clause_type == "Possession / Surrender":
        return "This clause governs delivery, return, or recovery of possession and the required condition of the premises."

    if clause_type == "Term and Renewal":
        return "This clause defines the lease duration, expiry, extension, holding over, or renewal rights."

    if clause_type == "Lease Grant":
        return "This clause creates or describes the tenant's leasehold right to occupy the premises."

    if risk_level == "High":
        return "This clause contains wording that may create significant legal or financial exposure."

    if risk_level == "Medium":
        return "This clause contains wording that should be reviewed before signing."

    return "No major risk wording was detected by the current prototype model."


def generate_recommended_action(clause_type: str, risk_level: str, clause_text: str) -> str:
    """
    Suggests a practical review action based on clause type and risk level.
    """

    if clause_type == "Payment / Rent":
        return "Confirm the amount, due date, grace period, late fee or interest rate, and payment method are clearly written."

    if clause_type == "Termination / Default":
        return "Check whether the clause includes written notice, a cure period, default period, and a fair termination process."

    if clause_type == "Liability / Indemnity":
        return "Review whether liability is capped, mutual, limited to direct losses, and covered by insurance where needed."

    if clause_type == "Assignment / Subletting":
        return "Check whether consent is required and whether consent can be unreasonably withheld."

    if clause_type == "Dispute Resolution":
        return "Confirm the governing law, forum, arbitration process, and cost responsibility are acceptable."

    if clause_type == "Confidentiality":
        return "Check the scope, duration, exceptions, and permitted disclosures."

    if clause_type == "Notice":
        return "Confirm the notice method, address, delivery timing, and required written form."

    if clause_type == "Signature / Execution":
        return "Confirm all required parties, witnesses, names, dates, and signatures are completed."

    if clause_type == "Repairs / Maintenance":
        return "Confirm which party handles structural, interior, routine, and emergency repairs and who pays the related costs."

    if clause_type == "Alterations / Improvements":
        return "Check consent, permit, contractor, restoration, ownership, and removal requirements for alterations or improvements."

    if clause_type == "Use of Premises":
        return "Confirm the permitted use is broad enough for the tenant's needs and complies with licensing and zoning requirements."

    if clause_type == "Quiet Enjoyment":
        return "Check that access, landlord entry, service interruptions, and interference rights are clearly and fairly limited."

    if clause_type == "Taxes / Utilities":
        return "Confirm exactly which taxes, rates, utilities, shared charges, and reconciliation methods each party must pay."

    if clause_type == "Possession / Surrender":
        return "Check delivery and surrender dates, required condition, fixture removal, abandoned property, and holding-over consequences."

    if clause_type == "Term and Renewal":
        return "Confirm commencement, expiry, notice deadlines, renewal options, renewal rent, and holding-over rules."

    if clause_type == "Lease Grant":
        return "Confirm the premises, included rights, commencement conditions, and scope of occupation are accurately described."

    if risk_level == "High":
        return "Review this clause carefully with a qualified legal professional before signing."

    if risk_level == "Medium":
        return "Review and clarify this clause before relying on the contract."

    return "Keep this clause as part of the review record."


def analyze_clauses(raw_clauses: list[str]) -> list[dict[str, Any]]:
    analyzed = []

    for index, clause_text in enumerate(raw_clauses, start=1):
        # Risk prediction
        ml_prediction = predict_ml_risk(clause_text)
        risk_signals = detect_risk_signals(clause_text)
        risk_level = final_risk_level(
            ml_prediction,
            risk_signals,
            clause_text,
        )

        # Phase 6 validated hybrid clause-type prediction.
        hybrid_type_result = predict_clause_type_hybrid(clause_text)

        clause_type = hybrid_type_result["clause_type"]
        clause_type_source = hybrid_type_result[
            "clause_type_source"
        ]

        secondary_clause_types = detect_secondary_clause_types(
            clause_text=clause_text,
            primary_clause_type=clause_type,
        )

        party_affected = detect_party_affected(clause_text)

        risk_reason = generate_risk_reason(
            clause_text=clause_text,
            risk_level=risk_level,
            clause_type=clause_type,
            risk_signals=risk_signals,
        )

        recommended_action = generate_recommended_action(
            clause_type=clause_type,
            risk_level=risk_level,
            clause_text=clause_text,
        )

        analyzed.append({
            "clause_number": index,
            "text": clean_display_text(clause_text),
            "preview": smart_preview(clause_text),
            "risk_level": risk_level,
            "ml_prediction": ml_prediction,
            "risk_signals": risk_signals,
            "clause_type": clause_type,
            "clause_type_confidence": hybrid_type_result[
                "clause_type_confidence"
            ],
            "clause_type_source": clause_type_source,
            "clause_type_ml_prediction": hybrid_type_result[
                "ml_clause_type"
            ],
            "clause_type_rule_prediction": hybrid_type_result[
                "rule_clause_type"
            ],
            "clause_type_rule_matches": hybrid_type_result[
                "rule_matches"
            ],
            "secondary_clause_types": secondary_clause_types,
            "risk_reason": risk_reason,
            "party_affected": party_affected,
            "recommended_action": recommended_action,
        })

    return analyzed


# ============================================================
# FIELD EXTRACTION
# ============================================================

def extract_contract_type(text: str) -> str:
    lower_text = text.lower()

    if "lease" in lower_text or "rent" in lower_text:
        return "Lease Deed / Rent Agreement"

    if "employment" in lower_text:
        return "Employment Agreement"

    if "non-disclosure" in lower_text or "confidentiality" in lower_text:
        return "Non-Disclosure Agreement"

    if "service agreement" in lower_text:
        return "Service Agreement"

    return "Legal Contract"


def extract_parties(text: str) -> tuple[str, str]:
    lessor = "Not specified in the document."
    lessee = "Not specified in the document."

    lessor_match = re.search(
        r"between\s+(.+?)\s+hereinafter\s+called\s+['\"]?The Lessor",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    lessee_match = re.search(
        r"and\s+(.+?)\s+hereinafter\s+called\s+['\"]?The Lessee",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )

    if lessor_match:
        possible_lessor = clean_display_text(lessor_match.group(1))
        if not has_blank_or_placeholder(possible_lessor) and len(possible_lessor) < 120:
            lessor = possible_lessor

    if lessee_match:
        possible_lessee = clean_display_text(lessee_match.group(1))
        if not has_blank_or_placeholder(possible_lessee) and len(possible_lessee) < 120:
            lessee = possible_lessee

    return lessor, lessee


def extract_rent_amount(text: str) -> str:
    patterns = [
        r"monthly\s+ground\s+rent\s+of\s+Rs\.?\s*([^\s,.]+)",
        r"monthly\s+rent\s+of\s+Rs\.?\s*([^\s,.]+)",
        r"rent\s+of\s+Rs\.?\s*([^\s,.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1)
            return value_or_not_specified(value)

    return "Not specified in the document."


def extract_lease_term(text: str) -> str:
    match = re.search(
        r"for\s+a\s+term\s+of\s+(.+?)\s+years",
        text,
        flags=re.IGNORECASE
    )

    if match:
        value = match.group(1) + " years"
        return value_or_not_specified(value)

    return "Not specified in the document."


def extract_start_date(text: str) -> str:
    match = re.search(
        r"commencing\s+from\s+(.+?)(?:,|\s+but|\s+and)",
        text,
        flags=re.IGNORECASE
    )

    if match:
        value = match.group(1)
        return value_or_not_specified(value)

    return "Not specified in the document."


def extract_missing_fields(text: str) -> list[str]:
    missing = []

    if "Not specified" in extract_parties(text)[0]:
        missing.append("Lessor name")

    if "Not specified" in extract_parties(text)[1]:
        missing.append("Lessee name")

    if "Not specified" in extract_rent_amount(text):
        missing.append("Monthly rent amount")

    if "Not specified" in extract_lease_term(text):
        missing.append("Lease duration")

    if "Not specified" in extract_start_date(text):
        missing.append("Lease start date")

    if has_blank_or_placeholder(text):
        missing.append("Some contract blanks/placeholders are still unfilled")

    return missing


# ============================================================
# ANSWER HELPERS
# ============================================================

def get_overall_risk(analyzed_clauses: list[dict[str, Any]]) -> str:
    levels = [clause["risk_level"] for clause in analyzed_clauses]

    if "High" in levels:
        return "High"

    if "Medium" in levels:
        return "Medium"

    return "Low"


def find_clauses_by_keywords(
    analyzed_clauses: list[dict[str, Any]],
    keywords: list[str],
) -> list[dict[str, Any]]:
    results = []

    for clause in analyzed_clauses:
        lower_text = clause["text"].lower()

        if any(keyword in lower_text for keyword in keywords):
            results.append(clause)

    return results


def get_detected_risk_keywords(
    analyzed_clauses: list[dict[str, Any]],
) -> list[str]:
    keywords = []

    for clause in analyzed_clauses:
        for signal in clause["risk_signals"]:
            keywords.append(signal["keyword"])

    return sorted(list(set(keywords)))


def format_related_clauses(
    clauses: list[dict[str, Any]],
    limit: int = 5,
) -> str:
    if not clauses:
        return "No directly related clauses were found."

    lines = []

    for clause in clauses[:limit]:
        lines.append(
            f"- Clause {clause['clause_number']} — {clause['risk_level']} Risk: {clause['preview']}"
        )

    return "\n".join(lines)


# ============================================================
# POLISHED ANSWERS
# ============================================================

def build_summary_answer(contract_data: dict[str, Any]) -> str:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    contract_type = extract_contract_type(text)
    lessor, lessee = extract_parties(text)
    rent_amount = extract_rent_amount(text)
    lease_term = extract_lease_term(text)
    start_date = extract_start_date(text)
    overall_risk = get_overall_risk(clauses)
    risk_keywords = get_detected_risk_keywords(clauses)
    missing_fields = extract_missing_fields(text)

    risk_lines = []

    if "default" in risk_keywords:
        risk_lines.append("Rent/default wording appears in the contract.")

    if "termination" in risk_keywords or "earlier determination" in risk_keywords:
        risk_lines.append("The contract includes termination or early determination wording.")

    if "breach" in risk_keywords:
        risk_lines.append("The contract contains breach-related wording.")

    if not risk_lines:
        risk_lines.append("No major risk signals were detected by the prototype model.")

    missing_text = "\n".join([f"- {field}" for field in missing_fields]) if missing_fields else "- No major missing fields detected."

    risk_text = "\n".join([f"- {item}" for item in risk_lines])

    answer = f"""
Contract Summary

Contract type:
{contract_type}

Parties:
- Lessor: {lessor}
- Lessee: {lessee}

Main purpose:
This document appears to create a lease arrangement where the Lessor grants lease rights over the described premises to the Lessee.

Key terms:
- Lease term: {lease_term}
- Lease start date: {start_date}
- Monthly rent: {rent_amount}

Overall risk level:
{overall_risk}

Main risks detected:
{risk_text}

Missing or incomplete information:
{missing_text}

Important note:
This is a prototype ML-based contract review summary and not legal advice.
"""

    return answer.strip()


def build_payment_answer(contract_data: dict[str, Any]) -> str:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    payment_clauses = find_clauses_by_keywords(clauses, PAYMENT_KEYWORDS)

    rent_amount = extract_rent_amount(text)

    payment_date = "Not specified in the document."
    interest_rate = "Not specified in the document."

    if re.search(r"interest", text, flags=re.IGNORECASE):
        interest_rate = "Mentioned, but the exact interest rate appears to be blank or unclear."

    risk_level = "Medium" if payment_clauses else "Low"

    answer = f"""
Payment / Rent Review

Main rent obligation:
The Lessee appears to be required to pay monthly ground rent to the Lessor.

Rent amount:
{rent_amount}

Payment date:
{payment_date}

Late payment consequence:
{interest_rate}

Risk level:
{risk_level}

Why this matters:
The payment section appears incomplete because important commercial terms such as the rent amount, due date, and interest rate are not clearly filled in. This can create confusion or future disputes between the Lessor and the Lessee.

Related clauses:
{format_related_clauses(payment_clauses, limit=4)}

Important note:
This is a prototype ML-based contract review response and not legal advice.
"""

    return answer.strip()


def build_termination_answer(contract_data: dict[str, Any]) -> str:
    clauses = contract_data["clauses"]

    termination_clauses = find_clauses_by_keywords(clauses, TERMINATION_KEYWORDS)

    missing_items = [
        "Number of months before rent default applies",
        "Notice requirement before termination",
        "Cure period for the Lessee",
        "Clear procedure for early termination",
    ]

    missing_text = "\n".join([f"- {item}" for item in missing_items])

    answer = f"""
Termination / Default Review

Main issue:
The lease appears to allow early determination or termination if the Lessee fails to pay rent, falls into default, or breaches lease obligations.

Default triggers detected:
- Non-payment of monthly ground rent
- Breach of lease covenants by the Lessee
- Possible early determination of the lease

Risk level:
Medium

Why this may be risky:
The contract contains termination/default wording, but some important details appear blank or unclear. In particular, the default period and process for notice or cure are not clearly stated.

Missing or unclear information:
{missing_text}

Related clauses:
{format_related_clauses(termination_clauses, limit=4)}

Important note:
This is a prototype ML-based contract review response and not legal advice.
"""

    return answer.strip()


def build_risky_clauses_answer(contract_data: dict[str, Any]) -> str:
    clauses = contract_data["clauses"]

    risky_clauses = [
        clause for clause in clauses
        if clause["risk_level"] in ["Medium", "High"]
    ]

    if not risky_clauses:
        return """
Risky Clauses Review

No medium-risk or high-risk clauses were detected by the current prototype model.

Important note:
This is a prototype ML-based contract review response and not legal advice.
""".strip()

    lines = []

    for clause in risky_clauses[:8]:
        signals = clause.get("risk_signals", [])

        if signals:
            reasons = ", ".join([signal["keyword"] for signal in signals[:3]])
        else:
            reasons = "General risk wording detected."

        explanation = explain_clause_risk(clause)

        lines.append(
            f"""
Clause {clause['clause_number']} — {clause['risk_level']} Risk

Reason:
{reasons}

Explanation:
{explanation}

ML prediction:
{clause['ml_prediction']}

Preview:
{clause['preview']}
""".strip()
        )

    answer = f"""
Risky Clauses Review

I found {len(risky_clauses)} medium-risk or high-risk clause(s) in this contract.

{chr(10).join(lines)}

Important note:
This is a prototype ML-based contract review response and not legal advice.
"""

    return answer.strip()


def explain_clause_risk(clause: dict[str, Any]) -> str:
    text = clause["text"].lower()

    if "rent" in text and ("default" in text or "arrears" in text or "not paid" in text):
        return "This clause may create risk because it connects unpaid rent with default consequences."

    if "termination" in text or "earlier determination" in text:
        return "This clause may create risk because it allows the lease to end earlier than the full term."

    if "breach" in text or "covenants" in text:
        return "This clause may create risk because breach of obligations can trigger legal or contractual consequences."

    if "interest" in text:
        return "This clause may create risk because it mentions interest or late payment consequences, but the exact amount may be unclear."

    if "liability" in text or "indemnify" in text or "indemnification" in text:
        return "This clause may create risk because it may shift liability or financial responsibility to one party."

    return "This clause contains wording that the prototype model identified as requiring review."


def build_general_answer(
    contract_data: dict[str, Any],
    question: str,
) -> str:
    summary = build_summary_answer(contract_data)

    answer = f"""
I can help with contract-specific questions such as:

- Summarize this contract
- What are the risky clauses?
- What are the payment terms?
- Explain the termination clause
- What information is missing?

For now, here is the main contract summary:

{summary}
"""

    return answer.strip()


def build_missing_fields_answer(contract_data: dict[str, Any]) -> str:
    text = contract_data["text"]
    missing_fields = extract_missing_fields(text)

    if not missing_fields:
        return """
Missing Information Review

No major missing fields were detected by the current prototype model.

Important note:
This is a prototype ML-based contract review response and not legal advice.
""".strip()

    missing_text = "\n".join([f"- {field}" for field in missing_fields])

    return f"""
Missing Information Review

The following information appears missing, blank, or incomplete:

{missing_text}

Why this matters:
Blank fields in legal contracts can create confusion, disputes, or enforcement issues because the parties may not have clearly agreed on those terms.

Important note:
This is a prototype ML-based contract review response and not legal advice.
""".strip()


def answer_contract_question(
    contract_data: dict[str, Any],
    question: str,
) -> str:
    question_lower = question.lower()

    if any(word in question_lower for word in ["summary", "summarize", "overview"]):
        return build_summary_answer(contract_data)

    if any(word in question_lower for word in ["payment", "rent", "fee", "deposit", "amount"]):
        return build_payment_answer(contract_data)

    if any(word in question_lower for word in ["termination", "terminate", "default", "breach", "expiration"]):
        return build_termination_answer(contract_data)

    if any(word in question_lower for word in ["risk", "risky", "dangerous", "problem", "clause"]):
        return build_risky_clauses_answer(contract_data)

    if any(word in question_lower for word in ["missing", "blank", "not specified", "incomplete"]):
        return build_missing_fields_answer(contract_data)

    return build_general_answer(contract_data, question)


# ============================================================
# API ROUTES
# ============================================================
# ============================================================
# STRUCTURED ANSWER BUILDER
# ============================================================

def make_related_clause_items(
    clauses: list[dict[str, Any]],
    limit: int = 5,
) -> list[dict[str, Any]]:
    related = []

    for clause in clauses[:limit]:
        related.append({
            "clause_number": clause.get("clause_number"),
            "risk_level": clause.get("risk_level"),
            "ml_prediction": clause.get("ml_prediction"),
            "preview": clause.get("preview"),
            "clause_type": clause.get("clause_type"),
            "clause_type_confidence": clause.get(
                "clause_type_confidence",
                0.0,
            ),
            "clause_type_source": clause.get(
                "clause_type_source",
                "legacy",
            ),
            "secondary_clause_types": clause.get(
                "secondary_clause_types",
                [],
            ),
            "risk_reason": clause.get("risk_reason"),
            "party_affected": clause.get("party_affected"),
            "recommended_action": clause.get("recommended_action"),
            "risk_signals": [
                signal.get("keyword")
                for signal in clause.get("risk_signals", [])
            ],
        })

    return related


def build_structured_summary(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    contract_type = extract_contract_type(text)
    lessor, lessee = extract_parties(text)
    rent_amount = extract_rent_amount(text)
    lease_term = extract_lease_term(text)
    start_date = extract_start_date(text)
    overall_risk = get_overall_risk(clauses)
    missing_fields = extract_missing_fields(text)
    risk_keywords = get_detected_risk_keywords(clauses)

    key_points = [
        f"Contract type detected as {contract_type}.",
        "The document appears to create a lease relationship between a Lessor and a Lessee.",
        f"Overall risk level is {overall_risk}.",
    ]

    if lessor == "Not specified in the document.":
        key_points.append("The Lessor name is not clearly filled in.")

    if lessee == "Not specified in the document.":
        key_points.append("The Lessee name is not clearly filled in.")

    if rent_amount == "Not specified in the document.":
        key_points.append("The monthly rent amount is missing or not clearly stated.")

    if lease_term == "Not specified in the document.":
        key_points.append("The lease duration is missing or not clearly stated.")

    if start_date == "Not specified in the document.":
        key_points.append("The lease start date is missing or not clearly stated.")

    if risk_keywords:
        key_points.append(
            "Detected risk signals include: " + ", ".join(risk_keywords[:6]) + "."
        )

    related_clauses = [
        clause for clause in clauses
        if clause.get("risk_level") in ["Medium", "High"]
    ]

    fallback_answer = build_summary_answer(contract_data)

    return {
        "answer_type": "summary",
        "title": "Contract Summary",
        "risk_level": overall_risk,
        "summary": "This contract appears to be a lease/rent agreement where the Lessor grants lease rights to the Lessee. Several important fields may still be blank or incomplete.",
        "key_points": key_points,
        "missing_fields": missing_fields,
        "related_clauses": make_related_clause_items(related_clauses, limit=5),
        "contract_details": {
            "contract_type": contract_type,
            "lessor": lessor,
            "lessee": lessee,
            "lease_term": lease_term,
            "lease_start_date": start_date,
            "monthly_rent": rent_amount,
        },
        "answer": fallback_answer,
    }


def build_structured_payment(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    text = contract_data["text"]
    clauses = contract_data["clauses"]

    payment_clauses = find_clauses_by_keywords(clauses, PAYMENT_KEYWORDS)

    rent_amount = extract_rent_amount(text)

    missing_fields = []

    if rent_amount == "Not specified in the document.":
        missing_fields.append("Monthly rent amount")

    missing_fields.extend([
        "Payment due date",
        "Interest rate for late payment",
    ])

    key_points = [
        "The Lessee appears to be required to pay monthly ground rent to the Lessor.",
    ]

    if rent_amount == "Not specified in the document.":
        key_points.append("The rent amount appears blank or not clearly stated.")
    else:
        key_points.append(f"The detected rent amount is {rent_amount}.")

    key_points.append("The exact payment date is not clearly detected.")
    key_points.append("Late-payment interest is mentioned, but the exact rate appears unclear or blank.")

    fallback_answer = build_payment_answer(contract_data)

    return {
        "answer_type": "payment",
        "title": "Payment / Rent Review",
        "risk_level": "Medium",
        "summary": "The contract includes rent/payment obligations, but some commercial payment details appear missing or incomplete.",
        "key_points": key_points,
        "missing_fields": list(dict.fromkeys(missing_fields)),
        "related_clauses": make_related_clause_items(payment_clauses, limit=5),
        "contract_details": {
            "monthly_rent": rent_amount,
            "payment_due_date": "Not specified in the document.",
            "late_payment_interest": "Mentioned, but exact rate appears unclear or blank.",
        },
        "answer": fallback_answer,
    }


def build_structured_termination(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    clauses = contract_data["clauses"]

    termination_clauses = find_clauses_by_keywords(clauses, TERMINATION_KEYWORDS)

    missing_fields = [
        "Default period",
        "Notice requirement before termination",
        "Cure period for the Lessee",
        "Clear early-termination procedure",
    ]

    key_points = [
        "The contract appears to allow early determination or termination if the Lessee defaults.",
        "Non-payment of rent may trigger default-related consequences.",
        "Breach of lease covenants may also trigger consequences.",
        "The default period and cure process appear unclear or incomplete.",
    ]

    fallback_answer = build_termination_answer(contract_data)

    return {
        "answer_type": "termination",
        "title": "Termination / Default Review",
        "risk_level": "Medium",
        "summary": "The lease appears to include default and early-termination wording, but important procedural details are missing or unclear.",
        "key_points": key_points,
        "missing_fields": missing_fields,
        "related_clauses": make_related_clause_items(termination_clauses, limit=5),
        "contract_details": {
            "default_triggers": [
                "Non-payment of rent",
                "Breach of lease covenants",
                "Possible early determination of the lease",
            ],
            "notice_requirement": "Not clearly specified in the document.",
            "cure_period": "Not clearly specified in the document.",
        },
        "answer": fallback_answer,
    }


def build_structured_risks(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    clauses = contract_data["clauses"]

    risky_clauses = [
        clause for clause in clauses
        if clause.get("risk_level") in ["Medium", "High"]
    ]

    overall_risk = get_overall_risk(clauses)
    risk_keywords = get_detected_risk_keywords(clauses)

    key_points = []

    if risky_clauses:
        key_points.append(f"{len(risky_clauses)} medium-risk or high-risk clause(s) were detected.")
    else:
        key_points.append("No medium-risk or high-risk clauses were detected by the current prototype model.")

    if risk_keywords:
        key_points.append("Detected risk signals include: " + ", ".join(risk_keywords[:8]) + ".")

    key_points.append("The current risk result is generated using ML prediction plus keyword/rule validation.")

    fallback_answer = build_risky_clauses_answer(contract_data)

    return {
        "answer_type": "risk",
        "title": "Risky Clauses Review",
        "risk_level": overall_risk,
        "summary": "The model reviewed the contract clauses and identified clauses that may require closer review.",
        "key_points": key_points,
        "missing_fields": [],
        "related_clauses": make_related_clause_items(risky_clauses, limit=8),
        "contract_details": {
            "total_risky_clauses": len(risky_clauses),
            "detected_risk_signals": risk_keywords,
        },
        "answer": fallback_answer,
    }


def build_structured_missing_fields(
    contract_data: dict[str, Any],
) -> dict[str, Any]:
    text = contract_data["text"]

    missing_fields = extract_missing_fields(text)

    key_points = []

    if missing_fields:
        key_points.append(f"{len(missing_fields)} missing or incomplete item(s) were detected.")
        key_points.append("Blank fields can create uncertainty in contract interpretation.")
        key_points.append("The parties should fill in missing commercial and identity details before relying on the document.")
    else:
        key_points.append("No major missing fields were detected by the current prototype model.")

    fallback_answer = build_missing_fields_answer(contract_data)

    return {
        "answer_type": "missing_fields",
        "title": "Missing Information Review",
        "risk_level": "Medium" if missing_fields else "Low",
        "summary": "The system checked for blank, incomplete, or unclear contract information.",
        "key_points": key_points,
        "missing_fields": missing_fields,
        "related_clauses": [],
        "contract_details": {},
        "answer": fallback_answer,
    }


def build_structured_general(
    contract_data: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    fallback_answer = build_general_answer(contract_data, question)

    return {
        "answer_type": "general",
        "title": "Contract Agent Response",
        "risk_level": get_overall_risk(contract_data["clauses"]),
        "summary": "The agent can answer contract-specific questions about summary, risks, payment terms, termination, clauses, and missing fields.",
        "key_points": [
            "Try asking: Summarize this contract.",
            "Try asking: What are the risky clauses?",
            "Try asking: What are the payment terms?",
            "Try asking: Explain the termination clause.",
            "Try asking: What information is missing?",
        ],
        "missing_fields": [],
        "related_clauses": [],
        "contract_details": {},
        "answer": fallback_answer,
    }


def build_structured_answer(
    contract_data: dict[str, Any],
    question: str,
) -> dict[str, Any]:
    question_lower = question.lower()

    if any(word in question_lower for word in ["summary", "summarize", "overview"]):
        return build_structured_summary(contract_data)

    if any(word in question_lower for word in ["payment", "rent", "fee", "deposit", "amount"]):
        return build_structured_payment(contract_data)

    if any(word in question_lower for word in ["termination", "terminate", "default", "breach", "expiration"]):
        return build_structured_termination(contract_data)

    if any(word in question_lower for word in ["risk", "risky", "dangerous", "problem", "clause"]):
        return build_structured_risks(contract_data)

    if any(word in question_lower for word in ["missing", "blank", "not specified", "incomplete"]):
        return build_structured_missing_fields(contract_data)

    return build_llm_general_answer(contract_data, question)

@app.get("/")
def home() -> dict[str, str]:
    return {
        "message": "Legal Contract ML Agent SaaS backend is running with Gemini support."
    }


@app.post("/upload-contract")
async def upload_contract(
    file: UploadFile = File(...),
) -> dict[str, Any]:
    file_bytes = await file.read()

    extraction_result = extract_pdf_content(file_bytes)
    raw_text = extraction_result["text"]
    normalized_text = normalize_raw_text(raw_text)
    cleaned_contract_text = remove_irrelevant_sections(normalized_text)

    raw_clauses = split_into_clauses(cleaned_contract_text)
    analyzed_clauses = analyze_clauses(raw_clauses)

    contract_id = str(uuid.uuid4())
    overall_risk = get_overall_risk(analyzed_clauses)

    analysis = {
        "contract_type": extract_contract_type(cleaned_contract_text),
        "overall_risk": overall_risk,
        "total_clauses": len(analyzed_clauses),
        "risky_clauses": len([
            clause for clause in analyzed_clauses
            if clause["risk_level"] in ["Medium", "High"]
        ]),
        "missing_fields": extract_missing_fields(cleaned_contract_text),
        "summary": build_summary_answer({
            "text": cleaned_contract_text,
            "clauses": analyzed_clauses,
        }),
    }

    # Store in temporary memory for fast access during the current backend session.
    CONTRACT_STORE[contract_id] = {
        "filename": file.filename,
        "content_type": file.content_type,
        "text": cleaned_contract_text,
        "clauses": analyzed_clauses,
        "analysis": analysis,
    }

    # Store in SQLite so the contract can still be used after backend restart.
    save_contract_to_db(
        contract_id=contract_id,
        filename=file.filename,
        content_type=file.content_type,
        contract_text=cleaned_contract_text,
        clauses=analyzed_clauses,
        analysis=analysis,
    )

    response = {
        "contract_id": contract_id,
        "filename": file.filename,
        "content_type": file.content_type,
        "message": "Contract uploaded, analyzed, and saved successfully.",
        "raw_total_characters": len(raw_text),
        "total_characters": len(cleaned_contract_text),
        "page_count": extraction_result["page_count"],
        "extraction_method": extraction_result["extraction_method"],
        "text_pages": extraction_result["text_pages"],
        "ocr_pages": extraction_result["ocr_pages"],
        "failed_ocr_pages": extraction_result["failed_ocr_pages"],
        "text_preview": smart_preview(cleaned_contract_text, max_length=700),
        "analysis": analysis,
        "clauses": analyzed_clauses,
    }

    return response


@app.post("/ask-contract")
async def ask_contract(request: AskRequest) -> dict[str, Any]:
    contract_id = request.contract_id

    contract_data = None

    # 1. First try temporary in-memory store.
    if contract_id in CONTRACT_STORE:
        contract_data = CONTRACT_STORE[contract_id]

    # 2. If memory was reset, load the contract from SQLite.
    if contract_data is None:
        contract_data = load_contract_from_db(contract_id)

        if contract_data:
            CONTRACT_STORE[contract_id] = contract_data

    # 3. If still not found, return a structured error.
    if contract_data is None:
        return {
            "answer_type": "error",
            "title": "Contract Not Found",
            "risk_level": "Unknown",
            "summary": "No contract was found in memory or database.",
            "key_points": [
                "The contract ID was not found.",
                "Please upload the contract again.",
            ],
            "missing_fields": [],
            "related_clauses": [],
            "contract_details": {},
            "answer": "No contract found. Please upload the contract again."
        }

    structured_answer = build_structured_answer(contract_data, request.question)

    return structured_answer


@app.get("/contracts")
def get_contracts() -> dict[str, list[dict[str, Any]]]:
    """
    Returns saved contract history from SQLite.
    This is useful for testing persistence and later building a frontend history page.
    """

    return {
        "contracts": list_contracts_from_db()
    }

@app.get("/contracts/{contract_id}")
def get_contract_by_id(contract_id: str) -> dict[str, Any]:
    contract_data = load_contract_from_db(contract_id)

    if contract_data is None:
        return {
            "error": True,
            "message": "Contract not found.",
            "contract_id": contract_id,
        }

    return {
        "error": False,
        "contract_id": contract_id,
        "filename": contract_data.get("filename"),
        "message": "Contract loaded successfully.",
        "analysis": contract_data.get("analysis"),
        "clauses": contract_data.get("clauses"),
    }
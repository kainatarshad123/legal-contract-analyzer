# Legal Contract Analyzer

## Project Overview

The **Legal Contract Analyzer** is an ML-based legal document review SaaS prototype for lease, rent, tenancy, and related commercial agreements. It accepts PDF contracts, extracts searchable text or applies selective OCR, segments the document into clauses, predicts risk and clause type, identifies missing information, and presents structured review results through a React interface.

The system combines a character TF-IDF classifier, balanced Logistic Regression, deterministic legal rules, confidence-based human-review flags, SQLite persistence, Gemini-assisted explanations, retrieval-augmented generation (RAG), contract comparison, and PDF/DOCX export.

> **Project status: Completed and evaluation-strengthened.**
> All core phases and post-completion enhancements have been implemented and tested. A transformer feasibility experiment was completed, evaluated against the frozen external set, and intentionally not adopted because it underperformed the production hybrid classifier.

## Main Features

- Validated PDF upload with MIME, signature, corruption, file-size, and page-count checks
- Searchable-PDF extraction using PyMuPDF
- Page-level Tesseract OCR fallback for scanned or mixed PDFs
- Text cleaning and legal clause segmentation
- Low, Medium, and High risk classification with reasons and actions
- 22-category clause-type classification
- Character TF-IDF + balanced Logistic Regression production model
- High-precision legal-rule validation and hybrid prediction
- Confidence labels and `needs_manual_review` flagging
- Missing-field and placeholder detection
- Affected-party detection
- Clause-level plain-English explanation through Gemini
- RAG-based contract Q&A using stored chunks instead of full-contract prompt stuffing
- Visible Q&A sources with clause number, clause type, match score, and snippet
- Contract history, reopening, and deletion through SQLite
- Clause-level contract comparison with text changes and risk deltas
- PDF and DOCX analysis export
- Frozen 330-clause external evaluation
- Per-class precision, recall, F1-score, support, and confusion matrices
- Phase-5 label ablation study
- DistilBERT transformer feasibility experiment
- Static type checking with `mypy`

## Architecture

```text
React + Vite Frontend
        |
        v
FastAPI Routers and Pydantic Schemas
        |
        +---------------- PDF validation
        +---------------- PyMuPDF extraction / Tesseract OCR
        +---------------- Text cleaning and clause segmentation
        +---------------- Risk model and legal risk rules
        +---------------- Character TF-IDF clause classifier
        +---------------- High-precision hybrid clause rules
        +---------------- Confidence/manual-review metadata
        +---------------- SQLite contracts + contract_chunks
        +---------------- Contract comparison and report export
        +---------------- RAG retrieval -> Gemini
        |
        v
Structured results and evidence sources returned to frontend
```

## Technology Stack

### Frontend

- React
- Vite
- JavaScript
- CSS
- Fetch API

### Backend

- Python 3.13
- FastAPI
- Uvicorn
- Pydantic
- SQLite
- Python type hints and `mypy`

### Document Processing

- PyMuPDF (`fitz`)
- Tesseract OCR
- `pytesseract`
- Pillow
- Regular expressions and custom clause segmentation

### Machine Learning

- Scikit-learn
- Character and word TF-IDF
- Balanced Logistic Regression
- Hybrid ML + deterministic legal rules
- Joblib model persistence
- Hugging Face Transformers and PyTorch for the DistilBERT experiment

### AI and Retrieval

- Gemini API (`gemini-2.5-flash-lite`)
- Clause-aware document chunks
- TF-IDF retrieval
- Cosine-similarity ranking
- Top-k evidence selection

## Application Workflow

1. The user uploads a PDF contract.
2. The backend validates size, type, signature, structure, and page count.
3. PyMuPDF extracts selectable text.
4. Pages with insufficient text are processed with Tesseract OCR.
5. Text is cleaned and segmented into legal clauses.
6. The risk model and legal rules assign risk, reason, affected party, and action.
7. The character TF-IDF classifier predicts one of 22 clause categories.
8. High-precision rules validate or correct selected low-confidence predictions.
9. Confidence labels and manual-review flags are generated.
10. Contract text, structured analysis, and RAG chunks are stored in SQLite.
11. The frontend displays summaries, risks, missing fields, and detailed clause cards.
12. Open-ended Q&A retrieves the most relevant chunks and sends only those excerpts to Gemini.
13. Users can inspect evidence sources, explain one clause, compare contracts, or export reports.

## RAG-Based Contract Q&A

The Q&A system no longer sends the full contract text to Gemini for ordinary open-ended questions.

- Clauses are stored as retrieval chunks in SQLite.
- Long clauses are divided using controlled overlap.
- TF-IDF represents the question and stored chunks.
- Cosine similarity ranks the chunks.
- The top five chunks above the similarity threshold are included in the prompt.
- Gemini must answer from those excerpts or state that the evidence is insufficient.
- The frontend displays source number, clause number, clause type, similarity score, chunk index, and snippet.
- Older contracts are backfilled automatically when they are first queried.
- Contract chunks are deleted with the parent contract.

Structured questions such as contract summary, risky clauses, missing information, payment terms, and termination terms continue to use deterministic answer builders.

## Clause Classification and Confidence

The production classifier combines:

- Character n-gram TF-IDF features
- Balanced Logistic Regression
- Prediction confidence
- High-precision legal keyword rules
- Secondary-category detection
- Fallback logic

Predictions include:

- `clause_type`
- `clause_type_confidence`
- `clause_type_source`
- `confidence_label`
- `needs_manual_review`
- `review_threshold` (0.55)
- `high_confidence_threshold` (0.70)

This avoids presenting all predictions as equally certain.

## Datasets

The final clause-type collection contains **6,486 rows**, including **5,041 training**, **749 validation**, and **696 test** examples across **22 labels**. It combines LEDGAR provisions with lease-domain clauses and additional Phase-5 examples.

The frozen external evaluation set contains **330 manually reviewed clauses** and was excluded from training. Dataset validation found zero duplicate clause-label rows and zero normalized text overlap between the combined dataset and frozen external set.

## Model Evaluation

| Approach | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Original external baseline | 41.21% | 36.49% | 44.89% |
| Character TF-IDF model | 50.00% | 44.67% | 49.74% |
| Final hybrid ML + rules | **53.33%** | **46.86%** | **53.13%** |

The final hybrid used **288 ML decisions** and **42 rule decisions**, showing that rules acted as targeted corrections rather than replacing the classifier.

### Phase-5 Ablation

| Training configuration | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Without Phase-5 labels | 46.97% | 42.37% | 47.54% |
| With Phase-5 labels | **50.00%** | **44.67%** | **49.74%** |
| Difference | **+3.03 pp** | **+2.30 pp** | **+2.20 pp** |

The additional lease examples improved the selected character model, although those automatically generated labels remain provisional pending complete legal review.

## Transformer Fine-Tuning Experiment

A DistilBERT sequence-classification pipeline was implemented using the same 22-class taxonomy and the same frozen 330-clause evaluation set.

Because the experiment was run locally on CPU under a same-day submission constraint, the feasibility model used a maximum sequence length of 96 and approximately 0.20 epoch.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Current hybrid | **53.33%** | **46.86%** | **53.13%** |
| DistilBERT feasibility run | 6.06% | 2.61% | 4.48% |

Additional transformer measurements:

- Average confidence: 5.38%
- Model size: approximately 1.02 GB
- CPU inference: approximately 103-120 ms per clause
- Adoption decision: **not adopted**

The reduced run was severely undertrained, so it does not establish the maximum possible transformer performance. It does establish that this trained artifact should not replace or augment the stronger, smaller production hybrid. A full multi-epoch GPU experiment with a legal-domain pretrained model remains future research.

## Contract Comparison

Two stored contracts can be compared at clause level. The interface reports:

- Added clauses
- Removed clauses
- Changed clauses
- Related clause text differences
- Clause-category changes
- Risk-level changes and risk deltas

This supports template-versus-signed-contract and draft-versus-draft review workflows.

## PDF and DOCX Export

The structured analysis can be exported as:

- A shareable PDF report
- An editable DOCX report

Exports include contract metadata, summary, risk overview, missing fields, clause categories, confidence, affected party, risk reason, and recommended action.

## Database

SQLite stores:

- Contract ID
- Filename and content type
- Full extracted contract text
- Structured clause analysis
- Overall analysis
- Creation date
- RAG chunks with chunk index, clause number, clause type, and chunk text

## Backend Structure

```text
backend/
├── main.py
├── routers/
│   ├── contracts.py
│   └── qa.py
├── services/
│   ├── upload_validator.py
│   ├── pdf_extraction.py
│   ├── ocr.py
│   ├── text_processing.py
│   ├── risk_analyzer.py
│   ├── clause_classifier.py
│   ├── clause_rules.py
│   ├── contract_analysis.py
│   ├── gemini_client.py
│   └── rag_service.py
├── db/
│   └── database.py
├── schemas/
│   └── qa.py
├── ml_model/
│   ├── clause_type_model.pkl
│   ├── clause_type_vectorizer.pkl
│   ├── combined_clause_type_dataset.csv
│   ├── external_test_clauses_final.csv
│   └── transformer_clause_model/
├── transformer_training/
│   ├── inspect_transformer_data.py
│   ├── prepare_transformer_dataset.py
│   ├── train_transformer_classifier.py
│   ├── evaluate_transformer_classifier.py
│   └── compare_clause_models.py
├── evaluation/
└── contracts.db
```

## Installation

### Backend

```cmd
cd "C:\Users\YourName\Documents\legal_contract_analyzer\backend"
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Create `backend/.env`:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Start the backend:

```cmd
uvicorn main:app --reload
```

Backend: `http://127.0.0.1:8000`  
Swagger UI: `http://127.0.0.1:8000/docs`

### Frontend

```cmd
cd "C:\Users\YourName\Documents\legal_contract_analyzer\frontend"
npm install
npm run dev
```

Frontend: normally `http://localhost:5173`

## Main API Routes

| Method | Route | Purpose |
|---|---|---|
| `GET` | `/` | Backend status |
| `POST` | `/upload-contract` | Validate, extract, analyze, chunk, store, and return a contract |
| `POST` | `/ask-contract` | Structured or RAG-grounded contract Q&A |
| `POST` | `/explain-clause` | Explain one selected clause in plain English |
| `GET` | `/contracts` | List saved contracts |
| `GET` | `/contracts/{contract_id}` | Retrieve one stored contract |
| `DELETE` | `/contracts/{contract_id}` | Delete a contract and its chunks |

Comparison and export endpoints follow the active router implementation used by the frontend.

## Testing

Recommended checks:

```cmd
python -m mypy main.py evaluate_external_test.py validate_hybrid_clause_model.py evaluation_strengthening
python -m py_compile main.py validate_hybrid_clause_model.py
python evaluation_strengthening\generate_detailed_evaluation.py
python evaluation_strengthening\run_phase5_ablation.py
uvicorn main:app --reload
```

Also verify:

- Searchable, scanned, and mixed PDFs
- Invalid, oversized, and excessive-page uploads
- Low-confidence manual-review badges
- Structured Q&A and open-ended RAG Q&A
- RAG source snippets and match scores
- Clause explanation
- Contract comparison
- PDF and DOCX export
- Contract persistence and deletion
- Frozen external model evaluation

## Security and Legal Limitations

- API keys must remain in environment variables and `.env` must not be committed.
- Uploaded documents must be validated before extraction or OCR.
- The prototype does not include production authentication, encryption, multi-tenancy, or secure cloud storage.
- The classifier achieves 53.33% external accuracy and can make errors.
- OCR quality depends on scan quality and layout.
- RAG reduces unsupported Gemini answers but generated explanations still require review.
- The system provides legal information and decision support, not professional legal advice.

## Future Work

- Manually audit the 188 provisional Phase-5 labels
- Expand manually reviewed lease data across jurisdictions and drafting styles
- Run full multi-epoch GPU transformer experiments and evaluate legal-domain pretrained models
- Calibrate confidence and abstention thresholds on larger external datasets
- Add a human correction and active-learning interface
- Add authentication, role-based permissions, encryption, and retention controls
- Deploy securely to cloud infrastructure
- Add automated unit, integration, regression, performance, and security testing
- Support DOCX input, multilingual contracts, and additional legal document types

## Author

**Project:** Legal Contract Analyzer  
**Developer:** Kainat Arshad  
**Institution:** Career Institute  
**Program:** BSAI
**Submission Date:** July 20, 2026

## Legal Disclaimer

The Legal Contract Analyzer is an educational and assistive software project. It does not provide professional legal advice and should not be treated as a substitute for consultation with a licensed lawyer. Model predictions and AI-generated explanations may contain errors or may not apply to every legal jurisdiction.
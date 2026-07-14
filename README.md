# Legal Contract Analyzer

## Project Overview

The **Legal Contract Analyzer** is an ML-based legal document analysis system designed to help users review contracts more efficiently.

The application accepts legal contracts in PDF format, extracts their text, identifies individual clauses, classifies clause types, predicts clause risk levels, highlights potentially problematic language, and allows users to ask questions about the uploaded contract.

The system combines machine learning, rule-based validation, OCR, document processing, and generative AI to provide a structured contract review experience.

> **Project Status: Completed and Evaluation-Strengthened**
> All planned development phases have been implemented and tested. Supervisor-requested evaluation strengthening, backend type checking, and the Phase-5 ablation study have also been completed.

---

## Main Features

* Upload and analyze legal contracts in PDF format
* Extract text from searchable PDFs
* Perform OCR on scanned or image-based PDFs
* Clean and preprocess extracted contract text
* Divide contracts into individual clauses
* Predict clause risk levels
* Classify legal clause types
* Apply rule-based validation to improve predictions
* Generate contract summaries
* Identify important and potentially risky clauses
* Ask contract-related questions
* Store analyzed contracts and results in SQLite
* View previously analyzed contracts
* Evaluate ML models using a frozen 330-clause external test set
* Generate 22-class confusion matrices and per-class precision, recall, F1-score, and support
* Run a Phase-5 label ablation study
* Apply static type checking across the active backend with `mypy`
* Display model predictions through a React frontend

---

## System Architecture

The system follows a client-server architecture.

```text
React + Vite Frontend
          |
          v
FastAPI Backend API
          |
          v
PDF Text Extraction / OCR
          |
          v
Text Cleaning and Clause Segmentation
          |
          v
Risk Classification Model
          |
          v
Hybrid Clause-Type Classifier
          |
          v
Rule-Based Validation
          |
          +----------------------+
          |                      |
          v                      v
     SQLite Database        Gemini API
          |                      |
          +----------+-----------+
                     |
                     v
          Results Returned to Frontend
```

---

## Technology Stack

### Frontend

* React
* Vite
* JavaScript
* HTML
* CSS
* Fetch API

### Backend

* Python 3.13
* FastAPI
* Uvicorn
* Pydantic
* SQLite
* Python type hints
* `mypy` static analysis

### Machine Learning

* Scikit-learn
* TF-IDF vectorization
* Supervised text classification
* Hybrid ML and rule-based classification
* Joblib for model persistence

### Document Processing

* PyMuPDF
* Regular expressions
* Text preprocessing
* Clause segmentation
* OCR processing for scanned documents

### AI Integration

* Google Gemini API

### Database

* SQLite

### Data Analysis and Evaluation

* Pandas
* NumPy
* Scikit-learn evaluation metrics
* Confusion matrix
* Classification report
* Accuracy score
* Precision
* Recall
* Macro F1 score
* Weighted F1 score

---

## Application Workflow

1. The user uploads a legal contract through the React frontend.
2. The frontend sends the document to the FastAPI backend.
3. The backend checks whether the PDF contains searchable text.
4. PyMuPDF extracts text from searchable PDFs.
5. OCR is used when a PDF is scanned or contains insufficient text.
6. Extracted text is cleaned and normalized.
7. The contract is divided into individual legal clauses.
8. The risk classification model predicts the risk level of each clause.
9. The clause-type classifier predicts the legal category of each clause.
10. Rule-based validation checks and improves selected predictions.
11. Contract information and analysis results are stored in SQLite.
12. Gemini generates explanations, summaries, and answers to user questions.
13. The processed results are returned to the frontend.
14. The frontend displays the contract summary, clause types, risks, explanations, and related information.

---

## Machine Learning Components

### Risk Classification Model

The risk model predicts whether a legal clause represents a particular level of contractual risk.

The risk categories used by the system may include:

* Low Risk
* Medium Risk
* High Risk

The model processes clause text using TF-IDF features and a supervised machine-learning classifier.

### Clause-Type Classification Model

The clause-type classifier categorizes clauses into legal groups such as:

* Assignment and Subletting
* Confidentiality
* Dispute Resolution
* Governing Law
* Insurance
* Maintenance and Repairs
* Payment Obligations
* Renewal
* Rent
* Security Deposit
* Termination
* Utilities
* General Obligations
* Other supported clause categories

### Hybrid Classification

The final clause classifier combines:

* Machine-learning predictions
* Prediction confidence
* Keyword patterns
* Rule-based legal validation
* Fallback classification logic

This hybrid approach helps improve system reliability when the ML model has low confidence.

---

## Datasets

The project uses a combined legal clause dataset created from multiple sources.

### LEDGAR Dataset

The LEDGAR dataset provides labeled legal clauses from commercial contracts and is used to train the clause-type classification model.

### Lease and Rental Agreement Dataset

A curated lease dataset is used to improve classification for rental and property-related clauses.

The lease dataset includes examples from:

* Residential lease agreements
* Commercial lease agreements
* Rental agreements
* Manually curated clause examples

### Combined Dataset

The final training dataset combines LEDGAR clauses with lease-specific clauses.

Dataset preparation includes:

* Category mapping
* Text cleaning
* Duplicate handling
* Train, validation, and test splits
* Source-document tracking
* External test separation
* Label normalization

---

## Model Evaluation

The clause-type models were evaluated on a **frozen external evaluation set containing 330 manually reviewed clauses**. This dataset was excluded from training and remained unchanged across experiments.

### Final Aggregate Results

| Approach | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Original external baseline | 41.21% | 36.49% | 44.89% |
| Character TF-IDF model | 50.00% | 44.67% | 49.74% |
| Final hybrid ML + rules | **53.33%** | **46.86%** | **53.13%** |

The final hybrid system used:

* **288 ML predictions**
* **42 rule-based corrections**

### Detailed 22-Class Evaluation

Aggregate accuracy was not treated as sufficient evidence of classifier quality. The final evaluation therefore also produced:

* A 22 × 22 confusion matrix
* Per-class precision
* Per-class recall
* Per-class F1-score
* Per-class support
* Macro and weighted averages
* Detailed prediction files

Generated files include:

```text
backend/evaluation/
├── character_model_classification_report.csv
├── character_model_confusion_matrix.csv
├── character_model_confusion_matrix.png
├── final_hybrid_classification_report.csv
├── final_hybrid_confusion_matrix.csv
├── final_hybrid_confusion_matrix.png
├── detailed_evaluation_predictions.csv
├── detailed_evaluation_summary.csv
└── detailed_evaluation_summary.txt
```

### Phase-5 Label Ablation Study

To measure the effect of the 188 provisional automatically labelled Phase-5 lease clauses, two otherwise identical character TF-IDF classifiers were trained:

1. Without the Phase-5 provisional labels
2. With the Phase-5 provisional labels

Both models used the same feature configuration, classifier settings, random seed, preprocessing, class order, and frozen evaluation set.

| Training configuration | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Without Phase-5 labels | 46.97% | 42.37% | 47.54% |
| With Phase-5 labels | **50.00%** | **44.67%** | **49.74%** |
| Difference | **+3.03 points** | **+2.30 points** | **+2.20 points** |

The ablation suggests that the Phase-5 lease clauses provided useful domain adaptation. However, because those labels were automatically generated and not fully manually audited, they remain provisional and should be legally reviewed before production deployment.

Generated ablation files include:

```text
backend/evaluation/
├── ablation_without_phase5_classification_report.csv
├── ablation_with_phase5_classification_report.csv
├── phase5_ablation_summary.csv
├── phase5_ablation_per_class.csv
└── phase5_ablation_summary.txt
```

### Backend Type Safety

Type hints were added consistently across the active backend, including:

* Database functions
* Contract storage
* PDF and OCR processing
* Text cleaning and clause segmentation
* Risk analysis
* Clause-type classification
* Structured answer builders
* API endpoints
* Evaluation scripts

Static analysis was run with:

```cmd
python -m mypy main.py evaluate_external_test.py validate_hybrid_clause_model.py evaluation_strengthening
```

Final result:

```text
Success: no issues found in 5 source files
```

The active files were also syntax-checked with:

```cmd
python -m py_compile main.py validate_hybrid_clause_model.py
```


## OCR Support

The system supports both searchable and scanned PDF contracts.

### Searchable PDFs

Text is extracted directly using PyMuPDF.

### Scanned PDFs

When insufficient text is detected, the application uses OCR to extract text from rendered PDF pages.

The extraction method may be recorded as:

* Text
* OCR
* Mixed

This allows the system to process a wider range of legal documents.

---

## Gemini API Integration

Google Gemini is used for natural-language contract assistance.

Gemini can help the system:

* Generate readable contract summaries
* Explain difficult legal language
* Answer questions about the uploaded contract
* Provide general guidance about identified clauses
* Describe potential concerns
* Produce user-friendly responses based on extracted contract content

Gemini is not used as a replacement for the trained ML models. The ML models perform structured classification, while Gemini provides natural-language explanations and question answering.

---

## Database

SQLite is used to store application data locally.

Stored information may include:

* Contract ID
* Original filename
* Extracted text
* Upload date
* Contract metadata
* Clause analysis results
* Risk predictions
* Clause-type predictions
* Contract history

---

## Project Structure

The exact structure may vary depending on the final version of the project.

```text
Legal-contract/
│
├── backend/
│   ├── main.py
│   ├── database files
│   ├── document extraction modules
│   ├── OCR modules
│   ├── clause processing modules
│   ├── Gemini integration
│   │
│   ├── ml_model/
│   │   ├── training scripts
│   │   ├── trained models
│   │   ├── vectorizers
│   │   ├── datasets
│   │   └── evaluation files
│   │
│   ├── evaluation/
│   │   ├── classification reports
│   │   ├── confusion matrix CSV files
│   │   ├── confusion matrix PNG files
│   │   ├── ablation reports
│   │   └── detailed prediction files
│   │
│   ├── evaluation_strengthening/
│   │   ├── generate_detailed_evaluation.py
│   │   └── run_phase5_ablation.py
│   │
│   ├── mypy.ini
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   └── main.jsx
│   │
│   ├── public/
│   ├── package.json
│   └── vite.config.js
│
├── development_archive/
│   ├── phase summaries
│   ├── evaluation records
│   ├── development notes
│   └── previous implementation files
│
├── README.md
└── .gitignore
```

Adjust this structure so that it matches the actual folders and filenames in your project.

---

## Installation

## 1. Clone or Open the Project

```bash
git clone YOUR_REPOSITORY_URL
cd Legal-contract
```

When running the project locally without Git, open the main project folder in Visual Studio Code.

---

## 2. Backend Setup

Open Command Prompt and navigate to the backend folder:

```cmd
cd "C:\Users\YourName\Documents\Legal-contract\backend"
```

Create a virtual environment:

```cmd
python -m venv venv
```

Activate the virtual environment:

```cmd
venv\Scripts\activate
```

Install the backend dependencies:

```cmd
pip install -r requirements.txt
```

---

## 3. Environment Variables

Create a `.env` file inside the backend folder.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

Do not upload the `.env` file or expose API keys in the source code.

---

## 4. Start the Backend

From the backend directory, run:

```cmd
python -m uvicorn main:app
```

The backend should be available at:

```text
http://127.0.0.1:8000
```

FastAPI documentation should be available at:

```text
http://127.0.0.1:8000/docs
```

---

## 5. Frontend Setup

Open a second Command Prompt window and navigate to the frontend folder:

```cmd
cd "C:\Users\YourName\Documents\Legal-contract\frontend"
```

Install frontend dependencies:

```cmd
npm install
```

Start the frontend:

```cmd
npm run dev
```

Vite will display the local frontend URL, normally:

```text
http://localhost:5173
```

Open this address in a web browser.

---

## API Endpoints

The final backend exposes the following main endpoints:

| Method   | Endpoint                   | Purpose                          |
| -------- | -------------------------- | -------------------------------- |
| `GET`    | `/`                        | Check backend status             |
| `POST`   | `/upload-contract`         | Upload and process a contract    |
| `POST`   | `/ask-contract`            | Ask a question about a contract  |
| `GET`    | `/contracts`               | Return stored contract history   |
| `GET`    | `/contracts/{contract_id}` | Return one stored contract       |
| `DELETE` | `/contracts/{contract_id}` | Delete a stored contract         |


---

## Recommended Screenshots

Add the following screenshots to the final documentation or README:

1. Application landing page
2. Contract upload interface
3. Uploaded contract information
4. Contract summary
5. Risk overview
6. Clause analysis results
7. High-risk clause display
8. Clause-type classification
9. Contract question-and-answer feature
10. Contract history page
11. Search functionality
12. Scanned PDF or OCR result
13. FastAPI Swagger documentation
14. Backend terminal while running
15. Frontend terminal while running
16. Model evaluation output
17. Confusion matrix
18. External prediction results

Create an `images` or `screenshots` folder and add images using Markdown:

```markdown
![Contract Upload Screen](screenshots/contract-upload.png)
```

---

## Testing

Before final submission, run the following technical checks:

```cmd
python -m mypy main.py evaluate_external_test.py validate_hybrid_clause_model.py evaluation_strengthening
python -m py_compile main.py validate_hybrid_clause_model.py
python evaluation_strengthening\generate_detailed_evaluation.py
python evaluation_strengthening\run_phase5_ablation.py
python -m uvicorn main:app
```

Then test the system using:

* A searchable PDF
* A scanned PDF
* A short contract
* A long contract
* A contract containing high-risk clauses
* A document with missing information
* An unsupported or invalid file
* A contract-related question
* A non-contract-related question
* Contract history loading
* Database persistence after restarting the backend

---

## Security Considerations

* API keys must be stored in environment variables.
* The `.env` file must not be committed to Git.
* Uploaded files should be validated.
* File size restrictions should be enforced.
* Unsupported file formats should be rejected.
* User input should be validated before processing.
* Generated legal responses should clearly display a disclaimer.

---

## Limitations

* The system does not replace a qualified lawyer.
* Predictions depend on the quality and coverage of the training dataset.
* OCR accuracy may decrease for blurry or low-resolution documents.
* Some uncommon clauses may be classified incorrectly.
* Gemini responses may occasionally contain incomplete or inaccurate information.
* The system may not support every legal jurisdiction.
* The external evaluation dataset may contain categories with limited examples.
* Low-frequency clause categories may have lower classification performance.

---

## Future Improvements

Possible future improvements include:

* Add user authentication
* Add role-based access
* Add cloud database support
* Deploy the frontend and backend
* Support DOCX and TXT contracts
* Add multilingual contract analysis
* Improve OCR preprocessing
* Train transformer-based legal language models
* Expand the legal clause dataset
* Add jurisdiction-specific legal rules
* Add contract comparison
* Add clause recommendation features
* Export analysis as PDF
* Add admin analytics
* Add subscription and payment functionality
* Add secure cloud document storage

---

## Legal Disclaimer

The Legal Contract Analyzer is an educational and assistive software project.

It does not provide professional legal advice and should not be treated as a substitute for consultation with a licensed lawyer. Model predictions and AI-generated explanations may contain errors or may not apply to every legal jurisdiction.

Users should consult a qualified legal professional before making decisions based on a contract or the results produced by this system.

---

## Author

**Project:** Legal Contract Analyzer
**Developer:** Kainat Arshad
**Institution:** Add your university or institution
**Program:** Add your degree or course
**Submission Date:** July 2026

---

## License

This project was developed for educational and academic purposes.

Add a formal open-source license only when required by your institution or when publishing the project publicly.

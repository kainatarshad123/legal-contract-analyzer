LEGAL CONTRACT ANALYZER — TRANSFORMER CLAUSE CLASSIFIER

Purpose
-------
Fine-tune DistilBERT as a 22-class clause-type classifier, evaluate it on the
same frozen 330-clause external set, and compare it with the existing
Character TF-IDF and hybrid baselines.

Recommended environment
-----------------------
Use a separate Python 3.12 virtual environment so the working FastAPI backend
environment remains unchanged.

From the project root:

py -3.12 -m venv transformer_venv
transformer_venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r transformer_training\requirements-transformer.txt

Place datasets
--------------
Keep the two source CSV files in backend\ml_model:

ml_model\combined_clause_type_dataset_v2.csv
ml_model\external_test_clauses_final.csv

Step 1 — Inspect data
---------------------

python transformer_training\inspect_transformer_data.py ^
  --train-csv ml_model\combined_clause_type_dataset_v2.csv ^
  --eval-csv ml_model\external_test_clauses_final.csv

Expected:
- Training rows: 6486
- Train split: 5041
- Validation split: 749
- Test split: 696
- Training labels: 22
- Frozen external evaluation rows: 330
- Duplicate counts: 0
- Text overlap: 0

Step 2 — Prepare data
---------------------

python transformer_training\prepare_transformer_dataset.py ^
  --train-csv ml_model\combined_clause_type_dataset_v2.csv ^
  --eval-csv ml_model\external_test_clauses_final.csv

Step 3 — Train DistilBERT
------------------------

GPU/default:

python transformer_training\train_transformer_classifier.py

Low-memory GPU:

python transformer_training\train_transformer_classifier.py ^
  --train-batch-size 4 ^
  --eval-batch-size 8 ^
  --gradient-accumulation 4 ^
  --max-length 320

CPU training is possible but may be slow. The scripts do not modify the
existing TF-IDF model files.

Step 4 — Frozen external evaluation
-----------------------------------

python transformer_training\evaluate_transformer_classifier.py

Outputs:
- transformer_external_predictions.csv
- transformer_classification_report.csv
- transformer_confusion_matrix.csv
- transformer_external_summary.json

Step 5 — Compare with existing baselines
----------------------------------------

python transformer_training\compare_clause_models.py

Current baselines:
- Character TF-IDF accuracy: 50.00%
- Character TF-IDF Macro F1: 44.67%
- Hybrid accuracy: 53.33%
- Hybrid Macro F1: 46.86%
- Hybrid Weighted F1: 53.13%

Do not replace the production classifier until the frozen external evaluation
and runtime comparison are complete.

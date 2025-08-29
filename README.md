# Student-Programming-Assessment
Student Programming Evaluation Pipeline

This repository provides Python scripts to process and analyze CSEDM Data Challenge (ProgSnap2) datasets.
It includes data cleaning utilities, baseline models (logistic regression), and deep learning models (Transformer + CNN hybrid).

Requirements

Python 3.9+

Install dependencies:

pip install -U pip
pip install pandas numpy scikit-learn matplotlib torch tqdm statsmodels scipy


If you have a GPU, install the CUDA build of PyTorch from pytorch.org
.

Data

Download the official CSEDM ProgSnap2 datasets:

Fall 2019: F19_Release_All_05_23_22.zip

Spring 2019: S19_All_Release_2_10_22.zip

Place the zip files in your working directory. The scripts will automatically extract them.

Usage
1. Clean raw dataset

Flatten the ProgSnap2 zip into a single CSV:

python csedm_cleaner.py --zip F19_Release_All_05_23_22.zip --out F19_all_events.csv


This produces F19_all_events.csv with normalized columns like:

student_id, problem_id, assignment_id, submission_id, codestate_id, timestamp, score, event_type, attempt, code_state

2. Quick experiment (lightweight)

Run all models on a sampled subset of the data (fast mode, CPU friendly):

python aete_deep_models_pro.py --zip-f19 F19_Release_All_05_23_22.zip --quick --outdir outputs_quick


Outputs go to outputs_quick/:

CSV metrics per fold

Global pooled metrics

ROC/PR curves

Confusion matrices

Learning curves

3. Full experiment (slower, better)

Use the full dataset and deeper models (recommended with GPU):

python aete_deep_models_pro.py --zip-f19 F19_Release_All_05_23_22.zip --epochs 20 --max-evt-len 400 --max-code-len 2000 --batch-size 64 --outdir outputs_pro

4. Cross-semester generalization

Train on Fall 2019 and test on Spring 2019:

python aete_deep_models_pro.py --zip-f19 F19_Release_All_05_23_22.zip --zip-s19 S19_All_Release_2_10_22.zip --quick --outdir outputs_cross


Outputs:

cross_semester_metrics.csv

Cross-semester ROC/PR plots

Confusion matrices

Notes

--quick is useful for debugging and testing (uses a subset of data, fewer epochs).

On CPU, full deep models can take many hours. For fast turnaround, use --quick.

Logistic Regression baselines are included and run automatically alongside deep models.

Plots and CSVs are saved in the specified --outdir.

Example Outputs

*_cv_metrics_summary.csv — mean metrics across folds

*_ROC_cv.png / *_PR_cv.png — ROC/PR curves with shaded variance

*_confusion.png — confusion matrices (pooled predictions)

*_metrics_cross.csv — cross-semester evaluation results

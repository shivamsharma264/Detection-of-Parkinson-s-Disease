# Parkinson's Disease Detection

An educational machine-learning project using the public UCI Parkinson's voice
measurements dataset. It compares Logistic Regression and Random Forest models.
It is not a medical diagnostic tool and must not be used for medical decisions.

## Project contents

- `notebooks/parkinsons_detection.ipynb` — explained analysis, visualizations,
  model training, and evaluation.
- `train_model.py` — reusable training and evaluation command.
- `app.py` — Streamlit demonstration using a locally trained model.
- `tests/test_pipeline.py` — tests using explicitly synthetic software fixtures;
  these fixtures are not patient records or clinical results.
- `data/README.md` — authoritative dataset source and setup instructions.

## 1. Set up on Windows

Open PowerShell in this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell does not allow activation, do not change the machine's execution
policy; run commands using the project interpreter directly, for example:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 2. Download the real dataset

Download the official UCI Parkinson's dataset from
<https://archive.ics.uci.edu/static/public/174/parkinsons.zip>. Save the ZIP in
this project's `data` folder and extract it there. The resulting file must be
`data\parkinsons.data`. More details and expected columns are in
[`data/README.md`](data/README.md).

The dataset is not bundled with this project. Until you download it, training,
real-data evaluation, and a meaningful app prediction are unavailable.

## 3. Train and evaluate

From the project folder, with its environment active:

```powershell
python train_model.py
```

The script reports a stratified holdout evaluation and saves both fitted
preprocessing/model pipelines to
`submission\parkinsons_models.joblib`. It refuses to train if the real dataset
is missing or malformed; it does not substitute synthetic examples.

## 4. Run tests

```powershell
python -m pytest -q
```

Tests use deterministic synthetic fixtures only to check software behavior.
They do not represent patients, measure clinical performance, or replace
training and evaluation on the actual UCI dataset.

## 5. Execute and export the notebook

After downloading the dataset and installing dependencies, run all notebook
cells and export the executed notebook, including its code, outputs, and charts:

```powershell
jupyter nbconvert --to html --execute .\notebooks\parkinsons_detection.ipynb --output parkinsons_detection.html --output-dir .\submission --ExecutePreprocessor.timeout=600
```

Do not treat an export made without the dataset as a completed analysis. The
notebook explains the missing-data case and does not invent evaluation results.

## 6. Launch the educational app

Train the model first, then run:

```powershell
streamlit run app.py
```

The app displays an explicit medical-use warning. Its numeric inputs must match
the UCI features and units; placeholder defaults are not real measurements.

## Important limitations

This dataset is small and imbalanced, so a single train/test split can produce
unstable estimates. Voice measurements alone are not a diagnosis. Results may
not generalize to other people, recording devices, languages, or settings.
Precision, recall, F1-score, confusion matrices, and ROC-AUC (when calculable)
describe this dataset split only—not clinical effectiveness.
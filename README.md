# ML / DL Comparison Dashboard

Clean, modern, white Streamlit dashboard for the supplementary Machine Learning vs Deep Learning benchmark of the final project.

## What is included

- `app.py` — ready-to-run Streamlit application
- `requirements.txt` — deployment dependencies
- `.streamlit/config.toml` — light theme
- `data/hasil_komparasi_ML_vs_DL_lengkap.xlsx` — benchmark tables
- `data/predictions_test_CNN.csv` — held-out predictions for the best DL model
- `data/predictions_test_Linear_SVM.csv` — held-out predictions for the best ML model
- `assets/` — pipeline, learning-curve, and confusion-matrix figures

## Dashboard pages

1. Overview
2. Performance
3. Per Label
4. Diagnostics
5. Prediction Explorer
6. Method

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Extract this ZIP.
2. Create a GitHub repository.
3. Upload **all files and folders inside this project folder** to the repository root.
4. Open Streamlit Community Cloud.
5. Create a new app from the GitHub repository.
6. Set the entrypoint to:
   `app.py`
7. Deploy.

No secrets, model weights, GPU, API key, or Google Drive mount are required.

## Important

This dashboard is intentionally read-only. It visualizes the saved results of the completed comparison run and does not retrain the models or perform live inference.

The benchmark uses final TA silver labels. Scores should not be interpreted as clinical diagnostic accuracy.

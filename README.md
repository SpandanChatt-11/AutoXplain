# AutoXplain — Explainable Used Car Valuation System

An end-to-end machine learning system that predicts used car prices in the Kolkata market and explains **why** it made each prediction — not just what the number is.

**Live Demo**: [https://autoxplain.onrender.com](https://autoxplain.onrender.com)

> Note: hosted on Render's free tier — the first request after ~15 minutes of inactivity takes about 60 seconds to wake up. Subsequent requests are fast.

---

## What this project does

Most used-car price predictors stop at a trained model and an accuracy score. AutoXplain goes further: every prediction comes with a **feature-level explanation** (via LIME), and the model selection process itself was validated with **global explainability** (via SHAP) across four independently trained models — so the conclusions aren't an artifact of any single algorithm's quirks.

The full pipeline — scraping, cleaning, modeling, explainability, and deployment — was built from scratch on real, messy, live web data rather than a pre-cleaned dataset.

---

## Pipeline overview

```
Web Scraping (Selenium)  →  Data Cleaning & Feature Engineering  →  EDA
        →  Model Training + Hyperparameter Tuning  →  Explainability (SHAP + LIME)
        →  Deployment (Flask + Render)
```

### 1. Data Collection
- Scraped ~1,100 real used-car listings from CarDekho (Kolkata) using **Selenium** (the site is JavaScript-rendered, so a static request/BeautifulSoup approach wasn't viable).
- Handled real-world scraping challenges: JS-rendered pagination, lazy-loaded content requiring scroll-triggered loading, and anti-bot measures (randomized delays, `navigator.webdriver` patching).

### 2. Data Cleaning & Feature Engineering
- Deduplicated on listing URL (raw data had a ~37% duplicate rate from overlapping paginated results).
- Resolved missing values per-column based on missingness ratio and available fallback sources (e.g., cross-referencing the listing card against the detail page).
- Parsed inconsistent price formats (Lakh / Crore / Thousand suffixes), extracted `Car_Age`, `Owner_Number`, and `Brand` from raw fields, and geocoded locality names into latitude/longitude via Nominatim.
- **Caught and excluded a target-leakage feature** (EMI, which CarDekho computes directly from price) before it could contaminate the model.

### 3. Exploratory Data Analysis
- Distribution analysis, correlation heatmaps, and feature-vs-price relationships across fuel type, transmission, ownership, brand, and geography.

### 4. Modeling
Four models spanning different algorithmic paradigms, each tuned via cross-validated search:

| Model | Type | Test R² | RMSE (₹) |
|---|---|---|---|
| **XGBoost** | Boosting ensemble | **0.71** | ~3.19L |
| Random Forest | Bagging ensemble | 0.66 | ~3.47L |
| Ridge | Linear baseline | 0.64 | ~3.56L |
| SVR | Kernel-based | 0.59 | ~3.79L |

- Target variable (`Price`) was **log-transformed** before training — price is right-skewed and depreciation is multiplicative, not additive, so `log(price) ~ linear(age)` fits far better than the raw scale. This one change alone took SVR's R² from -0.06 (worse than predicting the mean) to 0.59.
- Hyperparameters tuned via `RandomizedSearchCV` / `GridSearchCV` with 5-fold cross-validation.

### 5. Explainability
- **Global SHAP** analysis on every model, using the correct explainer for each architecture: `TreeExplainer` (Random Forest, XGBoost), `LinearExplainer` (Ridge), `KernelExplainer` (SVR).
- `Car_Age` and `Transmission` emerged as the top two price drivers **consistently across all four models** — a much stronger signal than any single model's feature importance alone.
- **Local LIME** explanations generated in real time in the deployed app, so every individual prediction is accompanied by a plain-language breakdown of which inputs pushed the price up or down.

### 6. Deployment
- Served with **Flask** + **Gunicorn**, deployed on **Render** (free tier).
- Interactive form with dropdowns for categorical fields (Brand, Fuel Type, Transmission), returning a predicted price plus an embedded LIME explanation chart.

---

## Tech Stack

- **Scraping**: Selenium, ChromeDriver
- **Data processing**: Pandas, NumPy, Geopy
- **Modeling**: scikit-learn, XGBoost
- **Explainability**: SHAP, LIME
- **Visualization**: Matplotlib, Seaborn
- **Deployment**: Flask, Gunicorn, Render

---

## Project Structure

```
project_folder/
├── app/                      # Deployed Flask application
│   ├── app.py
│   ├── models/               # Trained model artifacts (bundled for deployment)
│   ├── requirements.txt
│   └── README.md
├── models/                   # Trained models, scaler, feature list, LIME background data
├── notebooks/                # Scraping, preprocessing/EDA, and model training notebooks
├── scraped data/              # Raw scraped CSVs
├── mlready data/              # Cleaned, feature-engineered dataset
└── output/                   # Scraping run backups
```

---

## Running Locally

**1. Clone the repo**
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>/app
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
python app.py       # local development (works on Windows, Mac, Linux)
```
or, for a production-style run (Mac/Linux only — `gunicorn` doesn't support Windows):
```bash
gunicorn app:app --bind 127.0.0.1:5000
```

Then open `http://127.0.0.1:5000`.

---

## Results Summary

- **Best model**: XGBoost, achieving R² = 0.71 on held-out test data.
- **Cross-model agreement**: Car age and transmission type were the top two predictive features across every model tested — linear, bagging, boosting, and kernel-based alike.
- **Data leakage avoided**: EMI was identified as a derived (not independent) feature and explicitly excluded from the model, preventing an artificially inflated — but meaningless — accuracy score.

---

## Limitations & Future Work

- Dataset is scoped to Kolkata listings only (615 rows after cleaning) — more cities and more listings would improve generalization.
- Prices reflect **listed asking price**, not verified final sale price.
- Planned improvements: larger-scale scraping, additional cities, log-transforming skewed input features (not just the target), and drift monitoring on the deployed app.

---

## License

This project was built for educational and portfolio purposes.

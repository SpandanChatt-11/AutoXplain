# Kolkata Used Car Price Predictor — Render Deployment

## Before deploying

1. Copy your **real** trained artifacts into a `models/` folder next to `app.py`:
   - `best_model.pkl`, `scaler.pkl`, `feature_names.pkl` (from the training notebook)
   - `X_train_background.csv` — run this in your training notebook if you haven't:
     ```python
     X_train.to_csv("models/X_train_background.csv", index=False)
     ```

2. **Confirm the 13th brand.** In `app.py`, `BRAND_OPTIONS` currently has 11 named brands + "Other"
   (derived from your ML-ready dataset's columns). One brand — the one-hot baseline dropped by
   `pd.get_dummies(drop_first=True)` — is missing because it can't be determined from column names
   alone. Run this in your training notebook (or preprocessing notebook, wherever `top_brands` is
   still in memory) and add the missing name to `BRAND_OPTIONS`:
   ```python
   print(top_brands.tolist())
   ```

3. Similarly double check `FUEL_OPTIONS` — `"CNG"` was inferred as the dropped baseline (the only
   fuel type alphabetically before "Diesel"). Confirm against your actual `Fuel Type` unique values.

## Deploying to Render (free tier)

1. Push this folder (`app.py`, `requirements.txt`, `models/`) to a GitHub repo.
2. On [render.com](https://render.com): New → Web Service → connect your repo.
3. Settings:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`
4. Deploy. First load after any 15-minute idle period takes ~60 seconds (free-tier cold start) — this is
   expected, not a bug.

## Local testing before deploying

```bash
pip install -r requirements.txt
gunicorn app:app --bind 127.0.0.1:5000
```
Then open http://127.0.0.1:5000

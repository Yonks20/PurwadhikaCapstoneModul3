# 🚀 Quick Start Guide — Hotel Booking Cancellation Prediction

Follow these steps to get the project running on your local machine or deploy to GitHub.

---

## 1️⃣ Clone & Setup

### Option A: From GitHub (after pushing)

```bash
# Clone repository
git clone https://github.com/your-username/hotel-booking-cancellation.git
cd hotel-booking-cancellation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Option B: Local Development (before pushing to GitHub)

```bash
# Navigate to project folder
cd /path/to/hotel-booking-cancellation

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 2️⃣ Run the Notebook

```bash
# Start Jupyter
jupyter notebook

# Open: HotelBookingDemand_Capstone_Yonkie.ipynb
# Run all cells in sequence
```

The notebook will:
- Load and clean data (~10k bookings)
- Perform EDA (distributions, correlations)
- Preprocess and scale features
- Train & benchmark 7 models
- Analyze threshold tradeoffs
- Export trained model (pickle)

**Expected runtime:** ~3–5 minutes on standard laptop

---

## 3️⃣ Make Predictions

Once model is trained:

```python
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load pipeline & model
pipeline = joblib.load('preprocessing_pipeline.pkl')
model = joblib.load('xgboost_model.pkl')

# Example booking
booking = pd.DataFrame({
    'country': ['PRT'],
    'market_segment': ['Online TA'],
    'deposit_type': ['No Deposit'],
    'customer_type': ['Transient'],
    'reserved_room_type': ['A'],
    'previous_cancellations': [0],
    'booking_changes': [2],
    'days_in_waiting_list': [5],
    'required_car_parking_spaces': [0],
    'total_of_special_requests': [3]
})

# Transform & predict
booking_transformed = pipeline.transform(booking)
prob = model.predict_proba(booking_transformed)[0][1]

print(f"Cancellation risk: {prob:.1%}")
if prob > 0.5:
    print("⚠️ HIGH RISK — Contact guest")
else:
    print("✅ LOW RISK — Monitor")
```

---

## 4️⃣ Project Structure Setup

Organize your local folder like this before pushing to GitHub:

```
hotel-booking-cancellation/
├── README.md                          ← You are here
├── QUICKSTART.md                      ← This file
├── .gitignore                         ← What NOT to push
├── requirements.txt                   ← Dependencies
│
├── notebooks/
│   └── HotelBookingDemand_Capstone_Yonkie.ipynb
│
├── data/
│   └── data_hotel_booking_demand.csv  ← Raw data (if public)
│
├── models/
│   ├── xgboost_model.pkl              ← Trained model
│   ├── preprocessing_pipeline.pkl     ← Scaler + encoder
│   └── threshold_analysis.json        ← Tuning results
│
├── presentations/
│   ├── Prediksi_Pembatalan_Hotel_Capstone3_Yonkie.pptx
│   └── Prediksi_Pembatalan_Hotel_Capstone3_FINAL_v2_fixed.pptx
│
└── outputs/
    ├── confusion_matrix.png
    ├── feature_importance.png
    └── roc_curve.png
```

**Pro Tip:** Create these folders before first git commit:
```bash
mkdir -p notebooks data models presentations outputs
# Move/organize files accordingly
```

---

## 5️⃣ Push to GitHub

### First Time Setup

```bash
# Initialize git repo (if not already done)
git init

# Add all files (except .gitignore'd ones)
git add .

# First commit
git commit -m "Initial commit: Hotel booking cancellation prediction model"

# Add remote (replace with your GitHub URL)
git remote add origin https://github.com/your-username/hotel-booking-cancellation.git

# Push to GitHub
git branch -M main
git push -u origin main
```

### Subsequent Updates

```bash
# Make changes to files
git add .
git commit -m "Description of changes"
git push origin main
```

### What Gets Pushed? (What Gets Ignored?)

**✅ PUSHED to GitHub:**
- `README.md` — Overview & documentation
- `QUICKSTART.md` — Setup guide
- `.gitignore` — Git config
- `requirements.txt` — Dependencies
- `notebooks/*.ipynb` — Jupyter notebook
- `presentations/*.pptx` — PowerPoint files
- `outputs/*.png` — Visualizations

**❌ NOT PUSHED (in .gitignore):**
- `data/*.csv` — Large raw data (keep local or use Git LFS)
- `models/*.pkl` — Large trained models (use GitHub Releases instead)
- `__pycache__/` — Python cache
- `.ipynb_checkpoints/` — Notebook checkpoints
- `.venv/` — Virtual environment

---

## 6️⃣ Share Trained Model

If model file is too large (>100MB) for GitHub:

### Option A: GitHub Releases

```bash
# Tag a version
git tag -a v1.0 -m "Initial model release"

# Push tag
git push origin v1.0

# Go to GitHub → Releases → Upload model file manually
# Users can download via: 
# https://github.com/your-username/repo/releases/download/v1.0/xgboost_model.pkl
```

### Option B: GitHub Large File Storage (Git LFS)

```bash
# Install git-lfs
# Then track large files
git lfs track "models/*.pkl"
git add .gitattributes
git commit -m "Enable Git LFS for model files"
git push
```

### Option C: Separate Cloud Storage

- Upload model to Google Drive, Dropbox, or S3
- Link in README with download instructions
- Keep notebook self-contained (users train their own copy)

---

## 7️⃣ Update README on GitHub

GitHub automatically displays `README.md` on your repo's homepage:

1. Edit `README.md` locally
2. Commit & push
3. Refresh GitHub — README updates instantly (no rebuild needed)

**Markdown renders automatically** — no HTML or special config required.

---

## 🔄 Typical Workflow

1. **Develop locally:**
   ```bash
   # Activate venv
   source venv/bin/activate
   
   # Edit notebook or Python files
   jupyter notebook
   
   # Test locally
   python test_model.py
   ```

2. **Commit & push:**
   ```bash
   git add .
   git commit -m "Improve model performance to 86% recall"
   git push origin main
   ```

3. **GitHub displays:**
   - README on homepage
   - Notebook rendered (interactive preview)
   - All markdown files nicely formatted

---

## 📋 Checklist Before First Push

- [ ] `README.md` is comprehensive and up-to-date
- [ ] `requirements.txt` lists all dependencies
- [ ] `.gitignore` is present (ignore data, cache, venv)
- [ ] Notebook runs without errors start-to-finish
- [ ] No hardcoded file paths (use relative paths instead)
- [ ] Add comments explaining key steps
- [ ] Remove any API keys, passwords, or secrets
- [ ] Folder structure is organized

---

## ❓ Common Questions

**Q: Does GitHub automatically run my notebook?**  
A: No — GitHub just displays it as a static preview. Users must run `jupyter notebook` locally to execute.

**Q: Can I update README without re-uploading the notebook?**  
A: Yes! Edit `README.md`, commit, and push. GitHub updates the homepage instantly.

**Q: Should I track `.pkl` files?**  
A: Only if <100MB. For larger models, use Releases or cloud storage (see Section 6).

**Q: What if I forget `.gitignore`?**  
A: Create it now, then run:
   ```bash
   git rm -r --cached .  # Remove tracked cache files
   git add .
   git commit -m "Add .gitignore and remove cache"
   git push
   ```

**Q: How do I update the model after retraining?**  
A: Retrain locally, overwrite `.pkl` file, commit, and push:
   ```bash
   git add models/xgboost_model.pkl
   git commit -m "Retrain model with expanded dataset"
   git push
   ```

---

## 📚 Next Steps

1. **Set up GitHub repo** (create on github.com if not already done)
2. **Clone/organize files locally** per structure in Section 4
3. **Run notebook end-to-end** to verify everything works
4. **Commit & push** first version
5. **Share link** with stakeholders
6. **Iterate:** Update notebook, retrain, push improvements

---

**Happy deploying! 🎉**

For more info, see [README.md](./README.md) or GitHub Docs: https://docs.github.com/en/repositories


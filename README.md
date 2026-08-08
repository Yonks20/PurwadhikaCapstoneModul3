# 🏨 Hotel Booking Cancellation Prediction

**Supervised binary classification model to detect high-risk bookings and enable proactive Revenue Management intervention**

> Capstone Project Module 3 | Purwadhika Digital Technology School  
> Author: Yonkie Yudha Ardika

---

## 📋 Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset Overview](#dataset-overview)
- [Methodology](#methodology)
- [Results](#results)
- [Key Findings](#key-findings)
- [How to Use](#how-to-use)
- [Project Structure](#project-structure)
- [Files & Deliverables](#files--deliverables)
- [Tech Stack](#tech-stack)
- [Limitations & Future Work](#limitations--future-work)

---

## 🎯 Problem Statement

### The Business Context

Hotel networks face a critical operational challenge: **approximately 23.8% of bookings are canceled** — roughly 1 in 4 rooms marked as "sold" become empty without warning. Since hotel rooms cannot be inventoried like physical goods, an empty room tonight cannot be resold tomorrow.

### Current State

- No systematic way to identify which bookings are at risk of cancellation
- All reservations treated equally; retention efforts are reactive, not proactive
- Cancellations are discovered *after* they occur, when it's too late to reallocate the room

### Solution

Build a **machine learning model** to:
1. Identify high-risk bookings before the check-in date
2. Enable Revenue Management and Customer Service teams to intervene proactively
3. Prioritize outreach based on cancellation probability

### Why Machine Learning?

Cancellation patterns involve **complex interactions** of many factors (deposit type, booking channel, guest profile, length of stay) that are difficult to capture via manual rules — and the volume is too large for staff to assess individually.

---

## 📊 Dataset Overview

| Aspect | Details |
|--------|---------|
| **Total Records** | 10,097 bookings (after cleaning) |
| **Target Variable** | `is_canceled` (binary: 1 = canceled, 0 = completed) |
| **Target Distribution** | 76.2% completed, 23.8% canceled |
| **Features** | 10 predictors (5 categorical, 5 numerical) |
| **Missing Values** | None (post-cleaning) |
| **Duplicates** | None (post-cleaning) |

### Feature Breakdown

**Categorical Features:**
- `country` — Guest origin (162 unique countries; Portugal dominates with 19.9%)
- `market_segment` — Booking channel (Online TA, Direct, Corporate, etc.)
- `deposit_type` — Financial guarantee (No Deposit: 97.6%, Refundable, Non-Refund)
- `customer_type` — Reservation type (Transient, Transient-Party, Contract, Group)
- `reserved_room_type` — Room type code (10 variants; A and B most common)

**Numerical Features:**
- `previous_cancellations` — Guest's historical cancellation count
- `booking_changes` — Number of modifications to the reservation
- `days_in_waiting_list` — Days between booking and arrival
- `required_car_parking_spaces` — Parking spaces requested
- `total_of_special_requests` — Special requests count (positively correlated with commitment)

### Data Characteristics

⚠️ **Important**: These features capture *booking characteristics only* — not temporal dynamics (e.g., time-to-arrival, price trends) or guest identity tracking. This inherently limits model ceiling but ensures practical deployment feasibility.

---

## 🔍 Methodology

### 1. **Data Cleaning & Preparation**

- Handled 351 missing values in `country` (0.42%) via mode imputation
- Removed duplicate records
- Validated data types and consistency
- No outlier removal — values like "26 previous cancellations" or "391-day waiting list" retain operational signal

### 2. **Exploratory Data Analysis (EDA)**

- Analyzed class imbalance: positive class (cancellation) = 23.8%
- Examined feature distributions and correlations
- Identified skewed numerical features (most are median = 0)
- Segmented analysis by deposit type, booking channel, and customer type

### 3. **Feature Engineering & Preprocessing**

- **Scaling:** StandardScaler for numerical features (due to high variance/skewness)
- **Encoding:** One-Hot Encoding for categorical features
- **Handling Imbalance:** Evaluated class_weight strategies in models
- **Pipeline:** Built reusable `Pipeline` object for train-test consistency

### 4. **Model Selection & Benchmarking**

Tested 7 classification algorithms:

| Algorithm | AUC-ROC | Recall | Precision | F1-Score |
|-----------|---------|--------|-----------|----------|
| **Logistic Regression** | 0.816 | 0.823 | 0.441 | 0.577 |
| **Decision Tree** | 0.829 | 0.852 | 0.471 | 0.607 |
| **Random Forest** | 0.836 | 0.837 | 0.485 | 0.615 |
| **Gradient Boosting** | 0.839 | 0.843 | 0.491 | 0.621 |
| **XGBoost** | 0.846 | 0.863 | 0.454 | 0.595 |
| **LightGBM** | 0.837 | 0.851 | 0.482 | 0.616 |
| **Voting Classifier** | 0.843 | 0.860 | 0.488 | 0.619 |

### 5. **Threshold Tuning**

Analyzed the tradeoff between Recall and Precision across thresholds:

- **Threshold 0.3:** Recall 93.6%, Precision 41.1%, ~1,096 bookings to contact → High sensitivity, resource-heavy
- **Threshold 0.5** (selected): Recall 86.3%, Precision 45.4%, ~915 bookings to contact → Balanced operability
- **Threshold 0.7:** Recall 50.5%, Precision 50.5%, ~424 bookings to contact → Conservative, may miss real cancellations

**Decision Rationale:**
- False Negative (missed cancellation) = **permanent revenue loss** ($$$)
- False Positive (flagged as risky, but actually completed) = **low-cost intervention** (~cost of one phone call)
- Therefore: **Recall is prioritized**, but Precision is monitored to respect team capacity

---

## 📈 Results

### Final Model Performance (XGBoost on Test Set)

| Metric | Value |
|--------|-------|
| **Recall** | 0.863 (86.3%) |
| **Precision** | 0.454 (45.4%) |
| **F1-Score** | 0.595 |
| **AUC-ROC** | 0.846 |
| **Accuracy** | 0.762 |

⚠️ **Note on Accuracy:** Not used as primary metric because 76.2% of data is negative class — a "dummy" model predicting "no cancellation" for everything achieves 76% accuracy without any predictive value.

### Confusion Matrix (Test Set)

```
                Predicted No    Predicted Yes
Actual No:      2,066           297
Actual Yes:     66              415
```

- **True Positives (TP):** 415 — correctly identified cancellations
- **False Negatives (FN):** 66 — missed cancellations (biggest business risk)
- **False Positives (FP):** 297 — flagged as risky but completed

### Business Impact Simulation

At **Threshold 0.5**, model captures **86.3% of real cancellations**:
- Out of 481 bookings that actually cancel, **415 are caught by the model**
- Only **66 slip through** (revenue loss cannot be recovered)
- Team must contact ~915 bookings per period to reach the 415 at risk

---

## 🔑 Key Findings

### Feature Importance (XGBoost Top 10)

1. **`days_in_waiting_list`** — Longer waiting list = higher cancellation risk
2. **`previous_cancellations`** — Guest's past behavior is the strongest predictor
3. **`customer_type`** — Different guest types have different cancellation propensity
4. **`required_car_parking_spaces`** — Parking commitment correlates with stay commitment
5. **`total_of_special_requests`** — More requests = less likely to cancel
6. **`booking_changes`** — Frequency of modifications signals engagement
7. **`deposit_type`** — No-deposit bookings higher risk than refundable/non-refund
8. **`market_segment`** — Booking channel (Direct vs. TA) influences cancellation rate
9. **`country`** — Geographic origin has minor signal
10. **`reserved_room_type`** — Specific room type preferences matter

### Operational Insights

- **Deposit Type Effect:** 97.6% of bookings have no deposit. Non-refund deposits reduce cancellation rate substantially.
- **Special Requests as Commitment:** More special requests (e.g., "high floor," "late check-in") = stronger intention to actually stay.
- **Waiting List Duration:** Bookings placed far in advance carry higher risk — many plans change.
- **Repeat Guest Risk:** Guests with previous cancellations are significantly more likely to cancel again.

---

## 💻 How to Use

### Prerequisites

```bash
Python 3.8+
pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn
```

### Installation

```bash
# Clone repository
git clone <repo-url>
cd hotel-booking-cancellation

# Install dependencies
pip install -r requirements.txt
```

### Running the Model

```python
# Load the notebook
jupyter notebook HotelBookingDemand_Capstone_Yonkie.ipynb

# Or train from scratch
python train_model.py --threshold 0.5 --output model.pkl
```

### Making Predictions

```python
import joblib
import pandas as pd

# Load trained model
model = joblib.load('xgboost_model.pkl')

# Prepare new booking data
new_booking = pd.DataFrame({
    'country': ['PRT'],
    'market_segment': ['Online TA'],
    'deposit_type': ['No Deposit'],
    'customer_type': ['Transient'],
    'reserved_room_type': ['A'],
    'previous_cancellations': [0],
    'booking_changes': [1],
    'days_in_waiting_list': [2],
    'required_car_parking_spaces': [0],
    'total_of_special_requests': [2]
})

# Get cancellation probability
probability = model.predict_proba(new_booking)[0][1]
print(f"Cancellation risk: {probability:.1%}")

# Flag if above threshold (0.5)
if probability > 0.5:
    print("⚠️ HIGH RISK — Recommend intervention")
else:
    print("✅ LOW RISK — Monitor normally")
```

### Deployment in Operations

```
1. New booking arrives → Recorded in reservation system
2. Model scores → Calculates cancellation probability (0–1)
3. Daily list → Bookings ranked highest-to-lowest risk
4. Team contacts → Customer Service calls top-priority guests
5. Outcome recorded → Response fed back for model retraining
```

**Best Timing:** Score bookings daily and intervene *well before* check-in date. The closer to arrival, the smaller the window to reallocate the room.

---

## 📁 Project Structure

```
hotel-booking-cancellation/
│
├── README.md                                    # This file
├── requirements.txt                             # Python dependencies
│
├── notebooks/
│   └── HotelBookingDemand_Capstone_Yonkie.ipynb
│       ├── 1. Background & Business Problem
│       ├── 2. Data Cleaning (351 missing values handled)
│       ├── 3. EDA (distributions, correlations, segments)
│       ├── 4. Preprocessing Pipeline
│       ├── 5. Model Benchmarking (7 algorithms)
│       ├── 6. Threshold Analysis & Tuning
│       └── 7. Error Analysis & Insights
│
├── presentations/
│   ├── Prediksi_Pembatalan_Hotel_Capstone3_Yonkie.pptx
│   │   ├── Slide 1–3: Problem framing & stakeholders
│   │   ├── Slide 4–8: Business problem & metric justification
│   │   ├── Slide 9–15: EDA findings
│   │   ├── Slide 16–26: Preprocessing & modeling results
│   │   └── Slide 27–32: Threshold tuning & impact
│   │
│   └── Prediksi_Pembatalan_Hotel_Capstone3_FINAL_v2_fixed.pptx
│       └── (Updated version with improved chart readability)
│
├── data/
│   └── data_hotel_booking_demand.csv            # Original dataset (10,097 rows)
│
├── models/
│   ├── xgboost_model.pkl                        # Trained XGBoost classifier
│   ├── preprocessing_pipeline.pkl               # Scaler + encoder
│   └── threshold_analysis.json                  # Threshold tuning results
│
└── outputs/
    ├── confusion_matrix.png
    ├── feature_importance.png
    ├── threshold_tradeoff.png
    └── prediction_scores_distribution.png
```

---

## 📦 Files & Deliverables

### Jupyter Notebook
- **`HotelBookingDemand_Capstone_Yonkie.ipynb`**
  - Full end-to-end pipeline: EDA → preprocessing → modeling → evaluation
  - 7 algorithms benchmarked with detailed comparison
  - Threshold tuning analysis with business impact simulation
  - All code is reproducible; uses pandas, scikit-learn, XGBoost

### PowerPoint Presentations
- **`Prediksi_Pembatalan_Hotel_Capstone3_Yonkie.pptx`**
  - Full 45-minute presentation for stakeholders
  - Sections: Problem → Data → Modeling → Impact
  - Includes confusion matrix, feature importance, threshold tradeoff chart

- **`Prediksi_Pembatalan_Hotel_Capstone3_FINAL_v2_fixed.pptx`**
  - Polished version with improved visual clarity
  - Fixed chart overlapping issues (Recall/Precision labels)
  - Publication-ready for executive presentations

### This README
- Quick reference guide and project overview

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.8+ |
| **Data** | pandas, NumPy |
| **Preprocessing** | scikit-learn (StandardScaler, OneHotEncoder) |
| **Modeling** | scikit-learn, XGBoost, LightGBM |
| **Evaluation** | scikit-learn metrics (classification_report, confusion_matrix, roc_auc_score) |
| **Visualization** | matplotlib, seaborn |
| **Notebook** | Jupyter Notebook |
| **Presentation** | Microsoft PowerPoint |

---

## ⚠️ Limitations & Future Work

### Current Limitations

1. **No Temporal Dynamics**
   - Dataset lacks time-series features (days-to-arrival calculated as static, not updated)
   - Model doesn't account for price changes, market shifts, or seasonal booking behavior
   - Suitability: Best for daily batch scoring, not real-time dynamic scoring

2. **No Guest Identity Tracking**
   - Data is bookings-level, not guest-level
   - Can't link repeat guests across years to build long-term reputation scores
   - Impact: Limits ability to spot patterns in frequent cancelers

3. **No Room Inventory or Availability Data**
   - Can't assess whether model scores correlate with room type, floor, or special amenities
   - Assumes room type doesn't interact with guest's cancellation propensity

4. **Class Imbalance**
   - 76% vs. 24% split handled via class_weight, but more sophisticated techniques (SMOTE, cost-sensitive learning) could improve boundary decision-making

5. **Model Ceiling**
   - Best AUC-ROC achieved: 0.846
   - Realistically, 86% recall is near-maximum given feature constraints
   - Data quality (aggregated features, no individual IDs) intentionally keeps model from over-optimizing on noise

### Recommended Next Steps

1. **Incorporate Temporal Features**
   - Ingest booking lead time *at prediction time* (not just at booking)
   - Model seasonality: holiday periods, local events, competitor activity
   - Predict recency: update scores as check-in date approaches

2. **Guest Identity & Loyalty**
   - Hash/pseudonymize guest identities to track repeat-visitor behavior over years
   - Build guest segments (high-value repeats, risk profiles)
   - Integrate with loyalty program data

3. **Advanced Imbalance Handling**
   - Experiment with SMOTE oversampling or cost-sensitive boosting
   - Calibrate model probabilities using isotonic regression

4. **Real-Time Scoring Pipeline**
   - Deploy model as API endpoint for instant scoring on booking arrival
   - Implement concept drift monitoring (model retraining schedule)
   - A/B test intervention strategies: which outreach tactics actually reduce cancellations?

5. **Outcome Feedback Loop**
   - Track actual intervention results: did phone call reduce cancellation rate?
   - Collect true labels on flagged bookings for model retraining
   - Measure ROI: cost of contacting 915 bookings vs. revenue recovered

---

## 📞 Contact & Attribution

- **Author:** Yonkie Yudha Ardika
- **Institution:** Purwadhika Digital Technology School
- **Project Type:** Capstone Project Module 3 (Supervised Learning)
- **Submission Date:** [Insert Date]

For questions or feedback, please open an issue on this repository or contact directly.

---

## 📄 License

[Specify license type — e.g., MIT, Creative Commons, or internal/educational use only]

---

## 🎓 Learning Outcomes

This project demonstrates:

✅ **End-to-end ML pipeline:** problem framing → EDA → preprocessing → modeling → evaluation  
✅ **Business acumen:** translating false negative/positive costs into metric selection  
✅ **Model benchmarking:** systematic comparison of 7 algorithms with reproducible evaluation  
✅ **Threshold tuning:** optimizing for recall while monitoring precision against operational constraints  
✅ **Stakeholder communication:** explaining model logic, limitations, and deployment plan to non-technical audiences  
✅ **Practical deployment thinking:** defining when, how, and by whom the model is used in real operations

---

**Happy predicting! 🚀**

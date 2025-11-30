# 🛒 CartSignal: Retail Propensity Engine

**A Machine Learning system that detects life-stage changes (becoming a "New Parent") based on grocery purchasing habits.**

![Python](https://img.shields.io/badge/Python-3.9-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-green)

## 📌 Project Overview
Retailers often struggle to identify high-value customer segments early. This project replicates the logic of the famous "Target Pregnancy Prediction" case study. By analyzing "Basket Drift" (changes in products, timing, and frequency), the model flags users likely to be new parents *before* they explicitly sign up for baby clubs.

## 🧠 Technical Architecture
* **Data Ingestion:** Merged Instacart Market Basket data (3M+ orders).
* **Feature Engineering:** Created behavioral signals:
    * `Healthy_Junk_Ratio`: Proxy for lifestyle changes.
    * `Inter_Order_Latency`: Measuring urgency (parents shop more frequently).
    * `Time_of_Day_Variance`: Detecting sleep schedule disruption.
* **Modeling:** Trained an **XGBoost Classifier** with `scale_pos_weight` to handle the 16% class imbalance.
* **Deployment:** Interactive **Streamlit Dashboard** for marketing stakeholders.

## 🚀 How to Run
1. Clone the repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt

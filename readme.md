# DiaPilot: AI-Based Personalized Diet Recommendation System for Diabetic Patients

## Project Overview
DiaPilot is an automated nutrition management system designed to assist Type 2 Diabetic patients in managing their dietary intake through machine learning. The system utilizes a dataset of over 180,000 global recipes, filtering them through a multi-stage clinical and cultural engine to provide medically safe and culturally relevant meal recommendations.

This project implements two distinct machine learning approaches to evaluate performance:
1. **KNN Baseline:** A spatial similarity model focused on ingredient-based clustering.
2. **XGBoost Ranker:** An advanced Learning-to-Rank (LTR) model utilizing Normalized Discounted Cumulative Gain (NDCG) to prioritize clinical safety and nutritional utility.

## ⚠️ Important Note Regarding Dataset (File Size Limitations)
Due to the university portal's strict file upload size limitations, the massive 180k+ global recipes dataset could not be included in this submission zip folder. 

**To successfully run this code:**
1. Please download the original `RAW_recipes.csv` dataset from its source (e.g., Food.com dataset on Kaggle).
2. Place the downloaded `RAW_recipes.csv` file directly inside the `data/raw/` directory.
3. Run the data pipeline script to generate the necessary Parquet files for the notebooks.

## Key Features
- **Clinical Macro Engine:** Dynamic calculation of TDEE (Total Daily Energy Expenditure) and glycemic thresholds based on patient risk profiles (Leicester Score).
- **Multi-Stage Filtering:** Automated exclusion of high-glycemic ingredients, religious constraints (e.g., Pork), and cultural sub-setting for South Asian diets.
- **Automated Reporting:** Generation of 30-day longitudinal diet plans and clinical experiment reports in PDF format.
- **Optimized Data Pipeline:** Transition from raw CSV processing to high-performance Parquet storage for low-latency execution.

## Project Structure
```text
DiaPilot_Submission/
├── notebooks/
│   ├── Exp1_KNN_Baseline.ipynb       # Baseline Similarity Model
│   └── Exp2_XGBoost_Ranker.ipynb     # Advanced Learning-to-Rank Model
├── src/
│   └── data_pipeline.py              # Data Cleaning and Pre-processing Script
├── data/
│   ├── raw/                          # (Place RAW_recipes.csv here)
│   └── processed/                    # (Generated automatically)
├── reports/                          # Generated PDF Diet Plans and Reports
├── requirements.txt                  # Python Dependencies
└── README.md                         # Project Documentation
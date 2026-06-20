import pandas as pd
import ast
import os

print("="*60)
print(" 🚀 Initializing DiaPilot Data Cleaning Pipeline (Food.com only)")
print("="*60)

# ---------------------------------------------------------
# FILE PATHS
# ---------------------------------------------------------
FOOD_COM_PATH = '../data/raw/RAW_recipes.csv'
PROCESSED_DATA_PATH = '../data/processed/DiaPilot_Combined_Data.parquet'

os.makedirs('../data/processed', exist_ok=True)

# SAFE PARSER FUNCTION (prevents IndexError crashes on malformed rows)
def parse_nutrition(nutrition_str):
    try:
        vals = ast.literal_eval(nutrition_str)
        if not isinstance(vals, list) or len(vals) < 7:
            return [None] * 7   # Mark as invalid, NaN drop handles it later
        return vals
    except:
        return [None] * 7

try:
    # =========================================================
    # PART 1: Process Food.com Dataset
    # =========================================================
    print("\n[1/2] Processing Food.com dataset...")
    raw_df = pd.read_csv(FOOD_COM_PATH)
    print(f"      -> Original Recipes Loaded: {len(raw_df)}")

    raw_df.dropna(subset=['nutrition', 'ingredients', 'name'], inplace=True)
    raw_df.drop_duplicates(subset=['name'], inplace=True)

    print("      -> Safely parsing nutrition strings...")
    raw_df['nutrition_list'] = raw_df['nutrition'].apply(parse_nutrition)

    # Filter out malformed rows immediately before any extraction
    before_parse = len(raw_df)
    raw_df = raw_df[raw_df['nutrition_list'].apply(lambda x: x[0] is not None)]
    print(f"      -> Dropped {before_parse - len(raw_df)} malformed nutrition rows")

    # Convert Percentage Daily Value (PDV) to Exact Grams
    print("      -> Converting PDV to absolute Grams (FDA Standard)...")
    raw_df['Calories']  = raw_df['nutrition_list'].apply(lambda x: x[0])
    raw_df['Fat_g']     = raw_df['nutrition_list'].apply(lambda x: round(x[1] * 78 / 100, 1))
    raw_df['Sugar_g']   = raw_df['nutrition_list'].apply(lambda x: round(x[2] * 50 / 100, 1))
    raw_df['Protein_g'] = raw_df['nutrition_list'].apply(lambda x: round(x[4] * 50 / 100, 1))
    raw_df['Carbs_g']   = raw_df['nutrition_list'].apply(lambda x: round(x[6] * 275 / 100, 1))

    food_com_df = raw_df[['name', 'ingredients', 'Calories', 'Carbs_g', 'Sugar_g', 'Protein_g', 'Fat_g']].copy()
    food_com_df.columns = ['Recipe_Name', 'Ingredients', 'Calories', 'Carbs_g', 'Sugar_g', 'Protein_g', 'Fat_g']
    food_com_df['Source'] = 'Food.com'

    # Apply Strict Medical & Heuristic Filters (ADA-aligned)
    print("      -> Applying ADA Baseline Filters...")
    food_com_df = food_com_df[
        (food_com_df['Calories'] > 50) & (food_com_df['Calories'] < 800) &
        (food_com_df['Protein_g'] > 2) &
        (food_com_df['Carbs_g'] < 150) &
        (food_com_df['Sugar_g'] < 25)
    ]
    print(f"      -> Cleaned Food.com Recipes Surviving: {len(food_com_df)}")

    combined_df = food_com_df

    # =========================================================
    # PART 2: Final Cleanup (NaN Poisoning Fix)
    # =========================================================
    print("\n[2/2] Finalizing and cleaning data...")

    nutrition_cols = ['Calories', 'Carbs_g', 'Sugar_g', 'Protein_g', 'Fat_g']
    before_len = len(combined_df)
    combined_df = combined_df.dropna(subset=nutrition_cols)
    dropped_nans = before_len - len(combined_df)

    if dropped_nans > 0:
        print(f"      -> ALERT: Dropped {dropped_nans} rows missing nutrition data (Silent NaN poisoning prevented).")

    combined_df = combined_df.drop_duplicates(subset=['Recipe_Name'])
    combined_df = combined_df.reset_index(drop=True)

    combined_df.to_parquet(PROCESSED_DATA_PATH, index=False)

    print("="*60)
    print(f" ✅ SUCCESS: Production Data Pipeline completed!")
    print(f" 💾 Saved to: {PROCESSED_DATA_PATH}")
    print(f" 📊 Total Clean Records: {len(combined_df)}")
    print("="*60)

except Exception as e:
    print(f"\n❌ ERROR IN PIPELINE: {e}")
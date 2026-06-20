# ==============================================================================
# DiaPilot - Final Combined Plan Generator (CSV Output) v2
# Location: src/generate_combined_csv.py
#
# Output columns:
#   Day | Meal | Recipe | Ingredients | Calories | Carbs_g | Sugar_g | Protein_g
#   Post_Meal_Exercise | Daily_Evening_Workout
# ==============================================================================

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')


# ── Patient profile ───────────────────────────────────────────────────────────
PATIENT = {
    'Age'                  : 45,
    'BMI'                  : 29.5,
    'Leicester_Score'      : 14,
    'Mode'                 : 1,
    'On_Medication'        : 1,
    'Current_Glucose_mmol' : 8.5,
    'Weight_kg'            : 85,
    'Activity_Level'       : 2
}

HIGH_GLUCOSE_THRESHOLD = 10.0  # mmol/L


# ── Post-meal walk per meal ───────────────────────────────────────────────────
def get_post_meal_walk(meal_calories, meal_type, patient):
    """
    Calculates walk duration after THIS specific meal.
    Total daily walk = sum of all 3 post-meal walks.
    Based on 30% of meal calories to burn via walking.

    Example: 600 kcal meal → burn 180 kcal walking
    MET 3.5 × 85kg = 297.5 kcal/hour → 180/297.5 × 60 = 36 mins
    Capped at 20 mins per meal (clinical recommendation)
    """
    age     = patient['Age']
    glucose = patient['Current_Glucose_mmol']
    bmi     = patient['BMI']
    weight  = patient['Weight_kg']

    # MET-based calculation
    cals_to_burn = meal_calories * 0.30
    met          = 3.5  # walking
    raw_mins     = (cals_to_burn / (met * weight)) * 60

    # Cap per meal: max 20 mins after any single meal
    walk_mins = max(10, min(round(raw_mins), 20))

    # Age adjustments
    if age >= 66:
        walk_mins = min(walk_mins, 10)
        pace = 'Slow Walk'
    elif age >= 56:
        walk_mins = min(walk_mins, 15)
        pace = 'Brisk Walk'
    elif glucose > HIGH_GLUCOSE_THRESHOLD:
        pace = 'Gentle Walk'
    elif bmi > 35:
        pace = 'Slow Walk'
    else:
        pace = 'Brisk Walk'

    return f'{pace} {walk_mins} mins after {meal_type}'


# ── Evening workout pools (age-aware) ────────────────────────────────────────
# Patient Age 45, Management mode, On Medication → middle_moderate pool
EVENING_POOLS = {

    # Age 66+ — very light only
    'elderly': [
        'Chair Leg Raises x10 + Seated Arm Circles x15',
        'Gentle Yoga Stretching 15 mins',
        'Wall Push-ups x8 + Ankle Rotations x15',
        'Deep Breathing Exercises 10 mins',
        'Seated Marching 10 mins',
        'Chair Yoga 15 mins',
        'Light Arm Stretches 10 mins',
    ],

    # Age 56-65 — light to moderate
    'senior': [
        'Resistance Band Pull-Apart x12 + Calf Raises x15',
        'Dumbbell Bicep Curl x10 + Shoulder Press x10',
        'Wall Sit 30 sec + Bodyweight Squat x10',
        'Seated Band Row x12 + Chest Press x10',
        'Dumbbell Lunge x8 each leg (slow)',
        'Stationary Cycling 15 mins',
        'Dumbbell Deadlift x8 (light weight)',
    ],

    # Age 36-55, Management mode, On Medication → moderate
    'middle_moderate': [
        'Push-ups x15 + Dumbbell Curl x12',
        'Bodyweight Squat x20 + Plank 30 sec',
        'Dumbbell Shoulder Press x12 + Lateral Raise x10',
        'Band Row x15 + Chest Fly x12',
        'Walking Lunges x12 each + Calf Raises x20',
        'Dumbbell Deadlift x12',
        'Kettlebell Swing x15',
    ],

    # Age 36-55, Prevention mode or no meds → moderate-vigorous
    'middle_vigorous': [
        'Push-ups x20 + Burpees x10 + Plank 45 sec',
        'Dumbbell Press x15 + Squat Jump x10',
        'Jumping Jacks x30 + Dumbbell Curl x15 + Lunge x12',
        'Circuit: Squat x20, Push-up x15, Plank 45 sec x3 rounds',
        'Dumbbell: Deadlift x12 + Row x12 + Press x12',
        'Chest + Back Resistance Training 30 mins',
        'HIIT: Burpee x8, Mountain Climber x20, Jump Squat x10',
    ],

    # Age 25-35 — full vigorous
    'young': [
        'Push-ups x25 + Pull-ups x10 + Plank 60 sec',
        'Dumbbell Full Body Circuit x3 rounds (30 mins)',
        'Burpees x15 + Jump Squats x15 + Mountain Climbers x20',
        'Resistance Training: Legs + Shoulders (35 mins)',
        'HIIT Circuit: 40 sec on / 20 sec rest x6 exercises',
        'Deadlift x15 + Bent Row x15 + Curl x12',
        'Full Gym Session: Chest + Triceps (40 mins)',
    ],

    # High glucose override
    'high_glucose': [
        'Gentle Stretching 10 mins only (glucose high)',
        'Light Yoga 10 mins (no strain)',
        'Seated Leg Extensions x10',
        'Deep Breathing 10 mins',
        'Chair Yoga 10 mins',
        'Gentle Arm Stretches 10 mins',
        'Slow Walk 10 mins (very light)',
    ],

    # High BMI
    'high_bmi': [
        'Band Seated Row x12 + Chair Push-ups x10',
        'Seated Dumbbell Curl x10 + Shoulder Press x8',
        'Band Chest Press x12 + Leg Extensions x12',
        'Seated Resistance Band Exercise 15 mins',
        'Chair-Based Strength Circuit 15 mins',
        'Light Dumbbell Shoulder Press x10 seated',
        'Band Bicep Curl x12 + Tricep Extension x10 seated',
    ],
}


def get_evening_workout(day_num, patient):
    """
    Selects evening workout based on patient profile.
    Cycles through 7 different workouts (one per day of week).
    """
    age     = patient['Age']
    glucose = patient['Current_Glucose_mmol']
    bmi     = patient['BMI']
    mode    = patient['Mode']
    on_meds = patient['On_Medication']

    # Select pool
    if glucose > HIGH_GLUCOSE_THRESHOLD:
        pool = EVENING_POOLS['high_glucose']
    elif bmi > 35:
        pool = EVENING_POOLS['high_bmi']
    elif age >= 66:
        pool = EVENING_POOLS['elderly']
    elif age >= 56:
        pool = EVENING_POOLS['senior']
    elif mode == 1 or on_meds == 1:
        pool = EVENING_POOLS['middle_moderate']
    elif age >= 36:
        pool = EVENING_POOLS['middle_vigorous']
    else:
        pool = EVENING_POOLS['young']

    # Cycle through 7 workouts — each day of the week gets a different one
    idx = (day_num - 1) % 7
    return pool[idx % len(pool)]


# ── Meal swap (internal logic — only alert shown to user) ─────────────────────
def should_swap_meal(carbs_g, sugar_g, glucose):
    """
    Internal check only — not shown in CSV.
    Returns True if meal should be swapped due to high glucose.
    The app shows an alert: 'Glucose high, finding safer meal for you'
    The swap happens quietly — user only sees the new meal.
    """
    if glucose <= HIGH_GLUCOSE_THRESHOLD:
        return False
    # If meal already has low carbs it is safe — no swap needed
    if carbs_g <= 35 and sugar_g <= 3:
        return False
    return True


# ── Main ──────────────────────────────────────────────────────────────────────
def generate_combined_csv():
    print('=' * 58)
    print('  DiaPilot Final Combined Plan Generator v2')
    print('=' * 58)

    # Load diet plan
    paths = [
        '../reports/Diapilot_DNN_Diet_Plan.csv',
        '../reports/Diapilot_RF_Diet_Plan.csv',
        '../reports/Diapilot_v3_30Day_Plan.csv',
    ]
    diet_df = None
    for path in paths:
        if os.path.exists(path):
            diet_df = pd.read_csv(path)
            print(f'Loaded: {path}')
            print(f'  Rows: {len(diet_df)} | Columns: {list(diet_df.columns)}')
            break

    if diet_df is None:
        print('ERROR: No diet plan CSV found.')
        return

    # Detect columns
    recipe_col = next((c for c in ['Recipe','Recipe Name'] if c in diet_df.columns), 'Recipe')
    cal_col    = next((c for c in ['Calories','Cal']        if c in diet_df.columns), 'Calories')
    carb_col   = next((c for c in ['Carbs_g','Carbs']       if c in diet_df.columns), 'Carbs_g')
    sugar_col  = next((c for c in ['Sugar_g','Sugar']       if c in diet_df.columns), 'Sugar_g')
    pro_col    = next((c for c in ['Protein_g','Pro']        if c in diet_df.columns), 'Protein_g')

    days = diet_df['Day'].unique()
    print(f'Days: {len(days)} | Patient Age: {PATIENT["Age"]}')
    print(f'Exercise pool: middle_moderate (Age 45, Management, On Medication)')
    print()

    # ── Per-meal: Post_Meal_Exercise ─────────────────────
    post_meal_ex   = []
    evening_workout = []

    for day_num, day_label in enumerate(days, 1):
        day_rows = diet_df[diet_df['Day'] == day_label]
        workout  = get_evening_workout(day_num, PATIENT)

        for _, row in day_rows.iterrows():
            cal       = float(str(row.get(cal_col, 500)).replace('g','').strip() or 500)
            meal_type = str(row.get('Meal', 'Meal'))

            # Post meal walk for this specific meal
            ex = get_post_meal_walk(cal, meal_type, PATIENT)
            post_meal_ex.append(ex)

            # Same evening workout for all 3 meals of the day
            evening_workout.append(workout)

    diet_df['Post_Meal_Exercise']    = post_meal_ex
    diet_df['Daily_Evening_Workout'] = evening_workout

    # ── Keep only clean columns ───────────────────────────
    # Swap_Meal and Swap_Reason removed — internal logic only
    # Daily_Morning_Exercise removed — user already walks after each meal
    # Daily_Walk_Mins removed — confusing
    # Daily_Total_Cal removed — not needed for user

    keep_cols = [
        'Day', 'Meal', recipe_col,
        'Ingredients' if 'Ingredients' in diet_df.columns else None,
        cal_col, carb_col, sugar_col, pro_col,
        'Post_Meal_Exercise',
        'Daily_Evening_Workout'
    ]
    keep_cols = [c for c in keep_cols if c and c in diet_df.columns]
    final_df  = diet_df[keep_cols].copy()

    # Clean column names
    final_df = final_df.rename(columns={
        recipe_col: 'Recipe',
        cal_col   : 'Calories',
        carb_col  : 'Carbs_g',
        sugar_col : 'Sugar_g',
        pro_col   : 'Protein_g',
    })

    # ── Save ─────────────────────────────────────────────
    # Overwrites the SAME file that was loaded (Diapilot_DNN_Diet_Plan.csv)
    # so there is only one diet+exercise file, not a duplicate.
    out_path = path
    final_df.to_csv(out_path, index=False)

    print('=' * 58)
    print(f'CSV saved: {out_path}')
    print(f'Total rows: {len(final_df)}')
    print()
    print('Final columns:')
    for col in final_df.columns:
        print(f'  {col}')
    print()
    print('Sample — Day 1:')
    day1 = final_df[final_df['Day'] == 'Day 1']
    print(day1[['Meal','Recipe','Calories',
                'Post_Meal_Exercise',
                'Daily_Evening_Workout']].to_string(index=False))
    print()

    # Verify post-meal walk totals make sense
    print('Post-meal walk verification (Day 1):')
    for _, row in day1.iterrows():
        print(f'  {row["Meal"]:12s}: {row["Post_Meal_Exercise"]}')
    print()
    print('Evening workout cycles through 7 options across 30 days.')
    print('=' * 58)


if __name__ == '__main__':
    generate_combined_csv()
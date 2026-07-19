"""
test_gemini_direct.py
---------------------
Standalone test for the Gemini Pakistani food-conversion mechanism
used in DiaPilot's app.py.

Usage:
    1. Set your Gemini API key:
           $env:GEMINI_API_KEY = "your_key_here"   (PowerShell)
       OR
           set GEMINI_API_KEY=your_key_here         (CMD)

    2. Run from the workspace root:
           python test_gemini_direct.py

What it does:
  - Mirrors EXACTLY the same logic as `convert_plan_to_pakistani()` in app.py.
  - Uses a small 9-meal sample plan (3 days × 3 meals) so the request
    is tiny and we can inspect the output easily.
  - Tries every model in the same fallback chain as production.
  - Prints a clear PASS / FAIL result with before/after names.
"""

import os
import requests
import json

# ─── Config (copied verbatim from app.py) ───────────────────────────────────
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_ENABLED = bool(GEMINI_API_KEY)

GEMINI_MODEL_FALLBACK_CHAIN = [
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-latest",
]

GEMINI_REST_URL_TEMPLATE = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# ─── Small sample plan (3 days × 3 meals = 9 meals total) ───────────────────
SAMPLE_PLAN = [
    {
        "day": 1,
        "meals": [
            {"meal_type": "Breakfast", "recipe_name": "Greek Yogurt Parfait"},
            {"meal_type": "Lunch",     "recipe_name": "Grilled Chicken Caesar Salad"},
            {"meal_type": "Dinner",    "recipe_name": "Baked Salmon with Asparagus"},
        ],
    },
    {
        "day": 2,
        "meals": [
            {"meal_type": "Breakfast", "recipe_name": "Oatmeal with Blueberries"},
            {"meal_type": "Lunch",     "recipe_name": "Turkey and Avocado Wrap"},
            {"meal_type": "Dinner",    "recipe_name": "Beef Stir-Fry with Broccoli"},
        ],
    },
    {
        "day": 3,
        "meals": [
            {"meal_type": "Breakfast", "recipe_name": "Scrambled Eggs with Spinach"},
            {"meal_type": "Lunch",     "recipe_name": "Lentil Soup"},
            {"meal_type": "Dinner",    "recipe_name": "Grilled Tilapia with Quinoa"},
        ],
    },
]


def convert_plan_to_pakistani(plan):
    """Exact copy of the production function from app.py."""
    if not GEMINI_ENABLED:
        print("SKIP: GEMINI_API_KEY is not set. Cannot test conversion.")
        return plan

    flat_meals = []
    for day_entry in plan:
        for meal in day_entry["meals"]:
            flat_meals.append(meal["recipe_name"])

    numbered_list = "\n".join(
        f"{i+1}. {name}" for i, name in enumerate(flat_meals)
    )

    prompt = (
        "You are helping localize a diabetic meal plan for a Pakistani "
        "user. Below is a numbered list of dish names from a Western "
        "recipe dataset. For each one, reply with the closest common "
        "Pakistani dish name that a Pakistani household would "
        "recognize and could realistically cook as a substitute — do "
        "NOT change the nutritional meaning, just give the localized "
        "name. Reply with ONLY a numbered list in the exact same order, "
        "one Pakistani dish name per line, no extra commentary.\n\n"
        f"{numbered_list}"
    )

    print("\n" + "="*60)
    print("PROMPT SENT TO GEMINI:")
    print("="*60)
    print(prompt)
    print("="*60 + "\n")

    for model_name in GEMINI_MODEL_FALLBACK_CHAIN:
        url = GEMINI_REST_URL_TEMPLATE.format(model=model_name)
        print(f"  -> Trying model: {model_name} ...")

        try:
            response = requests.post(
                url,
                params={"key": GEMINI_API_KEY},
                json={
                    "contents": [
                        {"parts": [{"text": prompt}]}
                    ]
                },
                timeout=30,
            )

            if response.status_code != 200:
                print(f"    FAIL HTTP {response.status_code} --- {response.text[:200]}")
                print(f"      Trying next model in fallback chain.")
                continue

            result = response.json()
            raw_text = result["candidates"][0]["content"]["parts"][0]["text"]

            print(f"\n  RAW GEMINI RESPONSE (model={model_name}):")
            print("  " + raw_text.replace("\n", "\n  "))

            lines = [
                line.strip() for line in raw_text.strip().split("\n")
                if line.strip()
            ]

            converted_names = []
            for line in lines:
                if ". " in line:
                    converted_names.append(line.split(". ", 1)[1].strip())
                else:
                    converted_names.append(line.strip())

            if len(converted_names) != len(flat_meals):
                print(f"\n  FAIL Count mismatch: got {len(converted_names)}, "
                      f"expected {len(flat_meals)}. Trying next model.\n")
                continue

            # Apply
            idx = 0
            for day_entry in plan:
                for meal in day_entry["meals"]:
                    meal["recipe_name"] = converted_names[idx]
                    idx += 1

            print(f"\n  SUCCESS using model '{model_name}'")
            return plan

        except Exception as e:
            print(f"    FAIL Exception: {e}  Trying next model.")
            continue

    print("\n  FAIL: All models in the fallback chain failed.")
    return plan


# ─── Main ───────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  DiaPilot — Gemini Pakistani Food Conversion Test")
    print("="*60)

    if not GEMINI_ENABLED:
        print("\n[ERROR] GEMINI_API_KEY environment variable is NOT set.")
        print("        Set it first, then re-run:\n")
        print("          PowerShell : $env:GEMINI_API_KEY = 'your_key_here'")
        print("          CMD        : set GEMINI_API_KEY=your_key_here\n")
        exit(1)

    print(f"\n[OK] GEMINI_API_KEY is set (length={len(GEMINI_API_KEY)}).")
    print(f"[OK] Fallback chain: {GEMINI_MODEL_FALLBACK_CHAIN}")
    print(f"\nSample plan — BEFORE conversion:")
    for day in SAMPLE_PLAN:
        print(f"  Day {day['day']}:")
        for meal in day["meals"]:
            print(f"    [{meal['meal_type']}] {meal['recipe_name']}")

    import copy
    plan_copy = copy.deepcopy(SAMPLE_PLAN)
    result = convert_plan_to_pakistani(plan_copy)

    print("\n" + "="*60)
    print("BEFORE  ->  AFTER")
    print("="*60)
    original_names = [m["recipe_name"] for d in SAMPLE_PLAN for m in d["meals"]]
    converted_names = [m["recipe_name"] for d in result for m in d["meals"]]

    all_changed = False
    for orig, conv in zip(original_names, converted_names):
        changed = "[OK]" if orig.lower() != conv.lower() else "[??]"
        if orig.lower() != conv.lower():
            all_changed = True
        print(f"  {changed}  {orig:40s} ->  {conv}")

    print("\n" + "="*60)
    if all_changed:
        print("  RESULT: PASS -- Gemini converted all food names successfully!")
    else:
        changed_count = sum(
            1 for o, c in zip(original_names, converted_names) if o.lower() != c.lower()
        )
        if changed_count > 0:
            print(f"  RESULT: PARTIAL -- {changed_count}/{len(original_names)} names converted.")
        else:
            print("  RESULT: FAIL -- No names were converted (API may not be working).")
    print("="*60 + "\n")

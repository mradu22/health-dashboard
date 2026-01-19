"""
Configuration for health dashboard goals and targets.
"""
from __future__ import annotations
from datetime import date

# === BODY GOALS ===
WEIGHT_GOAL_LBS = 180
WEIGHT_GOAL_KG = WEIGHT_GOAL_LBS / 2.205  # ~81.6 kg
BODY_FAT_GOAL_PCT = 16.0

# === WORKOUT GOALS ===
GYM_SESSIONS_PER_WEEK = 3
CARDIO_MIN_PER_SESSION = 5  # minimum minutes to count as cardio

# === CARDIO DISTANCE GOALS ===
CARDIO_KM_PER_MONTH = 100
CARDIO_KM_PER_YEAR = 12000

# === PROTEIN GOAL ===
# Protein target = body weight in lbs (in grams) ± 15g
PROTEIN_TOLERANCE_G = 15

def get_protein_target(weight_kg: float) -> tuple[float, float, float]:
    """
    Returns (min, target, max) protein in grams based on weight.
    Target = weight in lbs (in grams)
    """
    weight_lbs = weight_kg * 2.205
    target = weight_lbs
    return (target - PROTEIN_TOLERANCE_G, target, target + PROTEIN_TOLERANCE_G)

# === SLEEP GOALS ===
SLEEP_HOURS_MIN = 7.0

# === YEAR TRACKING ===
CURRENT_YEAR = date.today().year
YEAR_START = date(CURRENT_YEAR, 1, 1)

# === DATA SOURCE ===
# For local development
CSV_PATH = "/home/adi/arch/life/data/apple-data/apple_data.csv"

# For Streamlit Cloud (Google Sheets)
# GOOGLE_SHEET_URL = "https://docs.google.com/spreadsheets/d/YOUR_SHEET_ID"

# === UI CONFIG ===
THEME = {
    "primary_color": "#4CAF50",
    "background_color": "#0E1117",
    "text_color": "#FAFAFA",
    "accent_positive": "#00C853",
    "accent_negative": "#FF5252",
    "accent_neutral": "#9E9E9E",
}

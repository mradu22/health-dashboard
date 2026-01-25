"""
Configuration for health dashboard goals and targets.
"""
from __future__ import annotations
from datetime import date

# === PROJECT VERSION ===
VERSION = "0.4.0"

# === UNIT CONVERSIONS ===
KG_TO_LBS = 2.205

# === BODY GOALS ===
WEIGHT_GOAL_LBS = 180
BODY_FAT_GOAL_PCT = 16.0

# === WORKOUT GOALS ===
GYM_SESSIONS_PER_WEEK = 3
CARDIO_SESSIONS_PER_WEEK = 5
CARDIO_MIN_PER_SESSION = 15  # minimum minutes to count as cardio session
STRETCH_SESSIONS_PER_WEEK = 5

# === CARDIO DISTANCE GOALS ===
CARDIO_KM_PER_WEEK = 25

# === PROTEIN GOAL ===
# Protein target = body weight in lbs (in grams) ± 15g
PROTEIN_TOLERANCE_G = 15

def get_protein_target(weight_lbs: float) -> tuple[float, float, float]:
    """
    Returns (min, target, max) protein in grams based on weight.
    Target = weight in lbs (in grams)
    """
    target = weight_lbs
    return (target - PROTEIN_TOLERANCE_G, target, target + PROTEIN_TOLERANCE_G)

# === SLEEP GOALS ===
SLEEP_HOURS_MIN = 7.0
SLEEP_DAYS_PER_WEEK = 7  # Goal: sleep 7+ hours every day

# === YEAR TRACKING ===
CURRENT_YEAR = date.today().year
YEAR_START = date(CURRENT_YEAR, 1, 1)

# === DATA SOURCE ===
# For local development
CSV_PATH_LOCAL = "/home/adi/arch/life/data/apple-data/apple_data.csv"

# For Streamlit Cloud (Google Drive direct download)
# File ID extracted from: https://drive.google.com/file/d/1GlakDmUsEyRDsra6q7-fPP2Tf-q3IeG_/view
GDRIVE_FILE_ID = "1GlakDmUsEyRDsra6q7-fPP2Tf-q3IeG_"
GDRIVE_URL = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"

# === UI CONFIG ===
THEME = {
    "primary_color": "#4CAF50",
    "background_color": "#0E1117",
    "text_color": "#FAFAFA",
    "accent_positive": "#00C853",
    "accent_negative": "#FF5252",
    "accent_neutral": "#9E9E9E",
}

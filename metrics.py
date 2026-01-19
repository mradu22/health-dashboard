"""
Metrics calculations for health dashboard goals and trends.
"""
import pandas as pd
from datetime import date, timedelta

from config import (
    GYM_SESSIONS_PER_WEEK,
    CARDIO_MIN_PER_SESSION,
    CARDIO_KM_PER_MONTH,
    CARDIO_KM_PER_YEAR,
    WEIGHT_GOAL_KG,
    BODY_FAT_GOAL_PCT,
    SLEEP_HOURS_MIN,
    get_protein_target,
)


# === WORKOUT METRICS ===

def count_gym_sessions(df: pd.DataFrame) -> int:
    """Count days with strength training > 0."""
    if 'strength_training_min' not in df.columns:
        return 0
    return (df['strength_training_min'] > 0).sum()


def count_cardio_sessions(df: pd.DataFrame, min_minutes: float = CARDIO_MIN_PER_SESSION) -> int:
    """Count days with cardio >= min_minutes."""
    if 'cardio_min' not in df.columns:
        return 0
    return (df['cardio_min'] >= min_minutes).sum()


def get_total_distance_km(df: pd.DataFrame) -> float:
    """Get total distance in km."""
    if 'distance_km' not in df.columns:
        return 0.0
    return df['distance_km'].sum()


def gym_goal_progress(df_week: pd.DataFrame) -> tuple[int, int, float]:
    """
    Returns (current, goal, percentage) for gym sessions this week.
    """
    current = count_gym_sessions(df_week)
    goal = GYM_SESSIONS_PER_WEEK
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    return (current, goal, pct)


def cardio_month_progress(df_month: pd.DataFrame) -> tuple[float, float, float]:
    """
    Returns (current_km, goal_km, percentage) for cardio this month.
    """
    current = get_total_distance_km(df_month)
    goal = CARDIO_KM_PER_MONTH
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    return (current, goal, pct)


def cardio_year_progress(df_year: pd.DataFrame) -> tuple[float, float, float]:
    """
    Returns (current_km, goal_km, percentage) for cardio this year.
    """
    current = get_total_distance_km(df_year)
    goal = CARDIO_KM_PER_YEAR
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    return (current, goal, pct)


# === BODY METRICS ===

def weight_change(current_kg: float, previous_kg: float | None) -> float | None:
    """Calculate weight change in kg."""
    if previous_kg is None:
        return None
    return current_kg - previous_kg


def weight_to_goal(current_kg: float) -> float:
    """Calculate kg remaining to weight goal."""
    return current_kg - WEIGHT_GOAL_KG


def body_fat_to_goal(current_pct: float) -> float:
    """Calculate body fat % remaining to goal."""
    return current_pct - BODY_FAT_GOAL_PCT


# === PROTEIN METRICS ===

def protein_status(protein_g: float, weight_kg: float) -> tuple[str, float, float]:
    """
    Returns (status, target, difference) for protein intake.
    Status: 'low', 'good', 'high'
    """
    min_p, target, max_p = get_protein_target(weight_kg)
    
    if protein_g < min_p:
        return ('low', target, protein_g - min_p)
    elif protein_g > max_p:
        return ('high', target, protein_g - max_p)
    else:
        return ('good', target, 0)


# === SLEEP METRICS ===

def sleep_quality_score(row: pd.Series) -> float | None:
    """
    Calculate sleep quality score (0-100).
    Weighted: deep sleep (40%) + REM (30%) + low interruptions (30%)
    """
    total = row.get('sleep_hours')
    deep = row.get('deep_sleep_hours')
    rem = row.get('rem_hours')
    interruptions = row.get('sleep_interruptions')
    
    if pd.isna(total) or total == 0:
        return None
    
    score = 0
    
    # Deep sleep: target ~1.5-2 hours (20-25% of sleep)
    if not pd.isna(deep):
        deep_pct = deep / total
        deep_score = min(deep_pct / 0.2, 1.0) * 40  # Max 40 points
        score += deep_score
    
    # REM: target ~1.5-2 hours (20-25% of sleep)
    if not pd.isna(rem):
        rem_pct = rem / total
        rem_score = min(rem_pct / 0.2, 1.0) * 30  # Max 30 points
        score += rem_score
    
    # Interruptions: 0 is perfect, 5+ is bad
    if not pd.isna(interruptions):
        int_score = max(0, (5 - interruptions) / 5) * 30  # Max 30 points
        score += int_score
    
    return score


def days_meeting_sleep_goal(df: pd.DataFrame) -> tuple[int, int, float]:
    """
    Returns (days_met, total_days, percentage) for sleep >= 7 hours.
    """
    if 'sleep_hours' not in df.columns:
        return (0, 0, 0)
    
    valid = df['sleep_hours'].notna()
    total = valid.sum()
    met = (df.loc[valid, 'sleep_hours'] >= SLEEP_HOURS_MIN).sum()
    pct = met / total * 100 if total > 0 else 0
    return (int(met), int(total), pct)


# === TREND CALCULATIONS ===

def rolling_average(df: pd.DataFrame, column: str, window: int = 7) -> pd.Series:
    """Calculate rolling average for a column."""
    if column not in df.columns:
        return pd.Series()
    return df[column].rolling(window=window, min_periods=1).mean()


def compare_periods(current_df: pd.DataFrame, previous_df: pd.DataFrame, column: str) -> float | None:
    """
    Compare average of column between two periods.
    Returns the difference (current - previous).
    """
    if column not in current_df.columns or column not in previous_df.columns:
        return None
    
    current_avg = current_df[column].mean()
    previous_avg = previous_df[column].mean()
    
    if pd.isna(current_avg) or pd.isna(previous_avg):
        return None
    
    return current_avg - previous_avg


# === STREAK CALCULATIONS ===

def meditation_streak(df: pd.DataFrame) -> int:
    """
    Count consecutive days with meditation > 0, ending at most recent day.
    """
    if 'meditation_min' not in df.columns:
        return 0
    
    # Start from most recent and count backwards
    streak = 0
    for idx in reversed(df.index):
        val = df.loc[idx, 'meditation_min']
        if pd.notna(val) and val > 0:
            streak += 1
        else:
            break
    return streak

"""
Metrics calculations for health dashboard goals and trends.
"""
from __future__ import annotations
import pandas as pd
from datetime import date, timedelta
from typing import Optional, Tuple

from config import (
    GYM_SESSIONS_PER_WEEK,
    CARDIO_SESSIONS_PER_WEEK,
    CARDIO_MIN_PER_SESSION,
    CARDIO_KM_PER_WEEK,
    STRETCH_SESSIONS_PER_WEEK,
    WEIGHT_GOAL_LBS,
    BODY_FAT_GOAL_PCT,
    SLEEP_HOURS_MIN,
    SLEEP_DAYS_PER_WEEK,
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


def cardio_sessions_progress(df_week: pd.DataFrame) -> tuple[int, int, float]:
    """
    Returns (current, goal, percentage) for cardio sessions this week.
    Cardio session = day with cardio_min >= 15.
    """
    current = count_cardio_sessions(df_week, CARDIO_MIN_PER_SESSION)
    goal = CARDIO_SESSIONS_PER_WEEK
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    return (current, goal, pct)


def cardio_km_week_progress(df_week: pd.DataFrame) -> tuple[float, float, float]:
    """
    Returns (current_km, goal_km, percentage) for cardio km this week.
    """
    current = get_total_distance_km(df_week)
    goal = CARDIO_KM_PER_WEEK
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    return (current, goal, pct)


def stretch_sessions_progress(df_week: pd.DataFrame) -> tuple[int, int, float]:
    """
    Returns (current, goal, percentage) for stretch sessions this week.
    Stretch session = day with stretch_min > 0.
    """
    if 'stretch_min' not in df_week.columns:
        return (0, STRETCH_SESSIONS_PER_WEEK, 0)
    
    current = (df_week['stretch_min'] > 0).sum()
    goal = STRETCH_SESSIONS_PER_WEEK
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    return (int(current), goal, pct)


def sleep_days_progress(df_week: pd.DataFrame) -> tuple[int, int, float]:
    """
    Returns (days_met, goal, percentage) for days with sleep >= 7 hours this week.
    """
    if 'sleep_hours' not in df_week.columns:
        return (0, SLEEP_DAYS_PER_WEEK, 0)
    
    valid = df_week['sleep_hours'].notna()
    met = (df_week.loc[valid, 'sleep_hours'] >= SLEEP_HOURS_MIN).sum()
    goal = SLEEP_DAYS_PER_WEEK
    pct = min(met / goal * 100, 100) if goal > 0 else 0
    return (int(met), goal, pct)


# === BODY METRICS ===

def get_weight_lbs(weight_col_value: Optional[float]) -> Optional[float]:
    """Get weight in lbs directly from weight_lbs column."""
    if weight_col_value is None or pd.isna(weight_col_value):
        return None
    return weight_col_value


def weight_change_lbs(current_lbs: Optional[float], previous_lbs: Optional[float]) -> Optional[float]:
    """Calculate weight change in lbs."""
    if current_lbs is None or previous_lbs is None:
        return None
    if pd.isna(current_lbs) or pd.isna(previous_lbs):
        return None
    return current_lbs - previous_lbs  # Already in lbs


def weight_to_goal_lbs(current_lbs: Optional[float]) -> Optional[float]:
    """Calculate lbs remaining to weight goal."""
    if current_lbs is None or pd.isna(current_lbs):
        return None
    return current_lbs - WEIGHT_GOAL_LBS  # Already in lbs


def body_fat_to_goal(current_pct: float) -> float:
    """Calculate body fat % remaining to goal."""
    return current_pct - BODY_FAT_GOAL_PCT


def body_fat_change(current_pct: Optional[float], previous_pct: Optional[float]) -> Optional[float]:
    """Calculate body fat % change."""
    if current_pct is None or previous_pct is None:
        return None
    if pd.isna(current_pct) or pd.isna(previous_pct):
        return None
    return current_pct - previous_pct


# === PROTEIN METRICS ===

def protein_status(protein_g: float, weight_lbs: float) -> tuple[str, float, float]:
    """
    Returns (status, target, difference) for protein intake.
    Status: 'low', 'good', 'high'
    """
    min_p, target, max_p = get_protein_target(weight_lbs)
    
    if protein_g < min_p:
        return ('low', target, protein_g - min_p)
    elif protein_g > max_p:
        return ('high', target, protein_g - max_p)
    else:
        return ('good', target, 0)


# === SLEEP METRICS ===

def sleep_quality_score(row: pd.Series) -> Optional[float]:
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


def compare_periods(current_df: pd.DataFrame, previous_df: pd.DataFrame, column: str) -> Optional[float]:
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

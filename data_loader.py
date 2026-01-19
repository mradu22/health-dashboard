"""
Data loading and preprocessing for health dashboard.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import streamlit as st

from config import CSV_PATH


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data() -> pd.DataFrame:
    """
    Load health data from CSV.
    Returns DataFrame with date as index.
    """
    df = pd.read_csv(CSV_PATH, parse_dates=['date'])
    df = df.set_index('date').sort_index()
    
    # Forward-fill certain columns that should persist
    forward_fill_cols = ['weight_kg', 'body_fat_pct', 'muscle_mass_kg', 'bmi', 'vo2_max']
    for col in forward_fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    return df


def get_latest_data(df: pd.DataFrame) -> pd.Series:
    """Get the most recent day's data."""
    return df.iloc[-1]


def get_yesterday_data(df: pd.DataFrame) -> pd.Series | None:
    """Get yesterday's data if available."""
    if len(df) >= 2:
        return df.iloc[-2]
    return None


def get_week_data(df: pd.DataFrame, weeks_ago: int = 0) -> pd.DataFrame:
    """
    Get data for a specific week.
    weeks_ago=0 means current week, weeks_ago=1 means last week, etc.
    """
    today = df.index.max()
    # Get start of week (Monday)
    start_of_current_week = today - timedelta(days=today.weekday())
    start_of_target_week = start_of_current_week - timedelta(weeks=weeks_ago)
    end_of_target_week = start_of_target_week + timedelta(days=6)
    
    mask = (df.index >= start_of_target_week) & (df.index <= end_of_target_week)
    return df[mask]


def get_month_data(df: pd.DataFrame, months_ago: int = 0) -> pd.DataFrame:
    """
    Get data for a specific month.
    months_ago=0 means current month.
    """
    today = df.index.max()
    target_month = today.month - months_ago
    target_year = today.year
    
    # Handle year rollover
    while target_month <= 0:
        target_month += 12
        target_year -= 1
    
    mask = (df.index.month == target_month) & (df.index.year == target_year)
    return df[mask]


def get_year_data(df: pd.DataFrame, year: int | None = None) -> pd.DataFrame:
    """Get data for a specific year."""
    if year is None:
        year = df.index.max().year
    return df[df.index.year == year]


def get_n_days_data(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Get data for the last n days."""
    today = df.index.max()
    start_date = today - timedelta(days=n_days - 1)
    return df[df.index >= start_date]


def get_weight_n_days_ago(df: pd.DataFrame, n_days: int) -> float | None:
    """Get weight from n days ago."""
    today = df.index.max()
    target_date = today - timedelta(days=n_days)
    
    # Find closest date on or before target
    mask = df.index <= target_date
    if mask.any():
        closest = df[mask].iloc[-1]
        return closest.get('weight_kg')
    return None

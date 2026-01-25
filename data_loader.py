"""
Data loading and preprocessing for health dashboard.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import streamlit as st
import os

from config import CSV_PATH_LOCAL, GDRIVE_URL


@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data() -> pd.DataFrame:
    """
    Load health data from CSV.
    Tries Google Drive URL first (for Streamlit Cloud), falls back to local path.
    Returns DataFrame with date as index.
    """
    df = None
    
    # Try Google Drive first (works on Streamlit Cloud)
    try:
        df = pd.read_csv(GDRIVE_URL, parse_dates=['date'])
        st.sidebar.success("📡 Data: Google Drive")
    except Exception as e:
        pass
    
    # Fall back to local file (for local development)
    if df is None:
        if os.path.exists(CSV_PATH_LOCAL):
            df = pd.read_csv(CSV_PATH_LOCAL, parse_dates=['date'])
            st.sidebar.info("📁 Data: Local")
        else:
            st.error(f"Could not load data from Google Drive or local path")
            return pd.DataFrame()
    
    df = df.set_index('date').sort_index()
    
    # Filter to only show data from Jan 12, 2026 onwards
    jan_12_2026 = pd.Timestamp('2026-01-12')
    df = df[df.index >= jan_12_2026]
    
    # Forward-fill certain columns that should persist
    forward_fill_cols = ['weight_lbs', 'body_fat_pct', 'muscle_mass_kg', 'bmi', 'vo2_max', 
                         'squat_lbs', 'bench_lbs', 'deadlift_lbs']
    for col in forward_fill_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    return df


def get_latest_data(df: pd.DataFrame) -> pd.Series:
    """Get the most recent day's data."""
    return df.iloc[-1]


def get_yesterday_data(df: pd.DataFrame) -> Optional[pd.Series]:
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


def get_year_data(df: pd.DataFrame, year: Optional[int] = None) -> pd.DataFrame:
    """Get data for a specific year."""
    if year is None:
        year = df.index.max().year
    return df[df.index.year == year]


def get_n_days_data(df: pd.DataFrame, n_days: int) -> pd.DataFrame:
    """Get data for the last n days."""
    today = df.index.max()
    start_date = today - timedelta(days=n_days - 1)
    return df[df.index >= start_date]


def get_weight_n_days_ago(df: pd.DataFrame, n_days: int) -> Optional[float]:
    """Get weight from n days ago."""
    today = df.index.max()
    target_date = today - timedelta(days=n_days)
    
    # Find closest date on or before target
    mask = df.index <= target_date
    if mask.any():
        closest = df[mask].iloc[-1]
        return closest.get('weight_lbs')
    return None


def get_body_fat_n_days_ago(df: pd.DataFrame, n_days: int) -> Optional[float]:
    """Get body fat % from n days ago."""
    today = df.index.max()
    target_date = today - timedelta(days=n_days)
    
    # Find closest date on or before target
    mask = df.index <= target_date
    if mask.any():
        closest = df[mask].iloc[-1]
        return closest.get('body_fat_pct')
    return None


def get_vo2_max_n_days_ago(df: pd.DataFrame, n_days: int) -> Optional[float]:
    """Get VO2 max from n days ago."""
    today = df.index.max()
    target_date = today - timedelta(days=n_days)
    
    # Find closest date on or before target
    mask = df.index <= target_date
    if mask.any():
        closest = df[mask].iloc[-1]
        return closest.get('vo2_max')
    return None

"""
Health Dashboard - Personal health tracking with Streamlit.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional, Tuple

from data_loader import (
    load_data,
    get_latest_data,
    get_yesterday_data,
    get_week_data,
    get_month_data,
    get_n_days_data,
    get_weight_n_days_ago,
    get_body_fat_n_days_ago,
    get_vo2_max_n_days_ago,
)
from metrics import (
    gym_goal_progress,
    cardio_sessions_progress,
    cardio_km_week_progress,
    sleep_days_progress,
    get_weight_lbs,
    weight_change_lbs,
    weight_to_goal_lbs,
    body_fat_to_goal,
    body_fat_change,
    protein_status,
    rolling_average,
    meditation_streak,
)
from config import (
    VERSION,
    WEIGHT_GOAL_LBS,
    BODY_FAT_GOAL_PCT,
    SLEEP_HOURS_MIN,
)

# === PAGE CONFIG ===
st.set_page_config(
    page_title=f"LifeOS V{VERSION}",
    page_icon="◉",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# === CUSTOM CSS ===
st.markdown("""
<style>
    /* Dark theme overrides */
    .stMetric {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1rem;
        border-radius: 12px;
        border: 1px solid #30475e;
    }
    .stMetric label {
        color: #a0a0a0 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        font-weight: 700;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 600;
        color: #e0e0e0;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #30475e;
    }
    
    /* Trend card */
    .trend-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #30475e;
        text-align: center;
    }
    .trend-title {
        color: #a0a0a0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .trend-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
</style>
""", unsafe_allow_html=True)


def format_delta(value: Optional[float], unit: str = "", inverse: bool = False) -> str:
    """Format a delta value with + or - prefix."""
    if value is None or pd.isna(value):
        return "—"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.1f}{unit}"


def delta_color(value: Optional[float], inverse: bool = False) -> str:
    """Return color based on delta direction."""
    if value is None:
        return "off"
    if inverse:
        return "inverse"  # For weight loss: down is good
    return "normal"


def create_progress_ring(current: float, goal: float, label: str, unit: str = "") -> go.Figure:
    """Create a circular progress indicator."""
    pct = min(current / goal * 100, 100) if goal > 0 else 0
    
    fig = go.Figure(go.Pie(
        values=[pct, 100 - pct],
        hole=0.75,
        marker=dict(colors=['#4CAF50', '#1a1a2e']),
        textinfo='none',
        hoverinfo='none',
    ))
    
    # Format display based on whether it's a decimal or int
    if isinstance(current, float) and current != int(current):
        display_text = f"<b>{current:.1f}</b>/{goal:.0f}"
    else:
        display_text = f"<b>{int(current)}</b>/{int(goal)}"
    
    fig.add_annotation(
        text=display_text,
        x=0.5, y=0.5,
        font=dict(size=16, color='white'),
        showarrow=False,
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=5, r=5, t=30, b=5),
        height=140,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=label, font=dict(size=12, color='#a0a0a0'), x=0.5),
    )
    
    return fig


def create_trend_chart(df: pd.DataFrame, column: str, title: str, 
                       color: str = "#4CAF50", show_rolling: bool = True,
                       y_label: str = "") -> go.Figure:
    """Create a line chart with optional rolling average."""
    fig = go.Figure()
    
    # Raw data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='markers+lines',
        name='Daily',
        line=dict(color=color, width=1),
        marker=dict(size=5),
        opacity=0.6,
    ))
    
    # Rolling average
    if show_rolling and len(df) >= 3:
        rolling = rolling_average(df, column, window=7)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=rolling,
            mode='lines',
            name='7-day avg',
            line=dict(color=color, width=2),
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666', title=y_label),
        legend=dict(orientation='h', y=1.15, font=dict(size=10)),
        hovermode='x unified',
    )
    
    return fig


def fit_polynomial_with_ci(x_vals: np.ndarray, y_vals: np.ndarray, degree: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit polynomial and calculate 95% CI.
    Returns: (x_smooth, y_fit, y_lower, y_upper, coefficients)
    """
    # Remove NaN values
    mask = ~np.isnan(y_vals)
    x_clean = x_vals[mask]
    y_clean = y_vals[mask]
    
    if len(x_clean) < degree + 1:
        return None, None, None, None, None
    
    # Fit polynomial
    coeffs = np.polyfit(x_clean, y_clean, degree)
    poly = np.poly1d(coeffs)
    
    # Calculate fitted values and residuals
    y_fit = poly(x_clean)
    residuals = y_clean - y_fit
    
    # Standard error of estimate
    n = len(y_clean)
    p = degree + 1  # number of parameters
    sse = np.sum(residuals**2)
    mse = sse / (n - p)
    se = np.sqrt(mse)
    
    # For prediction interval (approximate 95% CI using t-distribution)
    # Simplified: using 1.96 * SE for approximate 95% CI
    t_val = 1.96
    
    # Generate smooth x values for plotting
    x_smooth = np.linspace(x_clean.min(), x_clean.max(), 50)
    y_smooth = poly(x_smooth)
    
    # Calculate leverage-based CI (simplified)
    # For a more accurate CI, we'd need the full variance-covariance matrix
    # This is an approximation
    ci_range = t_val * se * 1.5  # multiplier for visualization
    y_lower = y_smooth - ci_range
    y_upper = y_smooth + ci_range
    
    return x_smooth, y_smooth, y_lower, y_upper, coeffs


def format_poly_equation(coeffs: np.ndarray, ci_width: float) -> str:
    """Format polynomial coefficients for display with 95% CI."""
    if coeffs is None or len(coeffs) < 3:
        return ""
    # coeffs[0] is x^2, coeffs[1] is x, coeffs[2] is constant
    a, b, c = coeffs[0], coeffs[1], coeffs[2]
    # Show simplified: slope direction and CI
    slope_per_day = 2 * a * 7 + b  # approximate weekly slope at midpoint
    return f"Δ/wk: {slope_per_day:.2f} ± {ci_width:.1f}"


def create_weight_chart_lbs(df: pd.DataFrame) -> go.Figure:
    """Create weight chart in lbs with quadratic trendline and 95% CI."""
    fig = go.Figure()
    
    weight_lbs = df['weight_lbs'].dropna()
    
    # Weight data points
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['weight_lbs'],
        mode='markers',
        name='Weight',
        marker=dict(color='#2196F3', size=7),
    ))
    
    # Quadratic trendline with 95% CI
    if len(weight_lbs) >= 3:
        # Convert dates to numeric for regression
        x_numeric = np.arange(len(weight_lbs))
        y_vals = weight_lbs.values
        
        x_smooth, y_smooth, y_lower, y_upper, coeffs = fit_polynomial_with_ci(x_numeric, y_vals, degree=2)
        
        if x_smooth is not None:
            # Map smooth x values back to dates
            date_range = weight_lbs.index
            x_dates = pd.to_datetime([date_range[0] + pd.Timedelta(days=int(x)) for x in x_smooth])
            
            # 95% CI band
            fig.add_trace(go.Scatter(
                x=list(x_dates) + list(x_dates[::-1]),
                y=list(y_upper) + list(y_lower[::-1]),
                fill='toself',
                fillcolor='rgba(33, 150, 243, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
                name='95% CI',
            ))
            
            # Trendline
            fig.add_trace(go.Scatter(
                x=x_dates,
                y=y_smooth,
                mode='lines',
                name='Trend',
                line=dict(color='#FF5722', width=2, dash='solid'),
            ))
            
            # Calculate CI width for annotation
            ci_width = (y_upper[len(y_upper)//2] - y_lower[len(y_lower)//2]) / 2
            eq_text = format_poly_equation(coeffs, ci_width)
    
    fig.update_layout(
        title=dict(text="Weight Trajectory", font=dict(size=14)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666', title='lbs'),
        legend=dict(orientation='h', y=1.15, font=dict(size=9)),
        hovermode='x unified',
    )
    
    return fig


def create_body_fat_chart(df: pd.DataFrame) -> go.Figure:
    """Create body fat % chart with quadratic trendline and 95% CI."""
    fig = go.Figure()
    
    body_fat = df['body_fat_pct'].dropna()
    
    # Body fat data points
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['body_fat_pct'],
        mode='markers',
        name='Body Fat',
        marker=dict(color='#FF9800', size=7),
    ))
    
    # Quadratic trendline with 95% CI
    if len(body_fat) >= 3:
        x_numeric = np.arange(len(body_fat))
        y_vals = body_fat.values
        
        x_smooth, y_smooth, y_lower, y_upper, coeffs = fit_polynomial_with_ci(x_numeric, y_vals, degree=2)
        
        if x_smooth is not None:
            date_range = body_fat.index
            x_dates = pd.to_datetime([date_range[0] + pd.Timedelta(days=int(x)) for x in x_smooth])
            
            # 95% CI band
            fig.add_trace(go.Scatter(
                x=list(x_dates) + list(x_dates[::-1]),
                y=list(y_upper) + list(y_lower[::-1]),
                fill='toself',
                fillcolor='rgba(255, 152, 0, 0.15)',
                line=dict(color='rgba(255,255,255,0)'),
                hoverinfo='skip',
                showlegend=False,
                name='95% CI',
            ))
            
            # Trendline
            fig.add_trace(go.Scatter(
                x=x_dates,
                y=y_smooth,
                mode='lines',
                name='Trend',
                line=dict(color='#E91E63', width=2, dash='solid'),
            ))
    
    fig.update_layout(
        title=dict(text="Body Fat Trajectory", font=dict(size=14)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666', title='%'),
        legend=dict(orientation='h', y=1.15, font=dict(size=9)),
        hovermode='x unified',
    )
    
    return fig


def create_sleep_chart(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart for sleep breakdown with 7-hour goal line."""
    fig = go.Figure()
    
    sleep_cols = ['deep_sleep_hours', 'rem_hours', 'core_sleep_hours']
    colors = ['#1565C0', '#42A5F5', '#90CAF9']
    names = ['Deep', 'REM', 'Core']
    
    for col, color, name in zip(sleep_cols, colors, names):
        if col in df.columns:
            fig.add_trace(go.Bar(
                x=df.index,
                y=df[col],
                name=name,
                marker_color=color,
            ))
    
    # 7-hour goal line (no label - just the line)
    fig.add_hline(
        y=SLEEP_HOURS_MIN,
        line=dict(color='#4CAF50', dash='dash', width=2),
    )
    
    fig.update_layout(
        barmode='stack',
        title=dict(text="Sleep Breakdown", font=dict(size=14)),
        margin=dict(l=20, r=20, t=40, b=20),
        height=220,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666', title='hours'),
        legend=dict(orientation='h', y=1.15, font=dict(size=10)),
        hovermode='x unified',
    )
    
    return fig


# === MAIN APP ===
def main():
    # Load data
    df = load_data()
    
    if df.empty:
        st.error("No data found. Check CSV path in config.py")
        return
    
    today = get_latest_data(df)
    yesterday = get_yesterday_data(df)
    week_data = get_week_data(df, weeks_ago=0)
    
    # === HEADER ===
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.title(f"LifeOS V{VERSION}")
    with col_date:
        last_date = df.index.max().strftime("%b %d, %Y")
        st.caption(f"Last update: {last_date}")
    
    # === DAYSTATE (9 metrics in 3 rows of 3) ===
    st.markdown('<div class="section-header">DayState</div>', unsafe_allow_html=True)
    
    # Row 1: Steps, Total Sleep, VO2 Max (swapped from deep sleep)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        steps = today.get('steps', 0)
        steps_yd = yesterday.get('steps') if yesterday is not None else None
        steps_delta = steps - steps_yd if steps_yd else None
        st.metric("Steps", f"{steps:,.0f}", delta=format_delta(steps_delta))
    
    with col2:
        sleep = today.get('sleep_hours', 0)
        sleep_yd = yesterday.get('sleep_hours') if yesterday is not None else None
        sleep_delta = sleep - sleep_yd if sleep_yd else None
        st.metric("Total Sleep", f"{sleep:.1f}h", delta=format_delta(sleep_delta, "h"))
    
    with col3:
        # VO2 Max with delta (swapped from deep sleep)
        vo2 = today.get('vo2_max')
        vo2_yd = yesterday.get('vo2_max') if yesterday is not None else None
        vo2_delta = vo2 - vo2_yd if vo2 and vo2_yd else None
        if vo2:
            st.metric("VO2 Max", f"{vo2:.1f}", delta=format_delta(vo2_delta))
        else:
            st.metric("VO2 Max", "—")
    
    # Row 2: Weight, Body Fat, HRV
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight_lbs = get_weight_lbs(today.get('weight_lbs'))
        weight_yd_lbs = get_weight_lbs(yesterday.get('weight_lbs')) if yesterday is not None else None
        w_delta_lbs = weight_change_lbs(weight_lbs, weight_yd_lbs) if weight_lbs and weight_yd_lbs else None
        if weight_lbs:
            st.metric("Weight", f"{weight_lbs:.1f} lbs", delta=format_delta(w_delta_lbs, " lbs"), delta_color=delta_color(w_delta_lbs, inverse=True))
        else:
            st.metric("Weight", "—")
    
    with col2:
        bf = today.get('body_fat_pct')
        bf_yd = yesterday.get('body_fat_pct') if yesterday is not None else None
        bf_delta = bf - bf_yd if bf and bf_yd else None
        if bf:
            st.metric("Body Fat", f"{bf:.1f}%", delta=format_delta(bf_delta, "%"), delta_color=delta_color(bf_delta, inverse=True))
        else:
            st.metric("Body Fat", "—")
    
    with col3:
        hrv = today.get('hrv')
        hrv_yd = yesterday.get('hrv') if yesterday is not None else None
        hrv_delta = hrv - hrv_yd if hrv and hrv_yd else None
        if hrv:
            st.metric("HRV", f"{hrv:.0f} ms", delta=format_delta(hrv_delta, " ms"))
        else:
            st.metric("HRV", "—")
    
    # Row 3: Resting HR, Protein, Calories
    col1, col2, col3 = st.columns(3)
    
    with col1:
        rhr = today.get('resting_hr')
        rhr_yd = yesterday.get('resting_hr') if yesterday is not None else None
        rhr_delta = rhr - rhr_yd if rhr and rhr_yd else None
        if rhr:
            st.metric("Resting HR", f"{rhr:.0f} bpm", delta=format_delta(rhr_delta, " bpm"), delta_color=delta_color(rhr_delta, inverse=True))
        else:
            st.metric("Resting HR", "—")
    
    with col2:
        protein = today.get('protein_g')
        weight_for_protein = today.get('weight_lbs')
        if protein and weight_for_protein:
            # Protein target = weight in lbs as grams
            target = weight_for_protein
            st.metric("Protein", f"{protein:.0f}g", delta=f"Target: {target:.0f}g")
        elif protein:
            st.metric("Protein", f"{protein:.0f}g")
        else:
            st.metric("Protein", "—")
    
    with col3:
        cals = today.get('calories')
        cals_yd = yesterday.get('calories') if yesterday is not None else None
        cals_delta = cals - cals_yd if cals and cals_yd else None
        st.metric("Calories", f"{cals:,.0f}" if cals else "—", delta=format_delta(cals_delta) if cals_delta else None)
    
    # === WEEKLY GOALS (4 rings) ===
    st.markdown('<div class="section-header">Weekly Goals</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        gym_current, gym_goal, gym_pct = gym_goal_progress(week_data)
        fig = create_progress_ring(gym_current, gym_goal, "STRENGTH")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cardio_current, cardio_goal, cardio_pct = cardio_sessions_progress(week_data)
        fig = create_progress_ring(cardio_current, cardio_goal, "CARDIO")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        km_current, km_goal, km_pct = cardio_km_week_progress(week_data)
        fig = create_progress_ring(km_current, km_goal, "WEEKLY KM")
        st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        sleep_current, sleep_goal, sleep_pct = sleep_days_progress(week_data)
        fig = create_progress_ring(sleep_current, sleep_goal, "7+ HR SLEEP")
        st.plotly_chart(fig, use_container_width=True)
    
    # === TRAJECTORIES ===
    st.markdown('<div class="section-header">Trajectories</div>', unsafe_allow_html=True)
    
    # Trend cards row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        weight_lbs = get_weight_lbs(today.get('weight_lbs'))
        weight_1w_lbs = get_weight_lbs(get_weight_n_days_ago(df, 7))
        change_1w = weight_change_lbs(weight_lbs, weight_1w_lbs)
        color_1w = '#4CAF50' if change_1w and change_1w < 0 else '#FF5252' if change_1w and change_1w > 0 else '#fff'
        st.markdown(f"""
        <div class="trend-card">
            <div class="trend-title">Weight vs 1 Week</div>
            <div class="trend-value" style="color: {color_1w}">
                {format_delta(change_1w, ' lbs')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        weight_1m_lbs = get_weight_lbs(get_weight_n_days_ago(df, 30))
        change_1m = weight_change_lbs(weight_lbs, weight_1m_lbs)
        color_1m = '#4CAF50' if change_1m and change_1m < 0 else '#FF5252' if change_1m and change_1m > 0 else '#fff'
        st.markdown(f"""
        <div class="trend-card">
            <div class="trend-title">Weight vs 1 Month</div>
            <div class="trend-value" style="color: {color_1m}">
                {format_delta(change_1m, ' lbs')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        bf = today.get('body_fat_pct')
        bf_1w = get_body_fat_n_days_ago(df, 7)
        bf_change_1w = body_fat_change(bf, bf_1w)
        color_bf = '#4CAF50' if bf_change_1w and bf_change_1w < 0 else '#FF5252' if bf_change_1w and bf_change_1w > 0 else '#fff'
        st.markdown(f"""
        <div class="trend-card">
            <div class="trend-title">Body Fat vs 1 Week</div>
            <div class="trend-value" style="color: {color_bf}">
                {format_delta(bf_change_1w, '%')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        bf_1m = get_body_fat_n_days_ago(df, 30)
        bf_change_1m = body_fat_change(bf, bf_1m)
        color_bf_1m = '#4CAF50' if bf_change_1m and bf_change_1m < 0 else '#FF5252' if bf_change_1m and bf_change_1m > 0 else '#fff'
        st.markdown(f"""
        <div class="trend-card">
            <div class="trend-title">Body Fat vs 1 Month</div>
            <div class="trend-value" style="color: {color_bf_1m}">
                {format_delta(bf_change_1m, '%')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Trend charts (side by side)
    col1, col2 = st.columns(2)
    
    df_14d = get_n_days_data(df, 14)
    
    with col1:
        if 'weight_lbs' in df_14d.columns and df_14d['weight_lbs'].notna().any():
            fig = create_weight_chart_lbs(df_14d)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'body_fat_pct' in df_14d.columns and df_14d['body_fat_pct'].notna().any():
            fig = create_body_fat_chart(df_14d)
            st.plotly_chart(fig, use_container_width=True)
    
    # === HOMEOSTASIS ===
    st.markdown('<div class="section-header">Homeostasis</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    df_7d = get_n_days_data(df, 7)
    
    with col1:
        if 'sleep_hours' in df_7d.columns:
            fig = create_sleep_chart(df_7d)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'hrv' in df_7d.columns:
            fig = create_trend_chart(df_7d, 'hrv', 'HRV Trend', color='#9C27B0', y_label='ms')
            st.plotly_chart(fig, use_container_width=True)
    
    # Sleep stats - Deep Sleep moved here (swapped with VO2 Max)
    col1, col2 = st.columns(2)
    
    with col1:
        deep = today.get('deep_sleep_hours', 0)
        deep_yd = yesterday.get('deep_sleep_hours') if yesterday is not None else None
        deep_delta = deep - deep_yd if deep and deep_yd else None
        st.metric("Deep Sleep", f"{deep:.2f}h" if deep else "—", delta=format_delta(deep_delta, "h") if deep_delta else None)
    
    with col2:
        med_streak = meditation_streak(df)
        st.metric("Meditation Streak", f"{med_streak} days")
    
    # === FUEL STACK ===
    if today.get('protein_g') or today.get('calories'):
        st.markdown('<div class="section-header">Fuel Stack</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            cals = today.get('calories')
            st.metric("Calories", f"{cals:,.0f}" if cals else "—")
        
        with col2:
            protein = today.get('protein_g')
            st.metric("Protein", f"{protein:.0f}g" if protein else "—")
        
        with col3:
            carbs = today.get('carbs_g')
            st.metric("Carbs", f"{carbs:.0f}g" if carbs else "—")
        
        with col4:
            fat = today.get('fat_g')
            st.metric("Fat", f"{fat:.0f}g" if fat else "—")
        
        with col5:
            sodium = today.get('sodium_mg')
            st.metric("Sodium", f"{sodium:.0f}mg" if sodium else "—")
    
    # === FOOTER ===
    st.markdown("---")
    st.caption("Data from Apple Health via Health Auto Export")


if __name__ == "__main__":
    main()

"""
Health Dashboard - Personal health tracking with Streamlit.
"""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Optional

from data_loader import (
    load_data,
    get_latest_data,
    get_yesterday_data,
    get_week_data,
    get_month_data,
    get_year_data,
    get_n_days_data,
    get_weight_n_days_ago,
)
from metrics import (
    gym_goal_progress,
    cardio_month_progress,
    cardio_year_progress,
    weight_change,
    weight_to_goal,
    body_fat_to_goal,
    protein_status,
    sleep_quality_score,
    days_meeting_sleep_goal,
    rolling_average,
    meditation_streak,
)
from config import (
    WEIGHT_GOAL_KG,
    WEIGHT_GOAL_LBS,
    BODY_FAT_GOAL_PCT,
    GYM_SESSIONS_PER_WEEK,
    CARDIO_KM_PER_MONTH,
    CARDIO_KM_PER_YEAR,
    THEME,
)

# === PAGE CONFIG ===
st.set_page_config(
    page_title="Health Dashboard",
    page_icon="🏃",
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
        font-size: 2rem !important;
        font-weight: 700;
    }
    
    /* Progress ring styling */
    .progress-ring {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding: 1rem;
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        border: 1px solid #30475e;
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
    
    /* Goal card */
    .goal-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #30475e;
        text-align: center;
    }
    .goal-title {
        color: #a0a0a0;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }
    .goal-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #ffffff;
    }
    .goal-target {
        color: #666;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


def format_delta(value: Optional[float], unit: str = "", inverse: bool = False) -> str:
    """Format a delta value with + or - prefix."""
    if value is None:
        return "—"
    sign = "+" if value > 0 else ""
    # For weight loss, negative is good (inverse=True)
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
    
    fig.add_annotation(
        text=f"<b>{current:.0f}</b>/{goal:.0f}",
        x=0.5, y=0.5,
        font=dict(size=18, color='white'),
        showarrow=False,
    )
    
    fig.update_layout(
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        height=150,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        title=dict(text=label, font=dict(size=14, color='#a0a0a0'), x=0.5),
    )
    
    return fig


def create_trend_chart(df: pd.DataFrame, column: str, title: str, 
                       color: str = "#4CAF50", show_rolling: bool = True) -> go.Figure:
    """Create a line chart with optional rolling average."""
    fig = go.Figure()
    
    # Raw data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[column],
        mode='markers+lines',
        name='Daily',
        line=dict(color=color, width=1),
        marker=dict(size=6),
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
            line=dict(color=color, width=3),
        ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666'),
        legend=dict(orientation='h', y=1.1),
        hovermode='x unified',
    )
    
    return fig


def create_weight_chart(df: pd.DataFrame) -> go.Figure:
    """Create weight chart with goal line."""
    fig = go.Figure()
    
    # Weight data
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['weight_kg'],
        mode='markers+lines',
        name='Weight',
        line=dict(color='#2196F3', width=2),
        marker=dict(size=6),
    ))
    
    # Goal line
    fig.add_hline(
        y=WEIGHT_GOAL_KG,
        line=dict(color='#4CAF50', dash='dash', width=2),
        annotation_text=f"Goal: {WEIGHT_GOAL_LBS} lbs",
        annotation_position="right",
    )
    
    fig.update_layout(
        title=dict(text="Weight Trend", font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666', title='kg'),
        hovermode='x unified',
    )
    
    return fig


def create_sleep_chart(df: pd.DataFrame) -> go.Figure:
    """Create stacked bar chart for sleep breakdown."""
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
    
    fig.update_layout(
        barmode='stack',
        title=dict(text="Sleep Breakdown", font=dict(size=16)),
        margin=dict(l=20, r=20, t=50, b=20),
        height=250,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#666'),
        yaxis=dict(showgrid=True, gridcolor='#333', color='#666', title='hours'),
        legend=dict(orientation='h', y=1.1),
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
    last_week_data = get_week_data(df, weeks_ago=1)
    month_data = get_month_data(df, months_ago=0)
    year_data = get_year_data(df)
    
    # === HEADER ===
    col_title, col_date = st.columns([3, 1])
    with col_title:
        st.title("🏃 Health Dashboard")
    with col_date:
        last_date = df.index.max().strftime("%b %d, %Y")
        st.caption(f"Last update: {last_date}")
    
    # === TODAY'S SNAPSHOT ===
    st.markdown('<div class="section-header">📊 Today\'s Snapshot</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        steps = today.get('steps', 0)
        steps_yd = yesterday.get('steps') if yesterday is not None else None
        steps_delta = steps - steps_yd if steps_yd else None
        st.metric("Steps", f"{steps:,.0f}", delta=format_delta(steps_delta))
    
    with col2:
        sleep = today.get('sleep_hours', 0)
        sleep_yd = yesterday.get('sleep_hours') if yesterday is not None else None
        sleep_delta = sleep - sleep_yd if sleep_yd else None
        st.metric("Sleep", f"{sleep:.1f}h", delta=format_delta(sleep_delta, "h"))
    
    with col3:
        weight = today.get('weight_kg')
        weight_1w = get_weight_n_days_ago(df, 7)
        w_delta = weight_change(weight, weight_1w) if weight else None
        if weight:
            st.metric("Weight", f"{weight:.1f} kg", delta=format_delta(w_delta, " kg"), delta_color=delta_color(w_delta, inverse=True))
        else:
            st.metric("Weight", "—")
    
    with col4:
        hrv = today.get('hrv')
        hrv_yd = yesterday.get('hrv') if yesterday is not None else None
        hrv_delta = hrv - hrv_yd if hrv and hrv_yd else None
        if hrv:
            st.metric("HRV", f"{hrv:.0f} ms", delta=format_delta(hrv_delta, " ms"))
        else:
            st.metric("HRV", "—")
    
    # === WEEKLY GOALS ===
    st.markdown('<div class="section-header">🎯 This Week\'s Goals</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        gym_current, gym_goal, gym_pct = gym_goal_progress(week_data)
        fig = create_progress_ring(gym_current, gym_goal, "GYM SESSIONS")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        cardio_km, cardio_goal, cardio_pct = cardio_month_progress(month_data)
        fig = create_progress_ring(cardio_km, cardio_goal, "MONTHLY KM")
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        year_km, year_goal, year_pct = cardio_year_progress(year_data)
        fig = create_progress_ring(year_km, year_goal, f"YEAR KM ({datetime.now().year})")
        st.plotly_chart(fig, use_container_width=True)
    
    # === WEIGHT TRENDS ===
    st.markdown('<div class="section-header">⚖️ Weight & Body Composition</div>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        weight = today.get('weight_kg')
        weight_1w = get_weight_n_days_ago(df, 7)
        change_1w = weight_change(weight, weight_1w) if weight else None
        st.markdown(f"""
        <div class="goal-card">
            <div class="goal-title">vs 1 Week Ago</div>
            <div class="goal-value" style="color: {'#4CAF50' if change_1w and change_1w < 0 else '#FF5252' if change_1w and change_1w > 0 else '#fff'}">
                {format_delta(change_1w, ' kg') if change_1w else '—'}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        weight_2w = get_weight_n_days_ago(df, 14)
        change_2w = weight_change(weight, weight_2w) if weight and weight_2w else None
        change_2w_str = format_delta(change_2w, ' kg') if change_2w is not None and not pd.isna(change_2w) else '—'
        color_2w = '#4CAF50' if change_2w and change_2w < 0 else '#FF5252' if change_2w and change_2w > 0 else '#fff'
        st.markdown(f"""
        <div class="goal-card">
            <div class="goal-title">vs 2 Weeks Ago</div>
            <div class="goal-value" style="color: {color_2w}">
                {change_2w_str}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        to_goal = weight_to_goal(weight) if weight else None
        st.markdown(f"""
        <div class="goal-card">
            <div class="goal-title">To Goal ({WEIGHT_GOAL_LBS} lbs)</div>
            <div class="goal-value">{f'{to_goal:.1f} kg' if to_goal else '—'}</div>
        </div>
        """, unsafe_allow_html=True)
    
    # Weight chart
    df_14d = get_n_days_data(df, 14)
    if 'weight_kg' in df_14d.columns and df_14d['weight_kg'].notna().any():
        fig = create_weight_chart(df_14d)
        st.plotly_chart(fig, use_container_width=True)
    
    # Body composition row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        bf = today.get('body_fat_pct')
        if bf:
            bf_to_goal = body_fat_to_goal(bf)
            st.metric("Body Fat", f"{bf:.1f}%", delta=f"{bf_to_goal:.1f}% to goal", delta_color="inverse")
        else:
            st.metric("Body Fat", "—")
    
    with col2:
        mm = today.get('muscle_mass_kg')
        if mm:
            st.metric("Muscle Mass", f"{mm:.1f} kg")
        else:
            st.metric("Muscle Mass", "—")
    
    with col3:
        bmi = today.get('bmi')
        if bmi:
            st.metric("BMI", f"{bmi:.1f}")
        else:
            st.metric("BMI", "—")
    
    # === SLEEP & RECOVERY ===
    st.markdown('<div class="section-header">😴 Sleep & Recovery</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        df_7d = get_n_days_data(df, 7)
        if 'sleep_hours' in df_7d.columns:
            fig = create_sleep_chart(df_7d)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'hrv' in df_7d.columns:
            fig = create_trend_chart(df_7d, 'hrv', 'HRV Trend', color='#9C27B0')
            st.plotly_chart(fig, use_container_width=True)
    
    # Sleep stats
    col1, col2, col3 = st.columns(3)
    
    with col1:
        met, total, pct = days_meeting_sleep_goal(week_data)
        st.metric("7+ Hours (This Week)", f"{met}/{total}", delta=f"{pct:.0f}%")
    
    with col2:
        rhr = today.get('resting_hr')
        if rhr:
            st.metric("Resting HR", f"{rhr:.0f} bpm")
        else:
            st.metric("Resting HR", "—")
    
    with col3:
        med_streak = meditation_streak(df)
        st.metric("Meditation Streak", f"{med_streak} days")
    
    # === NUTRITION ===
    if today.get('protein_g') or today.get('calories'):
        st.markdown('<div class="section-header">🍎 Nutrition</div>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            protein = today.get('protein_g')
            weight = today.get('weight_kg')
            if protein and weight:
                status, target, diff = protein_status(protein, weight)
                color = '#4CAF50' if status == 'good' else '#FF9800' if status == 'low' else '#FF5252'
                st.metric("Protein", f"{protein:.0f}g", delta=f"Target: {target:.0f}g")
            elif protein:
                st.metric("Protein", f"{protein:.0f}g")
            else:
                st.metric("Protein", "—")
        
        with col2:
            carbs = today.get('carbs_g')
            st.metric("Carbs", f"{carbs:.0f}g" if carbs else "—")
        
        with col3:
            fat = today.get('fat_g')
            st.metric("Fat", f"{fat:.0f}g" if fat else "—")
        
        with col4:
            cals = today.get('calories')
            st.metric("Calories", f"{cals:.0f}" if cals else "—")
    
    # === FOOTER ===
    st.markdown("---")
    st.caption("Data from Apple Health via Health Auto Export")


if __name__ == "__main__":
    main()

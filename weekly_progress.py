"""
Weekly Progress Heatmap Module

Generates comprehensive weekly progress chart showing all goals across all weeks of the year.
Modular design for easy addition of new goals.
"""
from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

# Optional imports for alternative heatmap styles
try:
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

try:
    import altair as alt
    HAS_ALTAIR = True
except ImportError:
    HAS_ALTAIR = False

from data_loader import get_week_data
from metrics import (
    gym_goal_progress,
    cardio_sessions_progress,
    cardio_km_week_progress,
    stretch_sessions_progress,
    sleep_days_progress,
)
from config import (
    GYM_SESSIONS_PER_WEEK,
    CARDIO_SESSIONS_PER_WEEK,
    CARDIO_KM_PER_WEEK,
    STRETCH_SESSIONS_PER_WEEK,
    SLEEP_DAYS_PER_WEEK,
    CURRENT_YEAR,
)


def get_week_number(date: pd.Timestamp) -> int:
    """Get ISO week number for a date."""
    return date.isocalendar()[1]


def get_week_start_date(year: int, week: int) -> pd.Timestamp:
    """Get the start date (Monday) of a given ISO week in a year."""
    # January 4th is always in week 1 of ISO week numbering
    jan4 = pd.Timestamp(year, 1, 4)
    # Get the Monday of the week containing Jan 4
    days_to_monday = (jan4.weekday() - 0) % 7
    week1_monday = jan4 - pd.Timedelta(days=days_to_monday)
    # Add weeks (week 1 starts at week1_monday)
    week_start = week1_monday + pd.Timedelta(weeks=week - 1)
    return week_start


def get_all_weeks_in_year(year: int) -> List[Tuple[int, pd.Timestamp]]:
    """Get all week numbers and their start dates for a year."""
    weeks = []
    # ISO weeks can go up to 52 or 53
    for week_num in range(1, 54):
        week_start = get_week_start_date(year, week_num)
        # Stop if we've gone into next year
        if week_start.year > year:
            break
        weeks.append((week_num, week_start))
    return weeks


def calculate_weekly_goal_value(
    df: pd.DataFrame, week_start: pd.Timestamp, goal_type: str
) -> Tuple[Any, Any, float]:
    """
    Calculate goal progress for a specific week.
    Returns (current, goal, percentage).
    """
    # Filter to the specific week
    week_end = week_start + timedelta(days=6)
    week_mask = (df.index >= week_start) & (df.index <= week_end)
    week_data = df[week_mask]
    
    if len(week_data) == 0:
        return None, None, 0.0
    
    if goal_type == "strength":
        current, goal, pct = gym_goal_progress(week_data)
        return current, goal, pct
    elif goal_type == "cardio":
        current, goal, pct = cardio_sessions_progress(week_data)
        return current, goal, pct
    elif goal_type == "stretch":
        current, goal, pct = stretch_sessions_progress(week_data)
        return current, goal, pct
    elif goal_type == "weekly_km":
        current, goal, pct = cardio_km_week_progress(week_data)
        return current, goal, pct
    elif goal_type == "sleep":
        current, goal, pct = sleep_days_progress(week_data)
        return current, goal, pct
    else:
        return None, None, 0.0


def get_goal_config() -> List[Dict[str, Any]]:
    """
    Returns configuration for all weekly goals.
    Modular: add new goals here.
    """
    return [
        {
            "id": "strength",
            "label": "Strength",
            "goal": GYM_SESSIONS_PER_WEEK,
            "unit": "sessions",
            "weight": 1.0,
        },
        {
            "id": "cardio",
            "label": "Cardio",
            "goal": CARDIO_SESSIONS_PER_WEEK,
            "unit": "sessions",
            "weight": 1.0,
        },
        {
            "id": "stretch",
            "label": "Stretch",
            "goal": STRETCH_SESSIONS_PER_WEEK,
            "unit": "sessions",
            "weight": 0.7,
        },
        {
            "id": "weekly_km",
            "label": "Run",
            "goal": CARDIO_KM_PER_WEEK,
            "unit": "km",
            "weight": 1.3,
        },
        {
            "id": "sleep",
            "label": "Sleep",
            "goal": SLEEP_DAYS_PER_WEEK,
            "unit": "days",
            "weight": 1.5,
        },
    ]


def build_weekly_progress_matrix(df: pd.DataFrame, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build matrices for weekly progress data and percentages.
    Returns (values_df, percentages_df) where:
    - values_df: cells contain "current/goal" strings
    - percentages_df: cells contain percentage values for heatmap coloring
    """
    goals = get_goal_config()
    weeks = get_all_weeks_in_year(year)
    
    # Initialize matrices
    values_matrix = []
    percentages_matrix = []
    goal_labels = [g["label"] for g in goals]
    
    # Get today's date to determine which weeks are in the future
    today = pd.Timestamp.now()
    
    for goal in goals:
        value_row = []
        pct_row = []
        
        for week_num, week_start in weeks:
            week_end = week_start + timedelta(days=6)
            
            # Check if week is in the future
            if week_start > today:
                value_row.append("—")
                pct_row.append(None)  # None for future weeks (will be greyed out)
            else:
                current, goal_val, pct = calculate_weekly_goal_value(df, week_start, goal["id"])
                
                if current is None or goal_val is None:
                    value_row.append("—")
                    pct_row.append(0.0)
                else:
                    # Format value based on type
                    if goal["id"] == "weekly_km":
                        value_str = f"{current:.1f}/{goal_val:.0f}"
                    else:
                        value_str = f"{int(current)}/{int(goal_val)}"
                    value_row.append(value_str)
                    pct_row.append(pct)
        
        values_matrix.append(value_row)
        percentages_matrix.append(pct_row)
    
    # Calculate totals row (weighted average)
    totals_value_row = []
    totals_pct_row = []
    weights = [g.get("weight", 1.0) for g in goals]
    
    for week_idx in range(len(weeks)):
        week_num, week_start = weeks[week_idx]
        if week_start > today:
            totals_value_row.append("—")
            totals_pct_row.append(None)
        else:
            weighted_sum = 0.0
            valid_weights = 0.0
            for i, pct_row in enumerate(percentages_matrix):
                pct = pct_row[week_idx]
                if pct is not None and not (isinstance(pct, float) and np.isnan(pct)):
                    weighted_sum += pct * weights[i]
                    valid_weights += weights[i]
            if valid_weights > 0:
                avg_pct = weighted_sum / valid_weights
                totals_pct_row.append(avg_pct)
                totals_value_row.append(f"{avg_pct:.0f}%")
            else:
                totals_value_row.append("—")
                totals_pct_row.append(0.0)
    
    values_matrix.append(totals_value_row)
    percentages_matrix.append(totals_pct_row)
    goal_labels.append("Totals")
    
    # Create DataFrames - use just week numbers for cleaner look
    week_labels = [str(w[0]) for w in weeks]  # Just "1", "2", "3" instead of "Week 1", "Week 2"
    values_df = pd.DataFrame(values_matrix, index=goal_labels, columns=week_labels)
    percentages_df = pd.DataFrame(percentages_matrix, index=goal_labels, columns=week_labels)
    
    return values_df, percentages_df


def create_weekly_progress_heatmap_altair(df: pd.DataFrame, year: int = None):
    """
    Create GitHub-style weekly progress heatmap using Altair.
    Mimics GitHub contribution graph aesthetics with:
    - Goals on Y-axis (like days in GitHub)
    - Weeks on X-axis with month labels (Jan, Feb, etc.)
    - Square cells with rounded corners
    - GitHub green color gradient
    - Hover tooltips with all details
    
    Args:
        df: Full dataframe with all health data
        year: Year to display (defaults to current year)
    
    Returns:
        Altair chart object
    """
    if not HAS_ALTAIR:
        raise ImportError("altair is required. Install with: pip install altair")
    
    if year is None:
        year = CURRENT_YEAR
    
    values_df, percentages_df = build_weekly_progress_matrix(df, year)
    weeks = get_all_weeks_in_year(year)
    
    # Create month label mapping for x-axis (like GitHub shows Jan, Feb, etc.)
    # Map week numbers to month abbreviations - only show label at start of each month
    month_labels = {}
    current_month = None
    for week_num, week_start in weeks:
        month = week_start.strftime('%b')  # Jan, Feb, Mar, etc.
        if month != current_month:
            month_labels[week_num] = month
            current_month = month
        else:
            month_labels[week_num] = ''  # Empty for non-first weeks of month
    
    # Prepare data in long format for Altair
    data_list = []
    today = pd.Timestamp.now()
    
    for goal_idx, goal_name in enumerate(values_df.index):
        for week_idx, week_num in enumerate(values_df.columns):
            week_start = get_week_start_date(year, int(week_num))
            week_end = week_start + timedelta(days=6)
            
            value_str = values_df.iloc[goal_idx, week_idx]
            pct = percentages_df.iloc[goal_idx, week_idx]
            
            # Determine if week is in future
            is_future = week_start > today
            
            # Extract current/goal from value string (e.g., "3/3" -> current=3, goal=3)
            current_val = None
            goal_val = None
            if value_str != "—" and "/" in str(value_str):
                parts = str(value_str).split("/")
                try:
                    current_val = float(parts[0])
                    goal_val = float(parts[1])
                except:
                    pass
            
            # Map percentage to color level (0-4 for GitHub colors, -1 for future)
            if is_future:
                color_level = -1  # Future weeks - dark grey
            elif pct is None or np.isnan(pct) or pct == 0:
                color_level = 0  # Empty/zero - light grey
            elif pct < 25:
                color_level = 1  # L1 - light green
            elif pct < 50:
                color_level = 2  # L2 - medium green
            elif pct < 75:
                color_level = 3  # L3 - darker green
            elif pct < 100:
                color_level = 4  # L4 - darkest green
            else:
                color_level = 4  # 100% - darkest green
            
            # Get month label for this week
            month_label = month_labels.get(int(week_num), '')
            
            data_list.append({
                'goal': goal_name,
                'week': int(week_num),
                'week_start': week_start.strftime('%Y-%m-%d'),
                'week_end': week_end.strftime('%Y-%m-%d'),
                'month_label': month_label,
                'value': value_str,
                'percentage': pct if pct is not None and not np.isnan(pct) else 0,
                'color_level': color_level,
                'current': current_val,
                'goal_target': goal_val,
                'is_future': is_future,
            })
    
    chart_data = pd.DataFrame(data_list)
    
    # Generate expression for month labels on x-axis
    # Only show month name at first week of each month (like GitHub)
    label_expr_parts = []
    for week_num, label in month_labels.items():
        if label:
            label_expr_parts.append(f"datum.value == {week_num} ? '{label}'")
    label_expr = " : ".join(label_expr_parts) + " : ''" if label_expr_parts else "''"
    
    # Create the heatmap - GitHub contribution graph style
    # Using Step sizing for responsive layout
    heatmap = alt.Chart(chart_data).mark_rect(
        cornerRadius=3,  # Rounded corners like GitHub
    ).encode(
        x=alt.X('week:O',
                axis=alt.Axis(
                    title=None,
                    tickSize=0,
                    domain=False,
                    grid=False,
                    labelExpr=label_expr,  # Show month names (Jan, Feb, etc.)
                    labelColor='#8b949e',
                    labelFontSize=12,
                    labelFont='-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
                    labelAngle=0,
                    labelPadding=8,
                ),
                sort='ascending'),
        y=alt.Y('goal:O',
                axis=alt.Axis(
                    title=None,
                    tickSize=0,
                    domain=False,
                    grid=False,
                    labelColor='#c9d1d9',
                    labelFontSize=13,
                    labelFont='-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
                    labelPadding=10,
                ),
                sort=None),
        color=alt.Color('color_level:Q',
                       scale=alt.Scale(
                           domain=[-1, 0, 1, 2, 3, 4],
                           range=['#161b22', '#ebedf0', '#9be9a8', '#40c463', '#30a14e', '#216e39'],
                       ),
                       legend=None),
        tooltip=[
            alt.Tooltip('goal:N', title='Goal'),
            alt.Tooltip('week_start:N', title='Week Start'),
            alt.Tooltip('week_end:N', title='Week End'),
            alt.Tooltip('value:N', title='Progress'),
            alt.Tooltip('percentage:Q', title='Progress %', format='.0f'),
        ],
    ).properties(
        width=alt.Step(18),   # Larger cell width with step sizing
        height=alt.Step(28),  # Larger cell height with step sizing
    )
    
    # Apply configurations for GitHub-style appearance
    chart = heatmap.configure_scale(
        bandPaddingInner=0.05,  # More gap between cells (like GitHub)
    ).configure_view(
        stroke=None,
        fill='transparent',
    ).configure_axis(
        grid=False,
    ).configure(
        font='-apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif',
    )
    
    return chart


def get_codepen_color(percentage: float, is_future: bool = False) -> str:
    """
    Map percentage to color using CodePen-style brightness scaling.
    Based on green rgb(86, 222, 147) with brightness adjustment.
    
    Args:
        percentage: 0-100 completion percentage
        is_future: Whether this is a future week (darker)
    
    Returns:
        RGB color string
    """
    if is_future:
        return "rgb(26, 29, 36)"  # Dark background for future weeks
    
    # Base green color from CodePen: rgb(86, 222, 147)
    green = [86, 222, 147]
    
    # Map percentage to brightness factor
    # 0% -> 0.15 (very dim), 100% -> 1.0 (full brightness)
    min_brightness = 0.15
    max_brightness = 1.0
    factor = min_brightness + (percentage / 100) * (max_brightness - min_brightness)
    
    # Apply brightness to green
    r = int(green[0] * factor)
    g = int(green[1] * factor)
    b = int(green[2] * factor)
    
    return f"rgb({r}, {g}, {b})"


def create_weekly_progress_heatmap(df: pd.DataFrame, year: int = None) -> go.Figure:
    """
    Create CodePen-style weekly progress heatmap using Plotly.
    
    Inspired by: https://codepen.io/Xalsier/pen/EaVQJZO
    
    Features:
    - Responsive design (works on mobile and resize)
    - No numbers in cells - only on hover
    - Month labels on X-axis (Jan, Feb, etc.)
    - Goals on Y-axis
    - CodePen green color gradient with brightness scaling
    - Clean aesthetic with proper gaps
    
    Args:
        df: Full dataframe with all health data
        year: Year to display (defaults to current year)
    
    Returns:
        Plotly figure with heatmap
    """
    if year is None:
        year = CURRENT_YEAR
    
    values_df, percentages_df = build_weekly_progress_matrix(df, year)
    weeks = get_all_weeks_in_year(year)
    today = pd.Timestamp.now()
    
    # Update index to uppercase to match weekly goals style
    values_df.index = [label.upper() for label in values_df.index]
    percentages_df.index = values_df.index
    
    # Build color matrix and hover text
    num_goals = len(values_df.index)
    num_weeks = len(values_df.columns)
    
    # We'll use a custom approach: create colors for each cell
    colors = []
    hover_text = []
    z_data = []  # Normalized data for heatmap
    
    for goal_idx, goal_name in enumerate(values_df.index):
        color_row = []
        hover_row = []
        z_row = []
        
        for week_idx, week_num in enumerate(values_df.columns):
            week_start = get_week_start_date(year, int(week_num))
            week_end = week_start + timedelta(days=6)
            value = values_df.iloc[goal_idx, week_idx]
            pct = percentages_df.iloc[goal_idx, week_idx]
            
            is_future = week_start > today
            
            if is_future or pct is None or (isinstance(pct, float) and np.isnan(pct)):
                pct_val = 0
                is_future = True
            else:
                pct_val = float(pct)
            
            # Get color based on percentage
            color = get_codepen_color(pct_val, is_future)
            color_row.append(color)
            
            # Normalize z value: -1 for future, 0-100 for actual
            if is_future:
                z_row.append(-10)  # Special value for future
            else:
                z_row.append(pct_val)
            
            # Create hover text
            if is_future:
                hover = f"<b>{goal_name}</b><br>Week {week_num}<br>{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}<br><i>Future week</i>"
            else:
                hover = f"<b>{goal_name}</b><br>Week {week_num}<br>{week_start.strftime('%b %d')} - {week_end.strftime('%b %d')}<br>Progress: {value}<br>Completion: {pct_val:.0f}%"
            hover_row.append(hover)
        
        colors.append(color_row)
        hover_text.append(hover_row)
        z_data.append(z_row)
    
    # Create month labels for X-axis (excluding December)
    month_ticks = []
    month_labels = []
    current_month = None
    for week_num, week_start in weeks:
        month = week_start.strftime('%b')
        # Skip December
        if month == 'Dec':
            continue
        if month != current_month:
            month_ticks.append(str(week_num))
            month_labels.append(month)
            current_month = month
    
    # CodePen-inspired colorscale using green brightness
    # Base green: rgb(86, 222, 147)
    colorscale = [
        [0.0, "rgb(26, 29, 36)"],      # Future weeks - dark background
        [0.001, "rgb(26, 29, 36)"],    # Future weeks boundary
        [0.01, "rgb(13, 33, 22)"],     # 0% - very dim green
        [0.10, "rgb(17, 44, 29)"],     # 10%
        [0.25, "rgb(21, 55, 37)"],     # 25% - dim green
        [0.50, "rgb(43, 111, 74)"],    # 50% - medium green
        [0.75, "rgb(64, 166, 110)"],   # 75% - bright green
        [1.0, "rgb(86, 222, 147)"],    # 100% - full brightness green
    ]
    
    fig = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            z=z_data,
            x=list(values_df.columns),
            y=list(values_df.index),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            colorscale=colorscale,
            zmin=-10,
            zmax=100,
            showscale=False,
            xgap=4.2,   # Gap between cells - 5% more space (4 * 1.05 = 4.2)
            ygap=4,   # Gap between cells
        )
    )
    
    # Responsive design: maintain square cells that scale with container width
    # For square cells: cell_width = cell_height
    # If we have num_weeks horizontally and num_goals vertically:
    #   plot_width / num_weeks = plot_height / num_goals
    #   plot_height = (plot_width / num_weeks) * num_goals
    
    # Calculate margins (left, right, top, bottom)
    margin_left = 100
    margin_right = 20
    margin_top = 15
    margin_bottom = 50
    total_h_margin = margin_left + margin_right
    total_v_margin = margin_top + margin_bottom
    
    # Base width estimate (will scale with container via autosize)
    # Use a reasonable base that works well on desktop
    base_width = 1200
    
    # Calculate height to maintain square cells
    plot_width = base_width - total_h_margin
    cell_width = plot_width / num_weeks
    # For square cells, cell_height = cell_width
    plot_height = cell_width * num_goals
    responsive_height = plot_height + total_v_margin
    
    # Minimum height for mobile (ensures cells don't get too small)
    min_cell_size = 8
    min_plot_height = num_goals * (min_cell_size + 4)  # +4 for gap
    min_height = min_plot_height + total_v_margin
    
    # Use calculated height, but ensure minimum for mobile
    final_height = max(int(responsive_height), min_height)
    
    fig.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=month_ticks,
            ticktext=month_labels,
            tickangle=0,
            tickfont=dict(size=12, color="#a0a0a0", family="system-ui, -apple-system, sans-serif"),
            showgrid=False,
            showline=False,
            zeroline=False,
            side='bottom',
            fixedrange=True,
            showticklabels=True,  # Show month labels but no tick marks
            ticks='',  # Remove tick marks
        ),
        yaxis=dict(
            tickfont=dict(size=12, color="#a0a0a0", family="system-ui, -apple-system, sans-serif"),
            showgrid=False,
            showline=False,
            zeroline=False,
            autorange='reversed',  # First goal at top
            fixedrange=True,
            ticks='',  # Remove tick marks
            # Maintain square cells: aspect ratio = num_goals / num_weeks
            # This ensures cell_width = cell_height as container resizes
            scaleanchor='x',
            scaleratio=num_goals / num_weeks,
        ),
        # Set responsive dimensions (will scale with container via autosize)
        height=final_height,
        width=base_width,
        margin=dict(l=margin_left, r=margin_right, t=margin_top, b=margin_bottom),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"),
        hoverlabel=dict(
            bgcolor="rgb(26, 29, 36)",
            bordercolor="rgb(42, 45, 53)",
            font=dict(size=13, color="#ffffff", family="system-ui, sans-serif"),
        ),
        # Enable responsive sizing - this will make it adapt to container width
        autosize=True,
    )
    
    return fig


def create_weekly_progress_heatmap_seaborn(df: pd.DataFrame, year: int = None) -> plt.Figure:
    """
    Create beautiful weekly progress heatmap using Seaborn.
    More aesthetic than Plotly version with better styling.
    Requires seaborn and matplotlib to be installed.
    
    Args:
        df: Full dataframe with all health data
        year: Year to display (defaults to current year)
    
    Returns:
        Matplotlib figure with heatmap
    """
    if not HAS_SEABORN:
        raise ImportError("seaborn and matplotlib are required for this function. Install with: pip install seaborn matplotlib")
    
    if year is None:
        year = CURRENT_YEAR
    
    values_df, percentages_df = build_weekly_progress_matrix(df, year)
    
    # Set aesthetic style
    sns.set_style("darkgrid", {
        "axes.facecolor": "#0E1117",
        "figure.facecolor": "#0E1117",
        "axes.labelcolor": "#FAFAFA",
        "text.color": "#FAFAFA",
        "xtick.color": "#FAFAFA",
        "ytick.color": "#FAFAFA",
        "axes.edgecolor": "#30475e",
        "grid.color": "#1a1a2e",
    })
    
    # Prepare data - convert percentages to 0-100 range, keep -1 for future
    heatmap_data = percentages_df.values.copy()
    heatmap_data = np.array([[float(x) if x is not None and not np.isnan(x) else -1 for x in row] for row in heatmap_data])
    
    # Create custom colormap
    # Grey for future (-1), then gradient from red (0%) through yellow (50%) to green (100%+)
    colors = ['#666666', '#FF5252', '#FF9800', '#FFC107', '#8BC34A', '#4CAF50', '#00C853']
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('progress', colors, N=n_bins)
    
    # Create figure with dark theme
    fig, ax = plt.subplots(figsize=(20, 6), facecolor='#0E1117')
    
    # Create mask for future weeks (where value is -1)
    mask = (heatmap_data == -1)
    
    # Normalize data: -1 stays as is, 0-100 maps to 0-100
    # For colormap, we'll handle -1 separately
    plot_data = heatmap_data.copy()
    plot_data[plot_data == -1] = -10  # Temporary value for masking
    
    # Create heatmap
    sns.heatmap(
        plot_data,
        annot=values_df.values,
        fmt='',
        cmap=cmap,
        vmin=-10,
        vmax=100,
        center=50,
        square=False,
        linewidths=0.5,
        linecolor='#30475e',
        cbar_kws={
            'label': 'Progress %',
            'orientation': 'vertical',
            'shrink': 0.8,
        },
        ax=ax,
        mask=mask,  # Mask future weeks
        annot_kws={'size': 9, 'color': 'white', 'weight': 'bold'},
        xticklabels=values_df.columns,
        yticklabels=values_df.index,
    )
    
    # Style the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Progress %', rotation=270, labelpad=20, color='#FAFAFA', fontsize=11)
    cbar.ax.tick_params(colors='#FAFAFA', labelsize=9)
    
    # Customize ticks and labels
    ax.set_xlabel('Week', fontsize=12, color='#FAFAFA', fontweight='600')
    ax.set_ylabel('Goal', fontsize=12, color='#FAFAFA', fontweight='600')
    ax.set_title(f'Weekly Progress Heatmap - {year}', fontsize=16, color='#FAFAFA', fontweight='700', pad=20)
    
    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', color='#FAFAFA', fontsize=9)
    plt.setp(ax.get_yticklabels(), color='#FAFAFA', fontsize=10, fontweight='500')
    
    # Add grey background for future weeks (manually draw rectangles)
    for i in range(len(values_df.index)):
        for j in range(len(values_df.columns)):
            if mask[i, j]:
                rect = plt.Rectangle((j, i), 1, 1, facecolor='#2a2a3e', edgecolor='#30475e', linewidth=0.5)
                ax.add_patch(rect)
                # Add "—" text
                ax.text(j + 0.5, i + 0.5, '—', ha='center', va='center', 
                       color='#666666', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    return fig

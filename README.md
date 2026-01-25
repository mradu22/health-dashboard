# Health Dashboard

Personal health tracking dashboard built with Streamlit.

## Features

- **Today's Snapshot**: Steps, sleep, weight, HRV with day-over-day changes
- **Weekly Goals**: Gym sessions (3x/week), monthly/yearly cardio km
- **Weight Trends**: 1-week and 2-week changes, progress to goal
- **Body Composition**: Body fat %, muscle mass, BMI
- **Sleep & Recovery**: Sleep breakdown, HRV trends, meditation streak
- **Nutrition**: Protein tracking vs body weight target, macros

## Goals Tracked

| Goal | Target |
|------|--------|
| Gym | 3x/week |
| Cardio | 100 km/month, 12,000 km/year |
| Protein | Body weight (lbs) in grams ±15g |
| Weight | 180 lbs |
| Body Fat | <16% |

## Setup

### Local Development

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Update `config.py` with your CSV path

3. Run the dashboard:
   ```bash
   streamlit run app.py
   ```

### Streamlit Cloud Deployment

1. Push to GitHub
2. Connect repo to [Streamlit Cloud](https://streamlit.io/cloud)
3. Set up Google Sheets connection for data (see below)

## Data Source

The dashboard reads from a CSV with health metrics exported from Apple Health.

**Columns expected:**
- `date`, `steps`, `active_cal`, `walk_and_run_distance_km`, `run_km`
- `strength_training_min`, `cardio_min`, `stretch_min`, `workout_avg_hr`
- `sleep_hours`, `deep_sleep_hours`, `rem_hours`, `core_sleep_hours`, `awake_hours`, `sleep_interruptions`
- `protein_g`, `carbs_g`, `fat_g`, `calories`, `sodium_mg`
- `weight_lbs`, `body_fat_pct`, `muscle_mass_kg`, `bmi`
- `resting_hr`, `rsr`, `vo2_max`, `hrv`, `meditation_min`
- `squat_lbs`, `bench_lbs`, `deadlift_lbs`

## Notes

- `run_km` = running distance only (extracted from running workouts)
- `walk_and_run_distance_km` = combined walking + running distance (from Apple Health)
- Dashboard uses `run_km` for cardio goals
- Data auto-refreshes every 5 minutes when the page is open

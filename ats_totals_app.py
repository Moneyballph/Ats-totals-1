# Moneyball Phil — ATS & Totals App (v2.5 Fixed Submit Buttons)
# Sports: MLB, NFL, NBA, NCAA Football, NCAA Basketball

import streamlit as st
import pandas as pd
import datetime
import math

# ---------------- Page Setup ----------------
st.set_page_config(page_title="Moneyball Phil — ATS & Totals", layout="wide")
st.title("🏆 Moneyball Phil — ATS & Totals App")

# ---------------- Session State ----------------
def init_state():
    defaults = {
        "parlay_slip": [],
        "last_sport": None,
        # inputs
        "home": "", "away": "",
        "home_pf": 0.0, "home_pa": 0.0,
        "away_pf": 0.0, "away_pa": 0.0,
        "spread_line_home": 0.0,     # enter home line; away mirrors
        "spread_odds_home": -110.0,
        "spread_odds_away": -110.0,
        "total_line": 0.0, "over_odds": -110.0, "under_odds": -110.0,
        "stake": 0.0,
        # selection & persisted results
        "selected_bet": None,
        "results_df": None,
        "proj_total": None,
        "proj_margin": None,
        "proj_home_pts": None,
        "proj_away_pts": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ---------------- Utilities ----------------
def american_to_implied(odds: float) -> float:
    return (100 / (odds + 100)) if odds > 0 else (abs(odds) / (abs(odds) + 100))

def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))

def calculate_ev_pct(true_prob_pct: float, odds: float) -> tuple[float, float]:
    implied = american_to_implied(odds) * 100
    return (true_prob_pct - implied), implied

def tier_by_true_prob(true_prob_pct: float) -> tuple[str, str]:
    if true_prob_pct >= 80:
        return "Elite", "#16a34a"
    if true_prob_pct >= 65:
        return "Strong", "#2563eb"
    if true_prob_pct >= 50:
        return "Moderate", "#f59e0b"
    return "Risky", "#dc2626"

def tier_badge_html(tier_name: str, color: str) -> str:
    return f"<span style='background:{color};color:white;padding:4px 10px;border-radius:999px;font-weight:700;display:inline-block'>{tier_name}</span>"

def simple_bar(label: str, pct: float):
    pct = max(0.0, min(100.0, float(pct)))
    st.markdown(
        f"""
        <div style="margin:8px 0 4px 0;">
          <div style="font-size:12px;opacity:.85">{label}: {pct:.1f}%</div>
          <div style="height:8px;background:#2f2f2f;border-radius:6px;">
            <div style="height:8px;width:{pct}%;background:#4fa3ff;border-radius:6px;"></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def _std_norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_sport_sigmas(sport: str) -> tuple[float, float]:
    if sport == "MLB": return 3.5, 3.0
    if sport == "NFL": return 10.0, 9.0
    if sport == "NCAA Football": return 12.0, 10.0
    if sport == "NBA": return 15.0, 12.0
    if sport == "NCAA Basketball": return 18.0, 14.0
    return 12.0, 10.0

# ---------------- Baseline Model ----------------
def project_scores_base(sport: str, H_pf: float, H_pa: float, A_pf: float, A_pa: float):
    return (H_pf + A_pa) / 2.0, (A_pf + H_pa) / 2.0

# ---------------- Adjustments ----------------
def apply_adjustments(
    sport: str,
    H_pts: float, A_pts: float,
    home_edge_pts: float, away_edge_pts: float,
    form_H_pct: float, form_A_pct: float,
    injury_H_pct: float, injury_A_pct: float,
    pace_pct_global: float, variance_pct: float,
    plays_pct: float = 0.0, redzone_H_pct: float = 0.0, redzone_A_pct: float = 0.0,
    to_margin_pts: float = 0.0,
    pace_pct_hoops: float = 0.0, ortg_H_pct: float = 0.0, ortg_A_pct: float = 0.0,
    drtg_H_pct: float = 0.0, drtg_A_pct: float = 0.0, rest_H_pct: float = 0.0, rest_A_pct: float = 0.0,
    sp_H_runs: float = 0.0, sp_A_runs: float = 0.0, bullpen_H_runs: float = 0.0, bullpen_A_runs: float = 0.0,
    park_total_pct: float = 0.0, weather_total_pct: float = 0.0
):
    sd_total, sd_margin = get_sport_sigmas(sport)
    H_pts *= (1 + (form_H_pct + injury_H_pct)/100.0)
    A_pts *= (1 + (form_A_pct + injury_A_pct)/100.0)
    H_pts += home_edge_pts; A_pts += away_edge_pts
    if sport in ["NFL","NCAA Football"]:
        if plays_pct: 
            scale = 1 + plays_pct/100.0
            H_pts *= scale; A_pts *= scale
        H_pts *= (1 + redzone_H_pct/100.0); A_pts *= (1 + redzone_A_pct/100.0)
        H_pts += max(0.0,to_margin_pts); A_pts += max(0.0,-to_margin_pts)
    if sport in ["NBA","NCAA Basketball"]:
        if pace_pct_hoops:
            scale = 1 + pace_pct_hoops/100.0
            H_pts *= scale; A_pts *= scale
        H_pts *= (1 + (ortg_H_pct - drtg_A_pct)/100.0)
        A_pts *= (1 + (ortg_A_pct - drtg_H_pct)/100.0)
        H_pts *= (1 + rest_H_pct/100.0); A_pts *= (1 + rest_A_pct/100.0)
    if sport=="MLB":
        H_pts += sp_H_runs + bullpen_H_runs
        A_pts += sp_A_runs + bullpen_A_runs
        scale = (1 + park_total_pct/100.0) * (1 + weather_total_pct/100.0)
        H_pts *= scale; A_pts *= scale
    if pace_pct_global:
        scale = 1 + pace_pct_global/100.0
        H_pts *= scale; A_pts *= scale
    sd_total *= (1 + variance_pct/100.0); sd_margin *= (1 + variance_pct/100.0)
    return H_pts, A_pts, sd_total, sd_margin

# ---------------- Layout ----------------
sport = st.selectbox("Select Sport", ["MLB","NFL","NBA","NCAA Football","NCAA Basketball"])
col_inputs, col_results = st.columns([1,2])

with col_inputs:
    st.header("📥 Inputs")
    with st.form("inputs_form", clear_on_submit=False):
        st.text_input("Home Team", key="home")
        st.text_input("Away Team", key="away")
        st.number_input("Home: Avg Scored", key="home_pf")
        st.number_input("Home: Avg Allowed", key="home_pa")
        st.number_input("Away: Avg Scored", key="away_pf")
        st.number_input("Away: Avg Allowed", key="away_pa")
        st.number_input("Home Spread (enter negative if favorite)", key="spread_line_home")
        st.caption(f"Away Spread (auto): {(-st.session_state.spread_line_home):+.2f}")
        st.number_input("Home Spread Odds", key="spread_odds_home")
        st.number_input("Away Spread Odds", key="spread_odds_away")
        st.number_input("Total Line", key="total_line")
        st.number_input("Over Odds", key="over_odds")
        st.number_input("Under Odds", key="under_odds")
        st.number_input("Stake ($)", key="stake")
        # --- buttons restored ---
        run_projection = st.form_submit_button("🔮 Run Projection")
        reset_clicked = st.form_submit_button("Reset Inputs")

# ---------------- Results ----------------
with col_results:
    st.header("📊 Results")
    if run_projection:
        st.success("Projection ran successfully ✅")





# Moneyball Phil — ATS & Totals App (v2.4 with Auto-Volatility)
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
    # Return (name, hex color)
    if true_prob_pct >= 80:
        return "Elite", "#16a34a"   # green-600
    if true_prob_pct >= 65:
        return "Strong", "#2563eb"  # blue-600
    if true_prob_pct >= 50:
        return "Moderate", "#f59e0b" # amber-500
    return "Risky", "#dc2626"       # red-600

def tier_badge_html(tier_name: str, color: str) -> str:
    return (
        f"<span style='background:{color};color:white;"
        f"padding:4px 10px;border-radius:999px;font-weight:700;"
        f"display:inline-block'>{tier_name}</span>"
    )

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
    if sport == "MLB":               # runs
        return 3.5, 3.0
    if sport == "NFL":               # points
        return 10.0, 9.0
    if sport == "NCAA Football":
        return 12.0, 10.0
    if sport == "NBA":
        return 15.0, 12.0
    if sport == "NCAA Basketball":
        return 18.0, 14.0
    return 12.0, 10.0

# ---------------- Auto Volatility ----------------
def auto_volatility(proj_total: float, vegas_total: float, proj_margin: float, vegas_spread: float) -> float:
    """
    Determine automatic volatility adjustment based on how close projections are to Vegas lines.
    Returns 0, 10, or 20 (%).
    """
    # Compare projected spread vs Vegas spread
    spread_gap = abs(proj_margin - vegas_spread)
    # Compare projected total vs Vegas total
    total_gap = abs(proj_total - vegas_total)

    # Look at whichever line is tighter (closer to projection)
    min_gap = min(spread_gap, total_gap)

    if min_gap <= 0.5:
        return 20.0
    elif min_gap <= 1.5:
        return 10.0
    else:
        return 0.0

# ---------------- Baseline Model ----------------
def project_scores_base(sport: str, H_pf: float, H_pa: float, A_pf: float, A_pa: float) -> tuple[float, float]:
    home_pts = (H_pf + A_pa) / 2.0
    away_pts = (A_pf + H_pa) / 2.0
    return home_pts, away_pts

# ---------------- Adjustments ----------------
def apply_adjustments(
    sport: str,
    H_pts: float, A_pts: float,
    # universal
    home_edge_pts: float, away_edge_pts: float,
    form_H_pct: float, form_A_pct: float,
    injury_H_pct: float, injury_A_pct: float,
    pace_pct_global: float, variance_pct: float,
    # NFL / NCAA Football
    plays_pct: float = 0.0, redzone_H_pct: float = 0.0, redzone_A_pct: float = 0.0,
    to_margin_pts: float = 0.0,
    # NBA / NCAA Basketball
    pace_pct_hoops: float = 0.0, ortg_H_pct: float = 0.0, ortg_A_pct: float = 0.0,
    drtg_H_pct: float = 0.0, drtg_A_pct: float = 0.0, rest_H_pct: float = 0.0, rest_A_pct: float = 0.0,
    # MLB
    sp_H_runs: float = 0.0, sp_A_runs: float = 0.0, bullpen_H_runs: float = 0.0, bullpen_A_runs: float = 0.0,
    park_total_pct: float = 0.0, weather_total_pct: float = 0.0
):
    sd_total, sd_margin = get_sport_sigmas(sport)

    # Universal
    H_pts *= (1 + (form_H_pct + injury_H_pct)/100.0)
    A_pts *= (1 + (form_A_pct + injury_A_pct)/100.0)
    H_pts += home_edge_pts
    A_pts += away_edge_pts

    # Football
    if sport in ["NFL", "NCAA Football"]:
        if plays_pct != 0:
            scale = 1 + plays_pct/100.0
            total_before = H_pts + A_pts
            if total_before > 0:
                factor = (total_before * scale) / total_before
                H_pts *= factor
                A_pts *= factor
        H_pts *= (1 + redzone_H_pct/100.0)
        A_pts *= (1 + redzone_A_pct/100.0)
        H_pts += max(0.0, to_margin_pts)
        A_pts += max(0.0, -to_margin_pts)

    # Basketball
    if sport in ["NBA", "NCAA Basketball"]:
        if pace_pct_hoops != 0:
            scale = 1 + pace_pct_hoops/100.0
            total_before = H_pts + A_pts
            if total_before > 0:
                factor = (total_before * scale) / total_before
                H_pts *= factor
                A_pts *= factor
        H_pts *= (1 + (ortg_H_pct - drtg_A_pct)/100.0)
        A_pts *= (1 + (ortg_A_pct - drtg_H_pct)/100.0)
        H_pts *= (1 + rest_H_pct/100.0)
        A_pts *= (1 + rest_A_pct/100.0)

    # MLB
    if sport == "MLB":
        H_pts += sp_H_runs + bullpen_H_runs
        A_pts += sp_A_runs + bullpen_A_runs
        total_scale = (1 + park_total_pct/100.0) * (1 + weather_total_pct/100.0)
        total_before = H_pts + A_pts
        if total_before > 0 and total_scale != 1.0:
            factor = (total_before * total_scale) / total_before
            H_pts *= factor
            A_pts *= factor

    # Global pace
    if pace_pct_global != 0:
        scale = 1 + pace_pct_global/100.0
        total_before = H_pts + A_pts
        if total_before > 0:
            factor = (total_before * scale) / total_before
            H_pts *= factor
            A_pts *= factor

    # Variance tweak
    sd_total *= (1 + variance_pct/100.0)
    sd_margin *= (1 + variance_pct/100.0)

    return H_pts, A_pts, sd_total, sd_margin

# ---------------- Sport Selector ----------------
sport = st.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NCAA Football", "NCAA Basketball"])
if st.session_state.get("last_sport") != sport:
    # clear inputs on sport change (keep parlay + results)
    for key in ["home","away","home_pf","home_pa","away_pf","away_pa",
                "spread_line_home","spread_odds_home","spread_odds_away",
                "total_line","over_odds","under_odds","stake"]:
        st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
    st.session_state["selected_bet"] = None
    st.session_state["last_sport"] = sport

# ---------------- Layout ----------------
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.header("📥 Inputs")

    # All inputs in a FORM to prevent auto-reruns
    with st.form("inputs_form", clear_on_submit=False):
        # Row 1: Teams
        n1, n2 = st.columns(2)
        with n1:
            st.session_state.home = st.text_input("Home Team", value=st.session_state.home)
        with n2:
            st.session_state.away = st.text_input("Away Team", value=st.session_state.away)

        # Row 2: PF/PA
        h_col, a_col = st.columns(2)
        with h_col:
            st.caption("**Home Averages**")
            st.session_state.home_pf = st.number_input("Home: Avg Scored", step=0.01, format="%.2f", value=float(st.session_state.home_pf))
            st.session_state.home_pa = st.number_input("Home: Avg Allowed", step=0.01, format="%.2f", value=float(st.session_state.home_pa))
        with a_col:
            st.caption("**Away Averages**")
            st.session_state.away_pf = st.number_input("Away: Avg Scored", step=0.01, format="%.2f", value=float(st.session_state.away_pf))
            st.session_state.away_pa = st.number_input("Away: Avg Allowed", step=0.01, format="%.2f", value=float(st.session_state.away_pa))

        # Row 3: Spread + odds
        s1, s2 = st.columns(2)
        with s1:
            st.session_state.spread_line_home = st.number_input(
                "Home Spread (enter negative if favorite)",
                step=0.01, format="%.2f", value=float(st.session_state.spread_line_home)
            )
        with s2:
            st.caption(f"Away Spread (auto): {(-st.session_state.spread_line_home):+.2f}")

        so1, so2 = st.columns(2)
        with so1:
            st.session_state.spread_odds_home = st.number_input(
                "Home Spread Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.spread_odds_home)
            )
        with so2:
            st.session_state.spread_odds_away = st.number_input(
                "Away Spread Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.spread_odds_away)
            )

        # Row 4: Total + O/U odds + stake
        t_row1, t_row2 = st.columns(2)
        with t_row1:
            st.session_state.total_line = st.number_input("Total Line (e.g., 218.5)", step=0.01, format="%.2f", value=float(st.session_state.total_line))
            st.session_state.over_odds = st.number_input("Over Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.over_odds))
        with t_row2:
            st.session_state.stake = st.number_input("Stake ($)", min_value=0.0, step=1.0, format="%.2f", value=float(st.session_state.stake))
            st.session_state.under_odds = st.number_input("Under Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.under_odds))

        # ---------- Advanced (per sport) ----------
        with st.expander("⚙️ Advanced adjustments (optional)", expanded=False):
            # Universal
            st.markdown("**Universal**")
            u1, u2, u3 = st.columns(3)
            with u1:
                home_edge_pts = st.number_input("Home edge (pts)", value=0.0, step=0.25, format="%.2f")
                form_H_pct = st.number_input("Home form (±% PF)", value=0.0, step=1.0, format="%.0f")
                injury_H_pct = st.number_input("Home injuries (±% PF)", value=0.0, step=1.0, format="%.0f")
            with u2:
                away_edge_pts = st.number_input("Away edge (pts)", value=0.0, step=0.25, format="%.2f")
                form_A_pct = st.number_input("Away form (±% PF)", value=0.0, step=1.0, format="%.0f")
                injury_A_pct = st.number_input("Away injuries (±% PF)", value=0.0, step=1.0, format="%.0f")
            with u3:
                pace_pct_global = st.number_input("Global pace (±% total)", value=0.0, step=1.0, format="%.0f")
                variance_pct = st.number_input("Volatility tweak (±% SD, manual override)", value=0.0, step=5.0, format="%.0f")

            # Football
            if sport in ["NFL", "NCAA Football"]:
                st.markdown("**Football specifics**")
                f1, f2, f3 = st.columns(3)
                with f1:
                    plays_pct = st.number_input("Plays/pace (±% total)", value=0.0, step=1.0, format="%.0f")
                    to_margin_pts = st.number_input("Turnover margin (pts to Home)", value=0.0, step=0.5, format="%.2f")
                with f2:
                    redzone_H_pct = st.number_input("Home red zone (±% PF)", value=0.0, step=1.0, format="%.0f")
                with f3:
                    redzone_A_pct = st.number_input("Away red zone (±% PF)", value=0.0, step=1.0, format="%.0f")
            else:
                plays_pct = redzone_H_pct = redzone_A_pct = to_margin_pts = 0.0

            # Basketball
            if sport in ["NBA", "NCAA Basketball"]:
                st.markdown("**Basketball specifics**")
                b1, b2, b3 = st.columns(3)
                with b1:
                    pace_pct_hoops = st.number_input("Pace (±% total)", value=0.0, step=1.0, format="%.0f")
                    rest_H_pct = st.number_input("Home rest/fatigue (±% PF)", value=0.0, step=1.0, format="%.0f")
                with b2:
                    ortg_H_pct = st.number_input("Home ORtg (±% PF)", value=0.0, step=1.0, format="%.0f")
                    rest_A_pct = st.number_input("Away rest/fatigue (±% PF)", value=0.0, step=1.0, format="%.0f")
                with b3:
                    ortg_A_pct = st.number_input("Away ORtg (±% PF)", value=0.0, step=1.0, format="%.0f")
                    drtg_H_pct = st.number_input("Home DRtg (±% opp PF)", value=0.0, step=1.0, format="%.0f")
                drtg_A_pct = st.number_input("Away DRtg (±% opp PF)", value=0.0, step=1.0, format="%.0f")
            else:
                pace_pct_hoops = ortg_H_pct = ortg_A_pct = drtg_H_pct = drtg_A_pct = rest_H_pct = rest_A_pct = 0.0

                       # MLB
            if sport == "MLB":
                st.markdown("**MLB specifics**")
                m1, m2, m3 = st.columns(3)
                with m1:
                    sp_H_runs = st.number_input("SP impact (Home, runs)", value=0.0, step=0.1, format="%.1f")
                    bullpen_H_runs = st.number_input("Bullpen (Home, runs)", value=0.0, step=0.1, format="%.1f")
                with m2:
                    sp_A_runs = st.number_input("SP impact (Away, runs)", value=0.0, step=0.1, format="%.1f")
                    bullpen_A_runs = st.number_input("Bullpen (Away, runs)", value=0.0, step=0.1, format="%.1f")
                with m3:
                    park_total_pct = st.number_input("Park factor (±% total)", value=0.0, step=1.0, format="%.0f")
                    weather_total_pct = st.number_input("Weather (±% total)", value=0.0, step=1.0, format="%.0f")
            else:
                sp_H_runs = sp_A_runs = bullpen_H_runs = bullpen_A_runs = 0.0
                park_total_pct = weather_total_pct = 0.0






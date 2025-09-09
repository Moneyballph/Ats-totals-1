# Moneyball Phil — ATS & Totals App (Final Stable v3.3 — Spread Fix + Full Reset)
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
        "spread_line_home": 0.0,
        "spread_odds_home": -110.0,
        "spread_odds_away": -110.0,
        "total_line": 0.0,
        "over_odds": -110.0, "under_odds": -110.0,
        "stake": 0.0,
        # results / selection
        "selected_bet": None,
        "results_df": None,
        "proj_total": None,
        "proj_margin": None,
        "proj_home_pts": None,
        "proj_away_pts": None,
        # advanced adjustments state so Reset can clear them
        "auto_volatility": True,
        "pace_pct_global": 0.0,
        "variance_pct_manual": 0.0,
        # universal
        "home_edge_pts": 0.0, "away_edge_pts": 0.0,
        "form_H_pct": 0.0, "form_A_pct": 0.0,
        "injury_H_pct": 0.0, "injury_A_pct": 0.0,
        # football
        "plays_pct": 0.0, "redzone_H_pct": 0.0, "redzone_A_pct": 0.0, "to_margin_pts": 0.0,
        # basketball
        "pace_pct_hoops": 0.0, "ortg_H_pct": 0.0, "ortg_A_pct": 0.0,
        "drtg_H_pct": 0.0, "drtg_A_pct": 0.0, "rest_H_pct": 0.0, "rest_A_pct": 0.0,
        # mlb
        "sp_H_runs": 0.0, "sp_A_runs": 0.0, "bullpen_H_runs": 0.0, "bullpen_A_runs": 0.0,
        "park_total_pct": 0.0, "weather_total_pct": 0.0,
        # for display
        "auto_vol_used": None, "auto_vol_mode": False,
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

def calculate_ev_pct(true_prob_pct: float, odds: float):
    implied = american_to_implied(odds) * 100
    return (true_prob_pct - implied), implied

def tier_by_true_prob(true_prob_pct: float):
    if true_prob_pct >= 80: return "Elite", "#16a34a"     # green
    if true_prob_pct >= 65: return "Strong", "#2563eb"    # blue
    if true_prob_pct >= 50: return "Moderate", "#f59e0b"  # amber
    return "Risky", "#dc2626"                              # red

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
    # Standard normal CDF for (X <= x)
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_sport_sigmas(sport: str):
    # (sd_total, sd_margin)
    if sport == "MLB": return 3.5, 3.0
    if sport == "NFL": return 10.0, 9.0
    if sport == "NCAA Football": return 12.0, 10.0
    if sport == "NBA": return 15.0, 12.0
    if sport == "NCAA Basketball": return 18.0, 14.0
    return 12.0, 10.0

def suggested_volatility(sport: str) -> float:
    # Default SD tweak by sport (percent on SD)
    mapping = {
        "NFL": 10.0,
        "NCAA Football": 12.0,
        "NBA": 12.0,
        "NCAA Basketball": 15.0,
        "MLB": 8.0,
    }
    return mapping.get(sport, 10.0)

# ---------------- Baseline Model ----------------
def project_scores_base(sport: str, H_pf: float, H_pa: float, A_pf: float, A_pa: float):
    # Simple symmetric average model
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
    # football
    plays_pct: float = 0.0, redzone_H_pct: float = 0.0, redzone_A_pct: float = 0.0,
    to_margin_pts: float = 0.0,
    # basketball
    pace_pct_hoops: float = 0.0, ortg_H_pct: float = 0.0, ortg_A_pct: float = 0.0,
    drtg_H_pct: float = 0.0, drtg_A_pct: float = 0.0, rest_H_pct: float = 0.0, rest_A_pct: float = 0.0,
    # mlb
    sp_H_runs: float = 0.0, sp_A_runs: float = 0.0, bullpen_H_runs: float = 0.0, bullpen_A_runs: float = 0.0,
    park_total_pct: float = 0.0, weather_total_pct: float = 0.0
):
    sd_total, sd_margin = get_sport_sigmas(sport)

    # Universal
    H_pts *= (1 + (form_H_pct + injury_H_pct)/100.0)
    A_pts *= (1 + (form_A_pct + injury_A_pct)/100.0)
    H_pts += home_edge_pts
    A_pts += away_edge_pts

    # Football specifics
    if sport in ["NFL", "NCAA Football"]:
        if plays_pct != 0:
            scale = 1 + plays_pct/100.0
            H_pts *= scale
            A_pts *= scale
        H_pts *= (1 + redzone_H_pct/100.0)
        A_pts *= (1 + redzone_A_pct/100.0)
        H_pts += max(0.0, to_margin_pts)
        A_pts += max(0.0, -to_margin_pts)

    # Basketball specifics
    if sport in ["NBA", "NCAA Basketball"]:
        if pace_pct_hoops != 0:
            scale = 1 + pace_pct_hoops/100.0
            H_pts *= scale
            A_pts *= scale
        H_pts *= (1 + (ortg_H_pct - drtg_A_pct)/100.0)
        A_pts *= (1 + (ortg_A_pct - drtg_H_pct)/100.0)
        H_pts *= (1 + rest_H_pct/100.0)
        A_pts *= (1 + rest_A_pct/100.0)

    # MLB specifics
    if sport == "MLB":
        H_pts += sp_H_runs + bullpen_H_runs
        A_pts += sp_A_runs + bullpen_A_runs
        total_scale = (1 + park_total_pct/100.0) * (1 + weather_total_pct/100.0)
        H_pts *= total_scale
        A_pts *= total_scale

    # Global pace affects both
    if pace_pct_global != 0:
        scale = 1 + pace_pct_global/100.0
        H_pts *= scale
        A_pts *= scale

    # Variance tweak (applies to SD only)
    sd_total *= (1 + variance_pct/100.0)
    sd_margin *= (1 + variance_pct/100.0)

    return H_pts, A_pts, sd_total, sd_margin

# ---------------- Sport Selector ----------------
sport = st.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NCAA Football", "NCAA Basketball"])
# On sport change, clear inputs (keep parlay + previous results)
if st.session_state.get("last_sport") != sport:
    for key in ["home","away","home_pf","home_pa","away_pf","away_pa",
                "spread_line_home","spread_odds_home","spread_odds_away",
                "total_line","over_odds","under_odds","stake",
                "home_edge_pts","away_edge_pts","form_H_pct","form_A_pct",
                "injury_H_pct","injury_A_pct","pace_pct_global","variance_pct_manual",
                "plays_pct","redzone_H_pct","redzone_A_pct","to_margin_pts",
                "pace_pct_hoops","ortg_H_pct","ortg_A_pct","drtg_H_pct","drtg_A_pct",
                "rest_H_pct","rest_A_pct",
                "sp_H_runs","sp_A_runs","bullpen_H_runs","bullpen_A_runs",
                "park_total_pct","weather_total_pct"]:
        st.session_state[key] = "" if isinstance(st.session_state[key], str) and key in ["home","away"] else 0.0
    st.session_state["selected_bet"] = None
    st.session_state["last_sport"] = sport

# ---------------- Layout ----------------
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.header("📥 Inputs")

    reset_clicked = False
    run_projection = False

    with st.form("inputs_form", clear_on_submit=False):
        # Teams
        n1, n2 = st.columns(2)
        with n1:
            st.session_state.home = st.text_input("Home Team", value=st.session_state.home)
        with n2:
            st.session_state.away = st.text_input("Away Team", value=st.session_state.away)

        # Averages
        h_col, a_col = st.columns(2)
        with h_col:
            st.caption("**Home Averages**")
            st.session_state.home_pf = st.number_input("Home: Avg Scored", step=0.01, format="%.2f", value=float(st.session_state.home_pf))
            st.session_state.home_pa = st.number_input("Home: Avg Allowed", step=0.01, format="%.2f", value=float(st.session_state.home_pa))
        with a_col:
            st.caption("**Away Averages**")
            st.session_state.away_pf = st.number_input("Away: Avg Scored", step=0.01, format="%.2f", value=float(st.session_state.away_pf))
            st.session_state.away_pa = st.number_input("Away: Avg Allowed", step=0.01, format="%.2f", value=float(st.session_state.away_pa))

        # Spread & odds
        s1, s2 = st.columns(2)
        with s1:
            st.session_state.spread_line_home = st.number_input(
                "Home Spread (enter negative if favorite)", step=0.01, format="%.2f",
                value=float(st.session_state.spread_line_home)
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

        # Totals & stake
        t_row1, t_row2 = st.columns(2)
        with t_row1:
            st.session_state.total_line = st.number_input("Total Line (e.g., 218.5)", step=0.01, format="%.2f", value=float(st.session_state.total_line))
            st.session_state.over_odds = st.number_input("Over Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.over_odds))
        with t_row2:
            st.session_state.stake = st.number_input("Stake ($)", min_value=0.0, step=1.0, format="%.2f", value=float(st.session_state.stake))
            st.session_state.under_odds = st.number_input("Under Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.under_odds))

        # Advanced adjustments
        with st.expander("⚙️ Advanced adjustments (optional)", expanded=False):
            st.markdown("**Universal**")
            u1, u2, u3 = st.columns(3)
            with u1:
                st.session_state.home_edge_pts = st.number_input("Home edge (pts)", value=float(st.session_state.home_edge_pts), step=0.25, format="%.2f")
                st.session_state.form_H_pct = st.number_input("Home form (±% PF)", value=float(st.session_state.form_H_pct), step=1.0, format="%.0f")
                st.session_state.injury_H_pct = st.number_input("Home injuries (±% PF)", value=float(st.session_state.injury_H_pct), step=1.0, format="%.0f")
            with u2:
                st.session_state.away_edge_pts = st.number_input("Away edge (pts)", value=float(st.session_state.away_edge_pts), step=0.25, format="%.2f")
                st.session_state.form_A_pct = st.number_input("Away form (±% PF)", value=float(st.session_state.form_A_pct), step=1.0, format="%.0f")
                st.session_state.injury_A_pct = st.number_input("Away injuries (±% PF)", value=float(st.session_state.injury_A_pct), step=1.0, format="%.0f")
            with u3:
                st.session_state.auto_volatility = st.checkbox("Auto volatility by sport", value=bool(st.session_state.auto_volatility))
                st.session_state.pace_pct_global = st.number_input("Global pace (±% total)", value=float(st.session_state.pace_pct_global), step=1.0, format="%.0f")
                st.session_state.variance_pct_manual = st.number_input("Volatility tweak (±% SD)", value=float(st.session_state.variance_pct_manual), step=5.0, format="%.0f",
                                                      help="Ignored when 'Auto volatility by sport' is ON")

            # Football
            if sport in ["NFL", "NCAA Football"]:
                st.markdown("**Football specifics**")
                f1, f2, f3 = st.columns(3)
                with f1:
                    st.session_state.plays_pct = st.number_input("Plays/pace (±% total)", value=float(st.session_state.plays_pct), step=1.0, format="%.0f")
                    st.session_state.to_margin_pts = st.number_input("Turnover margin (pts to Home)", value=float(st.session_state.to_margin_pts), step=0.5, format="%.2f")
                with f2:
                    st.session_state.redzone_H_pct = st.number_input("Home red zone (±% PF)", value=float(st.session_state.redzone_H_pct), step=1.0, format="%.0f")
                with f3:
                    st.session_state.redzone_A_pct = st.number_input("Away red zone (±% PF)", value=float(st.session_state.redzone_A_pct), step=1.0, format="%.0f")
            else:
                st.session_state.plays_pct = 0.0
                st.session_state.redzone_H_pct = 0.0
                st.session_state.redzone_A_pct = 0.0
                st.session_state.to_margin_pts = 0.0

            # Basketball
            if sport in ["NBA", "NCAA Basketball"]:
                st.markdown("**Basketball specifics**")
                b1, b2, b3 = st.columns(3)
                with b1:
                    st.session_state.pace_pct_hoops = st.number_input("Pace (±% total)", value=float(st.session_state.pace_pct_hoops), step=1.0, format="%.0f")
                    st.session_state.rest_H_pct = st.number_input("Home rest/fatigue (±% PF)", value=float(st.session_state.rest_H_pct), step=1.0, format="%.0f")
                with b2:
                    st.session_state.ortg_H_pct = st.number_input("Home ORtg (±% PF)", value=float(st.session_state.ortg_H_pct), step=1.0, format="%.0f")
                    st.session_state.rest_A_pct = st.number_input("Away rest/fatigue (±% PF)", value=float(st.session_state.rest_A_pct), step=1.0, format="%.0f")
                with b3:
                    st.session_state.ortg_A_pct = st.number_input("Away ORtg (±% PF)", value=float(st.session_state.ortg_A_pct), step=1.0, format="%.0f")
                    st.session_state.drtg_H_pct = st.number_input("Home DRtg (±% opp PF)", value=float(st.session_state.drtg_H_pct), step=1.0, format="%.0f")
                st.session_state.drtg_A_pct = st.number_input("Away DRtg (±% opp PF)", value=float(st.session_state.drtg_A_pct), step=1.0, format="%.0f")
            else:
                st.session_state.pace_pct_hoops = 0.0
                st.session_state.ortg_H_pct = 0.0
                st.session_state.ortg_A_pct = 0.0
                st.session_state.drtg_H_pct = 0.0
                st.session_state.drtg_A_pct = 0.0
                st.session_state.rest_H_pct = 0.0
                st.session_state.rest_A_pct = 0.0

            # MLB
            if sport == "MLB":
                st.markdown("**MLB specifics**")
                m1, m2, m3 = st.columns(3)
                with m1:
                    st.session_state.sp_H_runs = st.number_input("SP impact (Home, runs)", value=float(st.session_state.sp_H_runs), step=0.1, format="%.1f")
                    st.session_state.bullpen_H_runs = st.number_input("Bullpen (Home, runs)", value=float(st.session_state.bullpen_H_runs), step=0.1, format="%.1f")
                with m2:
                    st.session_state.sp_A_runs = st.number_input("SP impact (Away, runs)", value=float(st.session_state.sp_A_runs), step=0.1, format="%.1f")
                    st.session_state.bullpen_A_runs = st.number_input("Bullpen (Away, runs)", value=float(st.session_state.bullpen_A_runs), step=0.1, format="%.1f")
                with m3:
                    st.session_state.park_total_pct = st.number_input("Park factor (±% total)", value=float(st.session_state.park_total_pct), step=1.0, format="%.0f")
                    st.session_state.weather_total_pct = st.number_input("Weather (±% total)", value=float(st.session_state.weather_total_pct), step=1.0, format="%.0f")
            else:
                st.session_state.sp_H_runs = 0.0
                st.session_state.sp_A_runs = 0.0
                st.session_state.bullpen_H_runs = 0.0
                st.session_state.bullpen_A_runs = 0.0
                st.session_state.park_total_pct = 0.0
                st.session_state.weather_total_pct = 0.0

        # Form actions
        cbtn1, cbtn2 = st.columns(2)
        with cbtn1:
            reset_clicked = st.form_submit_button("Reset Inputs")
        with cbtn2:
            run_projection = st.form_submit_button("🔮 Run Projection")

    # Reset after the form (clears EVERYTHING)
    if reset_clicked:
        reset_keys = [
            # Teams & averages
            "home","away","home_pf","home_pa","away_pf","away_pa",
            # Spread / odds
            "spread_line_home","spread_odds_home","spread_odds_away",
            # Totals / odds / stake
            "total_line","over_odds","under_odds","stake",
            # Universal adjustments
            "home_edge_pts","away_edge_pts","form_H_pct","form_A_pct",
            "injury_H_pct","injury_A_pct","pace_pct_global","variance_pct_manual",
            # Football
            "plays_pct","redzone_H_pct","redzone_A_pct","to_margin_pts",
            # Basketball
            "pace_pct_hoops","ortg_H_pct","ortg_A_pct","drtg_H_pct","drtg_A_pct",
            "rest_H_pct","rest_A_pct",
            # MLB
            "sp_H_runs","sp_A_runs","bullpen_H_runs","bullpen_A_runs",
            "park_total_pct","weather_total_pct"
        ]
        for key in reset_keys:
            if key in st.session_state:
                st.session_state[key] = "" if key in ["home","away"] else 0.0
        st.session_state.selected_bet = None
        st.stop()

with col_results:
    st.header("📊 Results")

    # Run projection
    if run_projection:
        S = st.session_state

        # Baseline model
        home_pts, away_pts = project_scores_base(sport, S.home_pf, S.home_pa, S.away_pf, S.away_pa)

        # Resolve effective volatility
        auto_vol_used = suggested_volatility(sport) if S.auto_volatility else float(S.variance_pct_manual)

        # Apply adjustments
        home_pts, away_pts, sd_total, sd_margin = apply_adjustments(
            sport, home_pts, away_pts,
            # universal
            S.home_edge_pts, S.away_edge_pts, S.form_H_pct, S.form_A_pct,
            S.injury_H_pct, S.injury_A_pct, S.pace_pct_global, auto_vol_used,
            # football
            S.plays_pct, S.redzone_H_pct, S.redzone_A_pct, S.to_margin_pts,
            # basketball
            S.pace_pct_hoops, S.ortg_H_pct, S.ortg_A_pct, S.drtg_H_pct, S.drtg_A_pct, S.rest_H_pct, S.rest_A_pct,
            # mlb
            S.sp_H_runs, S.sp_A_runs, S.bullpen_H_runs, S.bullpen_A_runs, S.park_total_pct, S.weather_total_pct
        )

        proj_total = home_pts + away_pts
        proj_margin = home_pts - away_pts  # Home - Away

        rows = []

        # ---------------- SPREAD (FIXED) ----------------
        # Interpret "Home spread" line L_home as the number added to Home (negative if favorite).
        # If L_home = -1.5, then Home covers when Margin >= +1.5  --> threshold_home = -L_home
        L_home = float(S.spread_line_home)
        threshold_home = -L_home

        # Home spread probability: P(Margin >= threshold_home)
        z_home = (proj_margin - threshold_home) / sd_margin if sd_margin > 0 else 0.0
        true_home = _std_norm_cdf(z_home) * 100.0
        ev_home, impl_home = calculate_ev_pct(true_home, S.spread_odds_home)
        tier_home, _ = tier_by_true_prob(true_home)
        rows.append([f"{S.home} {S.spread_line_home:+.2f}", S.spread_odds_home, true_home, impl_home, ev_home, tier_home])

        # Away spread is the opposite line (+/-), covering when Margin <= L_home
        z_away = (L_home - proj_margin) / sd_margin if sd_margin > 0 else 0.0
        true_away = _std_norm_cdf(z_away) * 100.0
        ev_away, impl_away = calculate_ev_pct(true_away, S.spread_odds_away)
        tier_away, _ = tier_by_true_prob(true_away)
        rows.append([f"{S.away} {(-S.spread_line_home):+.2f}", S.spread_odds_away, true_away, impl_away, ev_away, tier_away])
        # ------------------------------------------------

        # Totals
        z_total_over = (proj_total - S.total_line) / sd_total if sd_total > 0 else 0.0
        true_over = _std_norm_cdf(z_total_over) * 100.0
        ev_over, impl_over = calculate_ev_pct(true_over, S.over_odds)
        tier_over, _ = tier_by_true_prob(true_over)
        rows.append([f"Over {S.total_line:.2f}", S.over_odds, true_over, impl_over, ev_over, tier_over])

        true_under = max(0.0, 100.0 - true_over)
        ev_under, impl_under = calculate_ev_pct(true_under, S.under_odds)
        tier_under, _ = tier_by_true_prob(true_under)
        rows.append([f"Under {S.total_line:.2f}", S.under_odds, true_under, impl_under, ev_under, tier_under])

        df = pd.DataFrame(rows, columns=["Bet Type", "Odds", "True %", "Implied %", "EV %", "Tier"])
        st.session_state.results_df = df
        st.session_state.proj_total = proj_total
        st.session_state.proj_margin = proj_margin
        st.session_state.proj_home_pts = home_pts
        st.session_state.proj_away_pts = away_pts
        st.session_state.auto_vol_used = auto_vol_used
        st.session_state.auto_vol_mode = bool(S.auto_volatility)

    # Show results if available
    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        proj_total = st.session_state.proj_total
        proj_margin = st.session_state.proj_margin
        home_pts = st.session_state.proj_home_pts
        away_pts = st.session_state.proj_away_pts
        auto_vol_used = st.session_state.get("auto_vol_used", None)
        auto_vol_mode = st.session_state.get("auto_vol_mode", False)
        S = st.session_state

        # Projection summary (favorite label fixed)
        st.subheader("Projected Game Outcome")
        st.write(f"**{S.home} (Home)**: {home_pts:.1f} — **{S.away} (Away)**: {away_pts:.1f}")
        fav_team = S.home if proj_margin > 0 else S.away
        st.write(f"**Projected Spread**: {fav_team} -{abs(proj_margin):.1f} | **Projected Total**: {proj_total:.1f}")
        if auto_vol_used is not None:
            mode_txt = "Auto" if auto_vol_mode else "Manual"
            st.caption(f"Volatility applied: **{mode_txt} {auto_vol_used:.0f}%** (affects standard deviation only)")

        # Results table
        st.subheader("Bet Results")
        st.dataframe(df, use_container_width=True)

        # Keep a stable selection
        if st.session_state.selected_bet is None and len(df) > 0:
            st.session_state.selected






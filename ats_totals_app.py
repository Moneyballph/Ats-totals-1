# Moneyball Phil — ATS & Totals App (v2.4 Fixed Layout, Two-Column Inputs, Correct Spread Label)
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
        "total_line": 0.0, "over_odds": -110.0, "under_odds": -110.0,
        "stake": 0.0,
        # results
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
    return (
        f"<span style='background:{color};color:white;"
        f"padding:4px 10px;border-radius:999px;font-weight:700;"
        f"display:inline-block'>{tier_name}</span>"
    )

def simple_bar(label: str, pct: float):
    pct = max(0.0, min(100.0, float(pct)))
    st.markdown(
        f"""
        <div style="margin:6px 0 4px 0;">
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
def project_scores_base(sport: str, H_pf: float, H_pa: float, A_pf: float, A_pa: float) -> tuple[float, float]:
    home_pts = (H_pf + A_pa) / 2.0
    away_pts = (A_pf + H_pa) / 2.0
    return home_pts, away_pts

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

    # Universal
    H_pts *= (1 + (form_H_pct + injury_H_pct)/100.0)
    A_pts *= (1 + (form_A_pct + injury_A_pct)/100.0)
    H_pts += home_edge_pts
    A_pts += away_edge_pts

    # Global pace
    if pace_pct_global != 0:
        scale = 1 + pace_pct_global/100.0
        total_before = H_pts + A_pts
        if total_before > 0:
            H_pts *= scale
            A_pts *= scale

    # Variance tweak
    sd_total *= (1 + variance_pct/100.0)
    sd_margin *= (1 + variance_pct/100.0)

    return H_pts, A_pts, sd_total, sd_margin

# ---------------- Sport Selector ----------------
sport = st.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NCAA Football", "NCAA Basketball"])

# ---------------- Layout ----------------
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.header("📥 Inputs")

    with st.form("inputs_form", clear_on_submit=False):
        # Row 1: Teams
        n1, n2 = st.columns(2)
        with n1: st.session_state.home = st.text_input("Home Team", value=st.session_state.home)
        with n2: st.session_state.away = st.text_input("Away Team", value=st.session_state.away)

        # Row 2: Averages
        h_col, a_col = st.columns(2)
        with h_col:
            st.caption("**Home Averages**")
            st.session_state.home_pf = st.number_input("Home: Avg Scored", value=float(st.session_state.home_pf))
            st.session_state.home_pa = st.number_input("Home: Avg Allowed", value=float(st.session_state.home_pa))
        with a_col:
            st.caption("**Away Averages**")
            st.session_state.away_pf = st.number_input("Away: Avg Scored", value=float(st.session_state.away_pf))
            st.session_state.away_pa = st.number_input("Away: Avg Allowed", value=float(st.session_state.away_pa))

        # Row 3: Spread + odds
        s1, s2 = st.columns(2)
        with s1:
            st.session_state.spread_line_home = st.number_input("Home Spread (enter negative if favorite)", value=float(st.session_state.spread_line_home))
        with s2:
            st.caption(f"Away Spread (auto): {(-st.session_state.spread_line_home):+.2f}")

        so1, so2 = st.columns(2)
        with so1:
            st.session_state.spread_odds_home = st.number_input("Home Spread Odds (American)", value=float(st.session_state.spread_odds_home))
        with so2:
            st.session_state.spread_odds_away = st.number_input("Away Spread Odds (American)", value=float(st.session_state.spread_odds_away))

        # Row 4: Totals
        t1, t2 = st.columns(2)
        with t1:
            st.session_state.total_line = st.number_input("Total Line", value=float(st.session_state.total_line))
            st.session_state.over_odds = st.number_input("Over Odds (American)", value=float(st.session_state.over_odds))
        with t2:
            st.session_state.stake = st.number_input("Stake ($)", value=float(st.session_state.stake))
            st.session_state.under_odds = st.number_input("Under Odds (American)", value=float(st.session_state.under_odds))

        # Advanced adjustments
        with st.expander("⚙️ Advanced adjustments (optional)", expanded=False):
            home_edge_pts = st.number_input("Home edge (pts)", value=0.0)
            away_edge_pts = st.number_input("Away edge (pts)", value=0.0)
            form_H_pct = st.number_input("Home form (±% PF)", value=0.0)
            form_A_pct = st.number_input("Away form (±% PF)", value=0.0)
            injury_H_pct = st.number_input("Home injuries (±% PF)", value=0.0)
            injury_A_pct = st.number_input("Away injuries (±% PF)", value=0.0)
            pace_pct_global = st.number_input("Global pace (±% total)", value=0.0)
            variance_pct = st.number_input("Volatility tweak (±% SD)", value=0.0)

        # Submit buttons
        run_projection = st.form_submit_button("🔮 Run Projection")
        reset_clicked = st.form_submit_button("Reset Inputs")

    if reset_clicked:
        for key in ["home","away","home_pf","home_pa","away_pf","away_pa",
                    "spread_line_home","spread_odds_home","spread_odds_away",
                    "total_line","over_odds","under_odds","stake"]:
            st.session_state[key] = 0.0 if not isinstance(st.session_state[key], str) else ""
        st.session_state.selected_bet = None
        st.stop()

with col_results:
    st.header("📊 Results")

    if run_projection:
        S = st.session_state
        home_pts, away_pts = project_scores_base(sport, S.home_pf, S.home_pa, S.away_pf, S.away_pa)
        home_pts, away_pts, sd_total, sd_margin = apply_adjustments(
            sport, home_pts, away_pts,
            home_edge_pts, away_edge_pts, form_H_pct, form_A_pct,
            injury_H_pct, injury_A_pct, pace_pct_global, variance_pct
        )

        proj_total = home_pts + away_pts
        proj_margin = home_pts - away_pts
        fav_team = S.home if proj_margin > 0 else S.away
        spread_label = f"{fav_team} {abs(proj_margin):+.1f}"

        rows = []

        # Spread (home)
        z_spread_home = (proj_margin - S.spread_line_home) / sd_margin if sd_margin > 0 else 0.0
        true_home = _std_norm_cdf(z_spread_home) * 100.0
        ev_home, impl_home = calculate_ev_pct(true_home, S.spread_odds_home)
        tier_home, _ = tier_by_true_prob(true_home)
        rows.append([f"{S.home} {S.spread_line_home:+.2f}", S.spread_odds_home, true_home, impl_home, ev_home, tier_home])

        # Spread (away)
        true_away = max(0.0, 100.0 - true_home)
        ev_away, impl_away = calculate_ev_pct(true_away, S.spread_odds_away)
        tier_away, _ = tier_by_true_prob(true_away)
        rows.append([f"{S.away} {(-S.spread_line_home):+.2f}", S.spread_odds_away, true_away, impl_away, ev_away, tier_away])

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

        df = pd.DataFrame(rows, columns=["Bet Type","Odds","True %","Implied %","EV %","Tier"])
        st.session_state.results_df = df
        st.session_state.proj_total = proj_total
        st.session_state.proj_margin = proj_margin
        st.session_state.proj_home_pts = home_pts
        st.session_state.proj_away_pts = away_pts

    if st.session_state.results_df is not None:
        df = st.session_state.results_df
        proj_total = st.session_state.proj_total
        proj_margin = st.session_state.proj_margin
        home_pts = st.session_state.proj_home_pts
        away_pts = st.session_state.proj_away_pts
        S = st.session_state

        st.subheader("Projected Game Outcome")
        fav_team = S.home if proj_margin > 0 else S.away
        st.write(f"**{S.home} (Home)**: {home_pts:.1f} — **{S.away} (Away)**: {away_pts:.1f}")
        st.write(f"**Projected Spread**: {fav_team} -{abs(proj_margin):.1f} | **Projected Total**: {proj_total:.1f}")

        st.subheader("Bet Results")
        st.dataframe(df, use_container_width=True)

        if st.session_state.selected_bet is None and len(df) > 0:
            st.session_state.selected_bet = df["Bet Type"].iloc[0]

        choice = st.selectbox("Select a bet to save/add to slip:", options=list(df["Bet Type"]), key="selected_bet")
        selected = df[df["Bet Type"] == st.session_state.selected_bet].iloc[0]

        st.markdown("### Bet Details")
        tier_name, tier_color = tier_by_true_prob(float(selected["True %"]))
        st.markdown(tier_badge_html(tier_name, tier_color), unsafe_allow_html=True)
        simple_bar("True Probability", selected["True %"])
        simple_bar("Implied Probability", selected["Implied %"])
        simple_bar("EV%", selected["EV %"])

        if st.button("💾 Save Straight Bet"):
            st.success("✅ Bet saved (copy row below to tracker).")
            st.code(
                f"{datetime.date.today()}, Straight, {selected['Bet Type']}, "
                f"{int(selected['Odds'])}, {selected['True %']:.1f}%, {selected['Implied %']:.1f}%, "
                f"{S.stake:.2f}, W/L, +/-$, {selected['EV %']:.1f}%, Cumulative, ROI%"
            )





# Moneyball Phil — ATS & Totals App (v2.1 Home/Away)
# Two-column compact inputs + sport-specific advanced panels + probability engine
# Sports: MLB, NFL, NBA, NCAA Football, NCAA Basketball

import streamlit as st
import pandas as pd
import datetime
import matplotlib.pyplot as plt
import math

# ============== Page Setup ==============
st.set_page_config(page_title="Moneyball Phil — ATS & Totals", layout="wide")
st.title("🏆 Moneyball Phil — ATS & Totals App")

# ------------- Session State -------------
def init_state():
    defaults = {
        "parlay_slip": [],
        "last_sport": None,
        # inputs
        "home": "", "away": "",
        "home_pf": 0.0, "home_pa": 0.0,
        "away_pf": 0.0, "away_pa": 0.0,
        "spread_line_home": 0.0,     # enter home side; away is auto-mirror
        "spread_odds_home": -110.0,
        "spread_odds_away": -110.0,
        "total_line": 0.0, "over_odds": -110.0, "under_odds": -110.0,
        "stake": 0.0
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ============== Utility Functions ==============
def american_to_implied(odds: float) -> float:
    """Return implied probability as a fraction (0-1)."""
    return (100 / (odds + 100)) if odds > 0 else (abs(odds) / (abs(odds) + 100))

def american_to_decimal(odds: float) -> float:
    return 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))

def calculate_ev_pct(true_prob_pct: float, odds: float) -> tuple[float, float]:
    """Return (EV%, implied_prob_pct). true_prob_pct as 0-100."""
    implied = american_to_implied(odds) * 100
    ev_pct = true_prob_pct - implied
    return ev_pct, implied

def tier_by_true_prob(true_prob_pct: float) -> tuple[str, str]:
    """Tier name + color based on TRUE probability (not EV)."""
    if true_prob_pct >= 80:
        return "Elite", "green"
    if true_prob_pct >= 65:
        return "Strong", "blue"
    if true_prob_pct >= 50:
        return "Moderate", "orange"
    return "Risky", "red"

def vertical_band_chart_for_tier(true_prob_pct: float):
    """Draw tier bands only (no marker) and render."""
    labels = ["Elite (≥80%)", "Strong (65–79%)", "Moderate (50–64%)", "Risky (<50%)"]
    colors = ["green", "blue", "orange", "red"]
    fig, ax = plt.subplots(figsize=(2.0, 3.6))
    ax.barh(labels[::-1], [1]*4, color=colors[::-1], alpha=0.25)  # reverse so Elite at top
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlim(0, 1)
    ax.set_frame_on(False)
    st.pyplot(fig, use_container_width=False)

# ===== Normal CDF + Sport Volatility (drives sport-specific probabilities) =====
def _std_norm_cdf(x: float) -> float:
    """Φ(x) via erf; returns probability in [0,1]."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def get_sport_sigmas(sport: str) -> tuple[float, float]:
    """Return (sd_total, sd_margin) tuned per sport (can refine later)."""
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

# ============== Scoring Model (baseline from PF/PA) ==============
def project_scores_base(sport: str, H_pf: float, H_pa: float, A_pf: float, A_pa: float) -> tuple[float, float]:
    """Return baseline projected points (home_pts, away_pts) from PF/PA inputs."""
    home_pts = (H_pf + A_pa) / 2.0
    away_pts = (A_pf + H_pa) / 2.0
    return home_pts, away_pts

# ============== Sport-Specific Adjustments ==============
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
    """
    Returns adjusted (H_pts, A_pts) and (sd_total, sd_margin) possibly tweaked by variance_pct.
    Adjustments applied as % shifts to means or additive points; then global pace/variance.
    """
    sd_total, sd_margin = get_sport_sigmas(sport)

    # ----- Universal mean tweaks -----
    H_pts *= (1 + (form_H_pct + injury_H_pct)/100.0)
    A_pts *= (1 + (form_A_pct + injury_A_pct)/100.0)

    H_pts += home_edge_pts
    A_pts += away_edge_pts

    # ----- Sport-specific tweaks -----
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
        # turnover margin tilt (positive favors Home)
        H_pts += max(0.0, to_margin_pts)
        A_pts += max(0.0, -to_margin_pts)

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

    if sport == "MLB":
        H_pts += sp_H_runs + bullpen_H_runs
        A_pts += sp_A_runs + bullpen_A_runs
        total_scale = (1 + park_total_pct/100.0) * (1 + weather_total_pct/100.0)
        total_before = H_pts + A_pts
        if total_before > 0 and total_scale != 1.0:
            factor = (total_before * total_scale) / total_before
            H_pts *= factor
            A_pts *= factor

    # Global pace (% to total)
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

# ============== Sport Selector (auto-clear inputs on change) ==============
sport = st.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NCAA Football", "NCAA Basketball"])
if st.session_state.get("last_sport") != sport:
    for key in ["home","away","home_pf","home_pa","away_pf","away_pa",
                "spread_line_home","spread_odds_home","spread_odds_away",
                "total_line","over_odds","under_odds","stake"]:
        st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
    st.session_state["last_sport"] = sport

# ============== Two-Column Layout ==============
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.header("📥 Inputs")

    # Row 1: Team names side-by-side
    n1, n2 = st.columns(2)
    with n1:
        st.session_state.home = st.text_input("Home Team", value=st.session_state.home)
    with n2:
        st.session_state.away = st.text_input("Away Team", value=st.session_state.away)

    # Row 2: PF/PA — home & away side-by-side
    h_col, a_col = st.columns(2)
    with h_col:
        st.caption("**Home Averages**")
        st.session_state.home_pf = st.number_input("Home: Avg Scored", step=0.01, format="%.2f", value=float(st.session_state.home_pf))
        st.session_state.home_pa = st.number_input("Home: Avg Allowed", step=0.01, format="%.2f", value=float(st.session_state.home_pa))
    with a_col:
        st.caption("**Away Averages**")
        st.session_state.away_pf = st.number_input("Away: Avg Scored", step=0.01, format="%.2f", value=float(st.session_state.away_pf))
        st.session_state.away_pa = st.number_input("Away: Avg Allowed", step=0.01, format="%.2f", value=float(st.session_state.away_pa))

    # Row 3: Spread line (home) + mirrored preview (away) + odds per side
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
            "Home Spread Odds (American)",
            step=1.0, format="%.0f", value=float(st.session_state.spread_odds_home)
        )
    with so2:
        st.session_state.spread_odds_away = st.number_input(
            "Away Spread Odds (American)",
            step=1.0, format="%.0f", value=float(st.session_state.spread_odds_away)
        )

    # Row 4: Total line + Over/Under odds + stake
    t_row1, t_row2 = st.columns(2)
    with t_row1:
        st.session_state.total_line = st.number_input("Total Line (e.g., 218.5)", step=0.01, format="%.2f", value=float(st.session_state.total_line))
        st.session_state.over_odds = st.number_input("Over Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.over_odds))
    with t_row2:
        st.session_state.stake = st.number_input("Stake ($)", min_value=0.0, step=1.0, format="%.2f", value=float(st.session_state.stake))
        st.session_state.under_odds = st.number_input("Under Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.under_odds))

    # ---------- Advanced Panels ----------
    with st.expander("⚙️ Advanced adjustments (optional)", expanded=False):
        # Universal tweaks
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
            variance_pct = st.number_input("Volatility tweak (±% SD)", value=0.0, step=5.0, format="%.0f")

        # Football specifics
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

        # Basketball specifics
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

        # MLB specifics
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

    # Row 5: Buttons side-by-side
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Reset Inputs"):
            for key in ["home","away","home_pf","home_pa","away_pf","away_pa",
                        "spread_line_home","spread_odds_home","spread_odds_away",
                        "total_line","over_odds","under_odds","stake"]:
                st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
            st.experimental_rerun()
    with b2:
        run_projection = st.button("🔮 Run Projection")

with col_results:
    st.header("📊 Results")

    if 'run_projection' not in locals():
        run_projection = False

    if run_projection:
        S = st.session_state

        # Base scores from PF/PA
        home_pts, away_pts = project_scores_base(
            sport,
            S.home_pf, S.home_pa,
            S.away_pf, S.away_pa
        )

        # Apply all adjustments (sport-specific + universal)
        home_pts, away_pts, sd_total, sd_margin = apply_adjustments(
            sport, home_pts, away_pts,
            # universal
            home_edge_pts, away_edge_pts,
            form_H_pct, form_A_pct,
            injury_H_pct, injury_A_pct,
            pace_pct_global, variance_pct,
            # football
            plays_pct, redzone_H_pct, redzone_A_pct, to_margin_pts,
            # hoops
            pace_pct_hoops, ortg_H_pct, ortg_A_pct, drtg_H_pct, drtg_A_pct, rest_H_pct, rest_A_pct,
            # mlb
            sp_H_runs, sp_A_runs, bullpen_H_runs, bullpen_A_runs, park_total_pct, weather_total_pct
        )

        proj_total = home_pts + away_pts
        proj_margin = home_pts - away_pts  # positive = Home favored

        st.subheader("Projected Game Outcome")
        st.write(f"**{S.home} (Home)**: {home_pts:.1f} — **{S.away} (Away)**: {away_pts:.1f}")
        st.write(f"**Projected Spread**: {S.home} {proj_margin:+.1f}")
        st.write(f"**Projected Total**: {proj_total:.1f}")

        # ---- sport-specific probability model (normal approximations) ----
        rows = []

        # Spread (Home side): P( (Home - Away) - line > 0 )
        z_spread_home = (proj_margin - S.spread_line_home) / sd_margin if sd_margin > 0 else 0.0
        true_home = _std_norm_cdf(z_spread_home) * 100.0
        ev_home, impl_home = calculate_ev_pct(true_home, S.spread_odds_home)
        tier_home, _ = tier_by_true_prob(true_home)
        rows.append([f"{S.home} {S.spread_line_home:+.2f}", S.spread_odds_home, true_home, impl_home, ev_home, tier_home])

        # Spread (Away side) = complement; line is mirrored
        true_away = max(0.0, 100.0 - true_home)
        ev_away, impl_away = calculate_ev_pct(true_away, S.spread_odds_away)
        tier_away, _ = tier_by_true_prob(true_away)
        rows.append([f"{S.away} {(-S.spread_line_home):+.2f}", S.spread_odds_away, true_away, impl_away, ev_away, tier_away])

        # Total Over: P( total - line > 0 )
        z_total_over = (proj_total - S.total_line) / sd_total if sd_total > 0 else 0.0
        true_over = _std_norm_cdf(z_total_over) * 100.0
        ev_over, impl_over = calculate_ev_pct(true_over, S.over_odds)
        tier_over, _ = tier_by_true_prob(true_over)
        rows.append([f"Over {S.total_line:.2f}", S.over_odds, true_over, impl_over, ev_over, tier_over])

        # Under: complement
        true_under = max(0.0, 100.0 - true_over)
        ev_under, impl_under = calculate_ev_pct(true_under, S.under_odds)
        tier_under, _ = tier_by_true_prob(true_under)
        rows.append([f"Under {S.total_line:.2f}", S.under_odds, true_under, impl_under, ev_under, tier_under])

        df = pd.DataFrame(rows, columns=["Bet Type", "Odds", "True %", "Implied %", "EV %", "Tier"])
        st.subheader("Bet Results")
        st.dataframe(df, use_container_width=True)

        # --- Choose a bet to act on ---
        st.markdown("### Actions")
        choice = st.selectbox("Select a bet to save/add to slip:", options=list(df["Bet Type"]))
        selected = df[df["Bet Type"] == choice].iloc[0]

        # Tier row (with color + band chart ONLY here)
        tier_name, tier_color = tier_by_true_prob(float(selected["True %"]))
        c1, c2 = st.columns([1, 1.3])
        with c1:
            st.markdown(f"**Tier:** <span style='color:{tier_color}'>{tier_name}</span>", unsafe_allow_html=True)
            st.write(f"**True Probability:** {selected['True %']:.1f}%")
            st.write(f"**Implied Probability:** {selected['Implied %']:.1f}%")
            st.write(f"**EV%:** {selected['EV %']:.1f}%")
        with c2:
            vertical_band_chart_for_tier(float(selected["True %"]))

        # Toggle: Save Straight vs Send to Parlay
        action = st.radio("What do you want to do with this bet?",
                          ["Save Straight Bet", "Send to Parlay Slip"])

        if action == "Save Straight Bet":
            st.success("✅ Bet saved (copy row below to your tracker).")
            st.code(
                f"{datetime.date.today()}, Straight, {selected['Bet Type']}, "
                f"{int(selected['Odds'])}, {selected['True %']:.1f}%, {selected['Implied %']:.1f}%, "
                f"{st.session_state.stake:.2f}, W/L, +/-$, {selected['EV %']:.1f}%, Cumulative, ROI%"
            )
        else:
            # Store as a simple list for the slip
            st.session_state.parlay_slip.append([
                selected["Bet Type"],
                int(selected["Odds"]),
                float(selected["True %"]),
                float(selected["Implied %"]),
                float(selected["EV %"]),
                tier_name
            ])
            st.success(f"➕ Added to parlay slip: {selected['Bet Type']}")

# ============== Parlay Slip (always at bottom) ==============
st.markdown("---")
st.subheader("🧾 Parlay Slip")

if len(st.session_state.parlay_slip) == 0:
    st.info("No legs yet. Run a projection, select a bet, and choose **Send to Parlay Slip**.")
else:
    slip_df = pd.DataFrame(
        st.session_state.parlay_slip,
        columns=["Bet Type", "Odds", "True %", "Implied %", "EV %", "Tier"]
    )
    st.dataframe(slip_df, use_container_width=True)

    # Combined metrics
    dec_product = 1.0
    true_prod = 1.0
    implied_prod = 1.0
    for _, r in slip_df.iterrows():
        dec = 1 + (r["Odds"]/100) if r["Odds"] > 0 else 1 + (100/abs(r["Odds"]))
        dec_product *= dec
        true_prod *= (r["True %"]/100.0)
        implied_prod *= (r["Implied %"]/100.0)

    combined_true_pct = true_prod * 100.0
    combined_implied_pct = implied_prod * 100.0
    combined_ev_pct = combined_true_pct - combined_implied_pct
    american = (dec_product - 1) * 100 if dec_product >= 2 else -100 / (dec_product - 1)

    # Display combined numbers
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Combined Odds (American)", f"{american:.0f}")
    c2.metric("Combined Odds (Decimal)", f"{dec_product:.2f}")
    c3.metric("True Probability", f"{combined_true_pct:.1f}%")
    c4.metric("EV%", f"{combined_ev_pct:.1f}%")

    # Tier (by TRUE %) + band chart only here
    tier_name, tier_color = tier_by_true_prob(combined_true_pct)
    cA, cB = st.columns([1, 1.3])
    with cA:
        st.markdown(f"**Parlay Tier:** <span style='color:{tier_color}'>{tier_name}</span>", unsafe_allow_html=True)
        st.write(f"**Implied Probability:** {combined_implied_pct:.1f}%")
    with cB:
        vertical_band_chart_for_tier(combined_true_pct)

    # Manage slip
    rm_leg = st.selectbox("Remove a leg (optional):", options=["—"] + list(slip_df["Bet Type"]))
    if rm_leg != "—":
        st.session_state.parlay_slip = [leg for leg in st.session_state.parlay_slip if leg[0] != rm_leg]
        st.experimental_rerun()

    if st.button("Clear Parlay Slip"):
        st.session_state.parlay_slip = []
        st.experimental_rerun()



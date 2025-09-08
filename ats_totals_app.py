# Moneyball Phil — ATS & Totals App (v1.2)
# Two-column layout + sport-specific probability engine (no placeholders)
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
        "teamA": "", "teamB": "",
        "teamA_pf": 0.0, "teamA_pa": 0.0,
        "teamB_pf": 0.0, "teamB_pa": 0.0,
        "spread_line": 0.0, "spread_odds": -110.0,
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
    """
    Return (sd_total, sd_margin) for the sport.
    These control how sharp/wide the distributions are.
    Tune later with historical data.
    """
    if sport == "MLB":
        return 3.5, 3.0       # runs
    if sport == "NFL":
        return 10.0, 9.0      # points
    if sport == "NCAA Football":
        return 12.0, 10.0     # points
    if sport == "NBA":
        return 15.0, 12.0     # points
    if sport == "NCAA Basketball":
        return 18.0, 14.0     # points
    return 12.0, 10.0

# ============== Sport Models (v1 scoring stubs) ==============
def project_scores(sport: str, A_pf: float, A_pa: float, B_pf: float, B_pa: float) -> tuple[float, float]:
    """
    Return projected points (A_pts, B_pts).
    v1 uses symmetric averaging placeholders per sport.
    Swap with MLB Poisson / NFL drives / NBA pace later.
    """
    A_pts = (A_pf + B_pa) / 2.0
    B_pts = (B_pf + A_pa) / 2.0
    return A_pts, B_pts

# ============== Sport Selector (auto-clear inputs on change) ==============
sport = st.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NCAA Football", "NCAA Basketball"])
if st.session_state.get("last_sport") != sport:
    # clear input fields on sport change (keep parlay slip)
    for key in ["teamA","teamB","teamA_pf","teamA_pa","teamB_pf","teamB_pa",
                "spread_line","spread_odds","total_line","over_odds","under_odds","stake"]:
        st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
    st.session_state["last_sport"] = sport

# ============== Two-Column Layout ==============
col_inputs, col_results = st.columns([1, 2])

with col_inputs:
    st.header("📥 Inputs")

    st.session_state.teamA = st.text_input("Team A Name", value=st.session_state.teamA)
    st.session_state.teamB = st.text_input("Team B Name", value=st.session_state.teamB)

    st.markdown("**Team A Averages**")
    st.session_state.teamA_pf = st.number_input("Team A Avg Points Scored", step=0.01, format="%.2f", value=float(st.session_state.teamA_pf))
    st.session_state.teamA_pa = st.number_input("Team A Avg Points Allowed", step=0.01, format="%.2f", value=float(st.session_state.teamA_pa))

    st.markdown("**Team B Averages**")
    st.session_state.teamB_pf = st.number_input("Team B Avg Points Scored", step=0.01, format="%.2f", value=float(st.session_state.teamB_pf))
    st.session_state.teamB_pa = st.number_input("Team B Avg Points Allowed", step=0.01, format="%.2f", value=float(st.session_state.teamB_pa))

    st.markdown("**Sportsbook Spread**")
    st.session_state.spread_line = st.number_input("Spread Line (e.g., -3.5 or +3.5)", step=0.01, format="%.2f", value=float(st.session_state.spread_line))
    st.session_state.spread_odds = st.number_input("Spread Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.spread_odds))

    st.markdown("**Sportsbook Total**")
    st.session_state.total_line = st.number_input("Total Line (e.g., 218.5)", step=0.01, format="%.2f", value=float(st.session_state.total_line))
    st.session_state.over_odds = st.number_input("Over Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.over_odds))
    st.session_state.under_odds = st.number_input("Under Odds (American)", step=1.0, format="%.0f", value=float(st.session_state.under_odds))

    st.session_state.stake = st.number_input("Stake ($)", min_value=0.0, step=1.0, format="%.2f", value=float(st.session_state.stake))

    # Reset button clears inputs (keeps parlay slip)
    if st.button("Reset Inputs"):
        for key in ["teamA","teamB","teamA_pf","teamA_pa","teamB_pf","teamB_pa",
                    "spread_line","spread_odds","total_line","over_odds","under_odds","stake"]:
            st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
        st.experimental_rerun()

    run_projection = st.button("🔮 Run Projection")

with col_results:
    st.header("📊 Results")

    if run_projection:
        A = st.session_state
        # Project scores using the (temporary) averaging model
        A_pts, B_pts = project_scores(
            sport,
            A.teamA_pf, A.teamA_pa,
            A.teamB_pf, A.teamB_pa
        )
        proj_total = A_pts + B_pts
        proj_spread = A_pts - B_pts  # positive = Team A favored

        st.subheader("Projected Game Outcome")
        st.write(f"**{A.teamA}**: {A_pts:.1f} pts — **{A.teamB}**: {B_pts:.1f} pts")
        st.write(f"**Projected Spread**: {A.teamA} {proj_spread:+.1f}")
        st.write(f"**Projected Total**: {proj_total:.1f}")

        # ---- sport-specific probability model (normal approximations) ----
        sd_total, sd_margin = get_sport_sigmas(sport)

        rows = []

        # Spread: P( (A-B) - line > 0 )
        z_spread_A = (proj_spread - A.spread_line) / sd_margin if sd_margin > 0 else 0.0
        true_A = _std_norm_cdf(z_spread_A) * 100.0
        evA, implA = calculate_ev_pct(true_A, A.spread_odds)
        tierA, _ = tier_by_true_prob(true_A)
        rows.append([f"{A.teamA} {A.spread_line:+.2f}", A.spread_odds, true_A, implA, evA, tierA])

        # Opposite side for Team B
        true_B = max(0.0, 100.0 - true_A)
        evB, implB = calculate_ev_pct(true_B, A.spread_odds)
        tierB, _ = tier_by_true_prob(true_B)
        rows.append([f"{A.teamB} {(-A.spread_line):+.2f}", A.spread_odds, true_B, implB, evB, tierB])

        # Total Over: P( total - line > 0 )
        z_total_over = (proj_total - A.total_line) / sd_total if sd_total > 0 else 0.0
        true_O = _std_norm_cdf(z_total_over) * 100.0
        evO, implO = calculate_ev_pct(true_O, A.over_odds)
        tierO, _ = tier_by_true_prob(true_O)
        rows.append([f"Over {A.total_line:.2f}", A.over_odds, true_O, implO, evO, tierO])

        # Under = complement
        true_U = max(0.0, 100.0 - true_O)
        evU, implU = calculate_ev_pct(true_U, A.under_odds)
        tierU, _ = tier_by_true_prob(true_U)
        rows.append([f"Under {A.total_line:.2f}", A.under_odds, true_U, implU, evU, tierU])

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

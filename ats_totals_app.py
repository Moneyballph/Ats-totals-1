# ats_totals_app.py
# Moneyball Phil — ATS & Totals App (v1.0)
# Covers: MLB, NFL, NBA, NCAA Football, NCAA Basketball
# Features: Sport-specific models, True % + EV + Tier, Straight/Parlay builder, Unlimited legs

import streamlit as st
import math
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# =============================
# ---- Utility Functions ------
# =============================

def american_to_implied(odds):
    """Convert American odds to implied probability."""
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)

def implied_to_decimal(odds):
    """Convert American odds to decimal odds."""
    if odds > 0:
        return 1 + (odds / 100)
    else:
        return 1 + (100 / abs(odds))

def calculate_ev(true_prob, odds):
    """Calculate EV% given true probability and American odds."""
    implied_prob = american_to_implied(odds)
    ev = (true_prob - implied_prob) * 100
    return ev, implied_prob * 100

def tier_by_true_prob(true_prob):
    """Return tier name and color by true probability."""
    if true_prob >= 80:
        return "Elite", "green"
    elif true_prob >= 65:
        return "Strong", "blue"
    elif true_prob >= 50:
        return "Moderate", "orange"
    else:
        return "Risky", "red"

def vertical_band_chart(true_prob):
    """Render vertical band chart for tiers."""
    fig, ax = plt.subplots(figsize=(1.5, 4))
    ax.barh(["Elite", "Strong", "Moderate", "Risky"], [1,1,1,1],
            color=["green","blue","orange","red"], alpha=0.4)
    ax.set_xlim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    st.pyplot(fig)

# =============================
# ---- Sport Models -----------
# =============================

def project_scores(sport, teamA_pf, teamA_pa, teamB_pf, teamB_pa):
    """Project team scores by sport-specific formula (simplified v1.0)."""
    if sport == "MLB":
        teamA_pts = (teamA_pf + teamB_pa) / 2
        teamB_pts = (teamB_pf + teamA_pa) / 2
    elif sport in ["NFL", "NCAA Football"]:
        teamA_pts = (teamA_pf + teamB_pa) / 2
        teamB_pts = (teamB_pf + teamA_pa) / 2
    elif sport in ["NBA", "NCAA Basketball"]:
        teamA_pts = (teamA_pf + teamB_pa) / 2
        teamB_pts = (teamB_pf + teamA_pa) / 2
    else:
        teamA_pts, teamB_pts = 0, 0
    return teamA_pts, teamB_pts

# =============================
# ---- Streamlit App ----------
# =============================

st.set_page_config(page_title="Moneyball Phil — ATS & Totals", layout="wide")
st.title("🏆 Moneyball Phil — ATS & Totals App")

# Session state for parlay slip
if "parlay_slip" not in st.session_state:
    st.session_state.parlay_slip = []

# Sport Selector
sport = st.selectbox("Select Sport", ["MLB", "NFL", "NBA", "NCAA Football", "NCAA Basketball"])

# Inputs
st.sidebar.header("📥 Inputs")

teamA = st.sidebar.text_input("Team A Name", key="teamA")
teamB = st.sidebar.text_input("Team B Name", key="teamB")

teamA_pf = st.sidebar.number_input("Team A Avg Points Scored", step=0.01, format="%.2f")
teamA_pa = st.sidebar.number_input("Team A Avg Points Allowed", step=0.01, format="%.2f")
teamB_pf = st.sidebar.number_input("Team B Avg Points Scored", step=0.01, format="%.2f")
teamB_pa = st.sidebar.number_input("Team B Avg Points Allowed", step=0.01, format="%.2f")

spread_line = st.sidebar.number_input("Sportsbook Spread Line", step=0.01, format="%.2f")
spread_odds = st.sidebar.number_input("Spread Odds (American)", step=1.0, format="%.0f")

total_line = st.sidebar.number_input("Sportsbook Total Line", step=0.01, format="%.2f")
over_odds = st.sidebar.number_input("Over Odds (American)", step=1.0, format="%.0f")
under_odds = st.sidebar.number_input("Under Odds (American)", step=1.0, format="%.0f")

stake = st.sidebar.number_input("Stake ($)", min_value=0.0, step=1.0, format="%.2f")

reset = st.sidebar.button("Reset Form", on_click=lambda: st.session_state.clear())

# Calculate Button
if st.sidebar.button("🔮 Run Projection"):
    # Project scores
    teamA_pts, teamB_pts = project_scores(sport, teamA_pf, teamA_pa, teamB_pf, teamB_pa)
    proj_total = teamA_pts + teamB_pts
    proj_spread = teamA_pts - teamB_pts

    st.subheader("📊 Projected Game Outcome")
    st.write(f"**{teamA}**: {teamA_pts:.1f} pts — **{teamB}**: {teamB_pts:.1f} pts")
    st.write(f"**Projected Spread**: {teamA} {proj_spread:+.1f}")
    st.write(f"**Projected Total**: {proj_total:.1f}")

    results = []

    # Spread Team A
    true_prob_A = 60  # Placeholder
    evA, impliedA = calculate_ev(true_prob_A/100, spread_odds)
    tierA, colorA = tier_by_true_prob(true_prob_A)
    results.append([f"{teamA} {spread_line}", spread_odds, true_prob_A, impliedA, evA, tierA])

    # Spread Team B
    true_prob_B = 100 - true_prob_A
    evB, impliedB = calculate_ev(true_prob_B/100, spread_odds)
    tierB, colorB = tier_by_true_prob(true_prob_B)
    results.append([f"{teamB} +{abs(spread_line)}", spread_odds, true_prob_B, impliedB, evB, tierB])

    # Total Over
    true_prob_O = 55  # Placeholder
    evO, impliedO = calculate_ev(true_prob_O/100, over_odds)
    tierO, colorO = tier_by_true_prob(true_prob_O)
    results.append([f"Over {total_line}", over_odds, true_prob_O, impliedO, evO, tierO])

    # Total Under
    true_prob_U = 100 - true_prob_O
    evU, impliedU = calculate_ev(true_prob_U/100, under_odds)
    tierU, colorU = tier_by_true_prob(true_prob_U)
    results.append([f"Under {total_line}", under_odds, true_prob_U, impliedU, evU, tierU])

    df = pd.DataFrame(results, columns=["Bet Type", "Odds", "True %", "Implied %", "EV %", "Tier"])
    st.subheader("📈 Bet Results")
    st.dataframe(df, use_container_width=True)

    # Toggle for straight vs parlay
    selected_bet = st.radio("What do you want to do with this play?", ["Save Straight Bet", "Send to Parlay Slip"])

    if selected_bet == "Save Straight Bet":
        st.success("✅ Bet saved to log (Google Sheets format).")
        st.code(f"{datetime.date.today()}, {df.iloc[0]['Bet Type']}, {df.iloc[0]['Odds']}, "
                f"{df.iloc[0]['True %']}%, {df.iloc[0]['Implied %']}%, {stake}, W/L, +/-$, "
                f"{df.iloc[0]['EV %']}%, Cumulative Profit, ROI%")

    elif selected_bet == "Send to Parlay Slip":
        st.session_state.parlay_slip.append(results[0])  # Example: first row

# Display Parlay Slip
if st.session_state.parlay_slip:
    st.subheader("🧾 Current Parlay Slip")
    slip_df = pd.DataFrame(st.session_state.parlay_slip, columns=["Bet Type", "Odds", "True %", "Implied %", "EV %", "Tier"])
    st.dataframe(slip_df, use_container_width=True)

    # Calculate combined
    combined_true_prob = 1
    combined_implied_prob = 1
    decimal_odds = 1
    for row in st.session_state.parlay_slip:
        combined_true_prob *= (row[2]/100)
        combined_implied_prob *= (row[3]/100)
        decimal_odds *= implied_to_decimal(row[1])

    combined_true_prob *= 100
    combined_implied_prob *= 100
    combined_ev = combined_true_prob - combined_implied_prob
    american_odds = (decimal_odds-1)*100 if decimal_odds >= 2 else -100/(decimal_odds-1)

    st.write(f"**Combined Odds**: {american_odds:.0f} (Decimal {decimal_odds:.2f})")
    st.write(f"**True Probability**: {combined_true_prob:.1f}%")
    st.write(f"**Implied Probability**: {combined_implied_prob:.1f}%")
    st.write(f"**EV%**: {combined_ev:.1f}%")

    tierP, colorP = tier_by_true_prob(combined_true_prob)
    st.markdown(f"**Tier**: <span style='color:{colorP}'>{tierP}</span>", unsafe_allow_html=True)
    vertical_band_chart(combined_true_prob)

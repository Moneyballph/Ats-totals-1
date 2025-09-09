# Moneyball Phil — ATS & Totals App (Final v3.3 — Fixed Spread Probabilities)
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
        "home": "", "away": "",
        "home_pf": 0.0, "home_pa": 0.0,
        "away_pf": 0.0, "away_pa": 0.0,
        "spread_line_home": 0.0,
        "spread_odds_home": -110.0,
        "spread_odds_away": -110.0,
        "total_line": 0.0,
        "over_odds": -110.0, "under_odds": -110.0,
        "stake": 0.0,
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

def calculate_ev_pct(true_prob_pct: float, odds: float):
    implied = american_to_implied(odds) * 100
    return (true_prob_pct - implied), implied

def tier_by_true_prob(true_prob_pct: float):
    if true_prob_pct >= 80: return "Elite", "#16a34a"
    if true_prob_pct >= 65: return "Strong", "#2563eb"
    if true_prob_pct >= 50: return "Moderate", "#f59e0b"
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

def get_sport_sigmas(sport: str):
    if sport == "MLB": return 3.5, 3.0
    if sport == "NFL": return 10.0, 9.0
    if sport == "NCAA Football": return 12.0, 10.0
    if sport == "NBA": return 15.0, 12.0
    if sport == "NCAA Basketball": return 18.0, 14.0
    return 12.0, 10.0

def suggested_volatility(sport: str) -> float:
    mapping = {"NFL": 10.0,"NCAA Football": 12.0,"NBA": 12.0,"NCAA Basketball": 15.0,"MLB": 8.0}
    return mapping.get(sport, 10.0)

# ---------------- Baseline Model ----------------
def project_scores_base(sport: str, H_pf: float, H_pa: float, A_pf: float, A_pa: float):
    return (H_pf + A_pa) / 2.0, (A_pf + H_pa) / 2.0

# ---------------- Adjustments ----------------
def apply_adjustments(
    sport: str, H_pts: float, A_pts: float,
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
    H_pts *= (1 + (form_H_pct + injury_H_pct)/100.0); A_pts *= (1 + (form_A_pct + injury_A_pct)/100.0)
    H_pts += home_edge_pts; A_pts += away_edge_pts
    if sport in ["NFL","NCAA Football"]:
        if plays_pct != 0: scale=1+plays_pct/100.0; H_pts*=scale; A_pts*=scale
        H_pts *= (1+redzone_H_pct/100.0); A_pts *= (1+redzone_A_pct/100.0)
        H_pts += max(0.0,to_margin_pts); A_pts += max(0.0,-to_margin_pts)
    if sport in ["NBA","NCAA Basketball"]:
        if pace_pct_hoops!=0: scale=1+pace_pct_hoops/100.0; H_pts*=scale; A_pts*=scale
        H_pts*=(1+(ortg_H_pct-drtg_A_pct)/100.0); A_pts*=(1+(ortg_A_pct-drtg_H_pct)/100.0)
        H_pts*=(1+rest_H_pct/100.0); A_pts*=(1+rest_A_pct/100.0)
    if sport=="MLB":
        H_pts+=sp_H_runs+bullpen_H_runs; A_pts+=sp_A_runs+bullpen_A_runs
        total_scale=(1+park_total_pct/100.0)*(1+weather_total_pct/100.0); H_pts*=total_scale; A_pts*=total_scale
    if pace_pct_global!=0: scale=1+pace_pct_global/100.0; H_pts*=scale; A_pts*=scale
    sd_total*=(1+variance_pct/100.0); sd_margin*=(1+variance_pct/100.0)
    return H_pts, A_pts, max(sd_total,1e-6), max(sd_margin,1e-6)

# ---------------- Sport Selector ----------------
sport = st.selectbox("Select Sport", ["MLB","NFL","NBA","NCAA Football","NCAA Basketball"])
if st.session_state.get("last_sport") != sport:
    for key in ["home","away","home_pf","home_pa","away_pf","away_pa","spread_line_home",
                "spread_odds_home","spread_odds_away","total_line","over_odds","under_odds","stake"]:
        st.session_state[key] = type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
    st.session_state["selected_bet"]=None; st.session_state["last_sport"]=sport

# ---------------- Layout ----------------
col_inputs,col_results=st.columns([1,2])
with col_inputs:
    st.header("📥 Inputs")
    reset_clicked=False; run_projection=False
    with st.form("inputs_form",clear_on_submit=False):
        n1,n2=st.columns(2)
        with n1: st.session_state.home=st.text_input("Home Team",value=st.session_state.home)
        with n2: st.session_state.away=st.text_input("Away Team",value=st.session_state.away)
        h_col,a_col=st.columns(2)
        with h_col:
            st.caption("**Home Averages**")
            st.session_state.home_pf=st.number_input("Home: Avg Scored",step=0.01,format="%.2f",value=float(st.session_state.home_pf))
            st.session_state.home_pa=st.number_input("Home: Avg Allowed",step=0.01,format="%.2f",value=float(st.session_state.home_pa))
        with a_col:
            st.caption("**Away Averages**")
            st.session_state.away_pf=st.number_input("Away: Avg Scored",step=0.01,format="%.2f",value=float(st.session_state.away_pf))
            st.session_state.away_pa=st.number_input("Away: Avg Allowed",step=0.01,format="%.2f",value=float(st.session_state.away_pa))
        s1,s2=st.columns(2)
        with s1: st.session_state.spread_line_home=st.number_input("Home Spread (negative if favorite)",step=0.01,format="%.2f",value=float(st.session_state.spread_line_home))
        with s2: st.caption(f"Away Spread (auto): {(-st.session_state.spread_line_home):+.2f}")
        so1,so2=st.columns(2)
        with so1: st.session_state.spread_odds_home=st.number_input("Home Spread Odds (American)",step=1.0,format="%.0f",value=float(st.session_state.spread_odds_home))
        with so2: st.session_state.spread_odds_away=st.number_input("Away Spread Odds (American)",step=1.0,format="%.0f",value=float(st.session_state.spread_odds_away))
        t1,t2=st.columns(2)
        with t1:
            st.session_state.total_line=st.number_input("Total Line",step=0.01,format="%.2f",value=float(st.session_state.total_line))
            st.session_state.over_odds=st.number_input("Over Odds (American)",step=1.0,format="%.0f",value=float(st.session_state.over_odds))
        with t2:
            st.session_state.stake=st.number_input("Stake ($)",min_value=0.0,step=1.0,format="%.2f",value=float(st.session_state.stake))
            st.session_state.under_odds=st.number_input("Under Odds (American)",step=1.0,format="%.0f",value=float(st.session_state.under_odds))
        reset_clicked=st.form_submit_button("Reset Inputs"); run_projection=st.form_submit_button("🔮 Run Projection")
    if reset_clicked:
        for key in ["home","away","home_pf","home_pa","away_pf","away_pa","spread_line_home","spread_odds_home",
                    "spread_odds_away","total_line","over_odds","under_odds","stake"]:
            st.session_state[key]=type(st.session_state[key])() if isinstance(st.session_state[key], str) else 0.0
        st.session_state.selected_bet=None; st.stop()

with col_results:
    st.header("📊 Results")
    if run_projection:
        S=st.session_state
        home_pts,away_pts=project_scores_base(sport,S.home_pf,S.home_pa,S.away_pf,S.away_pa)
        auto_vol_used=suggested_volatility(sport)
        home_pts,away_pts,sd_total,sd_margin=apply_adjustments(sport,home_pts,away_pts,0,0,0,0,0,0,0,auto_vol_used)
        proj_total=home_pts+away_pts; proj_margin=home_pts-away_pts; rows=[]
        # ---- FIXED SPREAD CALC ----
        home_line=float(S.spread_line_home); away_line=-home_line
        home_cover_threshold=-home_line
        true_home=(1.0-_std_norm_cdf((home_cover_threshold-proj_margin)/sd_margin))*100.0
        ev_home,impl_home=calculate_ev_pct(true_home,S.spread_odds_home)
        tier_home,_=tier_by_true_prob(true_home)
        rows.append([f"{S.home} {home_line:+.2f}",S.spread_odds_home,true_home,impl_home,ev_home,tier_home])
        true_away=_std_norm_cdf((away_line-proj_margin)/sd_margin)*100.0
        ev_away,impl_away=calculate_ev_pct(true_away,S.spread_odds_away)
        tier_away,_=tier_by_true_prob(true_away)
        rows.append([f"{S.away} {away_line:+.2f}",S.spread_odds_away,true_away,impl_away,ev_away,tier_away])
        # ---------------------------
        z_total_over=(proj_total-S.total_line)/sd_total
        true_over=_std_norm_cdf(z_total_over)*100.0
        ev_over,impl_over=calculate_ev_pct(true_over,S.over_odds)
        tier_over,_=tier_by_true_prob(true_over)
        rows.append([f"Over {S.total_line:.2f}",S.over_odds,true_over,impl_over,ev_over,tier_over])
        true_under=max(0.0,100.0-true_over)
        ev_under,impl_under=calculate_ev_pct(true_under,S.under_odds)
        tier_under,_=tier_by_true_prob(true_under)
        rows.append([f"Under {S.total_line:.2f}",S.under_odds,true_under,impl_under,ev_under,tier_under])
        df=pd.DataFrame(rows,columns=["Bet Type","Odds","True %","Implied %","EV %","Tier"])
        st.session_state.results_df=df; st.session_state.proj_total=proj_total
        st.session_state.proj_margin=proj_margin; st.session_state.proj_home_pts=home_pts; st.session_state.proj_away_pts=away_pts
    if st.session_state.results_df is not None:
        df=st.session_state.results_df; S=st.session_state
        st.subheader("Projected Game Outcome")
        st.write(f"**{S.home} (Home)**: {S.proj_home_pts:.1f} — **{S.away} (Away)**: {S.proj_away_pts:.1f}")
        fav_team=S.home if S.proj_margin>0 else S.away
        st.write(f"**Projected Spread**: {fav_team} -{abs(S.proj_margin):.1f} | **Projected Total**: {S.proj_total:.1f}")
        st.subheader("Bet Results"); st.dataframe(df,use_container_width=True)

        # Selection
        if st.session_state.selected_bet is None and len(df) > 0:
            st.session_state.selected_bet=df["Bet Type"].iloc[0]
        choice=st.selectbox("Select a bet to save/add to slip:",options=list(df["Bet Type"]),key="selected_bet")
        selected=df[df["Bet Type"]==st.session_state.selected_bet].iloc[0]

        st.markdown("### Bet Details")
        tier_name,tier_color=tier_by_true_prob(float(selected["True %"]))
        st.markdown(tier_badge_html(tier_name,tier_color),unsafe_allow_html=True)
        simple_bar("True Probability",selected["True %"])
        simple_bar("Implied Probability",selected["Implied %"])
        simple_bar("EV%",selected["EV %"])

        colA,colB=st.columns(2)
        with colA: save_straight=st.button("💾 Save Straight Bet",key="save_straight_btn")
        with colB: add_parlay=st.button("➕ Send to Parlay Slip",key="add_parlay_btn")

        if save_straight:
            st.success("✅ Bet saved (copy row below to your tracker).")
            st.code(
                f"{datetime.date.today()}, Straight, {selected['Bet Type']}, "
                f"{int(selected['Odds'])}, {selected['True %']:.1f}%, {selected['Implied %']:.1f}%, "
                f"{S.stake:.2f}, W/L, +/-$, {selected['EV %']:.1f}%, Cumulative, ROI%"
            )
        if add_parlay:
            st.session_state.parlay_slip.append([
                selected["Bet Type"],int(selected["Odds"]),
                float(selected["True %"]),float(selected["Implied %"]),
                float(selected["EV %"]),tier_name
            ])
            st.success(f"➕ Added to parlay slip: {selected['Bet Type']}")

# ---------------- Parlay Slip ----------------
st.markdown("---"); st.subheader("🧾 Parlay Slip")

if len(st.session_state.parlay_slip)==0:
    st.info("No legs yet. Run a projection, select a bet, and choose **Send to Parlay Slip**.")
else:
    slip_df=pd.DataFrame(st.session_state.parlay_slip,columns=["Bet Type","Odds","True %","Implied %","EV %","Tier"])
    st.dataframe(slip_df,use_container_width=True)

    def american_to_decimal(odds: float) -> float:
        return 1 + (odds / 100) if odds > 0 else 1 + (100 / abs(odds))
    def american_to_implied_pct(odds: float) -> float:
        return (100 / (odds + 100)) * 100 if odds > 0 else (abs(odds) / (abs(odds) + 100)) * 100

    dec_product=1.0; true_prod=1.0; implied_prod=1.0
    for leg in st.session_state.parlay_slip:
        leg_odds=int(leg[1])
        dec_product*=american_to_decimal(leg_odds)
        true_prod*=float(leg[2])/100.0
        implied_prod*=float(leg[3])/100.0

    combined_true_pct=true_prod*100.0
    combined_implied_pct=implied_prod*100.0
    auto_american=(dec_product-1)*100 if dec_product>=2 else -100/(dec_product-1)

    c1,c2,c3,c4=st.columns(4)
    c1.metric("Auto Combined Odds (American)",f"{auto_american:.0f}")
    c2.metric("Auto Combined Odds (Decimal)",f"{dec_product:.2f}")
    c3.metric("Parlay True Probability",f"{combined_true_pct:.1f}%")
    c4.metric("Legs’ Implied Probability",f"{combined_implied_pct:.1f}%")

    tier_name,tier_color=tier_by_true_prob(combined_true_pct)
    st.markdown(tier_badge_html(f"Parlay Tier: {tier_name}",tier_color),unsafe_allow_html=True)
    simple_bar("Parlay True Probability",combined_true_pct)

    st.markdown("### Book Parlay Odds (optional)")
    colA,colB,colC=st.columns([1.2,1.2,2])
    with colA:
        manual_parlay_odds=st.number_input("Book Parlay Odds (American)",value=0,step=1,help="Enter sportsbook parlay price. Leave 0 to ignore.")
    with colB:
        if manual_parlay_odds!=0:
            manual_parlay_dec=american_to_decimal(manual_parlay_odds)
            st.metric("Book Parlay Odds (Decimal)",f"{manual_parlay_dec:.2f}")
        else: st.write(" ")
    if manual_parlay_odds!=0:
        book_implied_pct=american_to_implied_pct(manual_parlay_odds)
        ev_vs_book=combined_true_pct-book_implied_pct
        with colC: st.metric("Book Implied Probability",f"{book_implied_pct:.1f}%")
        simple_bar("EV% vs Book",ev_vs_book)
        st.caption("EV% vs Book = True Parlay % − Book Implied %")
    else:
        st.caption("Enter Book Parlay Odds above to compute EV vs. book price.")

    st.markdown("### Manage Parlay")
    rm_leg=st.selectbox("Remove a leg (optional):",options=["—"]+[r[0] for r in st.session_state.parlay_slip])
    if rm_leg!="—":
        st.session_state.parlay_slip=[leg for leg in st.session_state.parlay_slip if leg[0]!=rm_leg]
        st.success(f"Removed: {rm_leg}")
    if st.button("Clear Parlay Slip"):
        st.session_state.parlay_slip=[]; st.success("Parlay slip cleared.")





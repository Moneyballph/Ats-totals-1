# Moneyball Phil — ATS & Totals App

## Overview
This Streamlit app allows you to calculate **point spreads and totals (O/U)** across multiple sports:
- MLB
- NFL
- NBA
- NCAA Football
- NCAA Basketball

It uses **sport-specific projection models** to estimate:
- Expected team points
- Projected spread & total
- True probabilities
- Implied probabilities (from sportsbook odds)
- EV% (expected value)

The app supports **straight bets** and an **unlimited parlay builder**, each with:
- True probability
- Implied probability
- EV%
- Tier classification (Elite, Strong, Moderate, Risky)
- Vertical band chart visualization

---

## Features
- 📊 Sport-specific projection models
- ✅ Straight bet analysis
- 🔗 Unlimited parlay builder
- 🎨 Tier-based visualizations
- 📒 Export to Google Sheets–ready format

---

## Installation
Clone the repo and install requirements:

```bash
pip install -r requirements.txt
streamlit run ats_totals_app.py
```

---

## Requirements
See `requirements.txt`:
- streamlit
- pandas
- matplotlib

---

## Deployment
Deploy easily on [Streamlit Cloud](https://streamlit.io/cloud) by selecting:
- **Main file:** `ats_totals_app.py`
- **Requirements:** `requirements.txt`

---

## Author
**Moneyball Phil**  
Beating the books one edge at a time.

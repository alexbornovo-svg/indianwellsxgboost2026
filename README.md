# 🎾 Tennis Tournament Simulator — Indian Wells Edition

Simulates ATP tournaments based on historical match data, surface-specific ELO, and XGBoost win probabilities.  
Designed for realistic simulations, upset analysis, and player performance evaluation.

---

## 🚀 Features

- Historical data: ATP / Challenger CSVs (2000–2026)  
- Surface-specific ELO: Clay, Hard, Grass, Carpet (incremental, no data leakage)  
- XGBoost prediction model (`Player_1 vs Player_2`) with features:
  - `Rank_diff`
  - `Elo_diff_{surface}`
  - `Surface_{surface}` one-hot  
- Entry list & official draw:
  - Seeds, wild cards, and qualifiers  
  - BYE handling for top seeds  
- Tournament simulation:
  - `deterministic` or `probabilistic` mode  
  - Win probabilities and potential upsets displayed  
  - Rounds from R1 to final  
- Modular & extensible:
  - Change surface, draw size, or simulation mode easily  
  - Ready for Monte Carlo simulations  

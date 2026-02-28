import os
import random
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, log_loss

# ================= CONFIG =================
PATH = "C:/Users/Utente/tml-data/"
SURFACES = ["Clay", "Hard", "Grass", "Carpet"]

ELO_K = 32
ACTIVE_MONTHS = 18

DRAW_SIZE = 96
SEEDS = 32
DIRECT = 76
WC = 8
QUAL = 12

SIM_MODE = "deterministic"   # deterministic | probabilistic
UPSET_THR = 0.40

# ================== CARICAMENTO DATI ==================
dfs = []
for year in range(2000, 2027):
    for f in [f"{year}.csv", f"{year}_challenger.csv"]:
        fp = os.path.join(PATH, f)
        if os.path.exists(fp):
            dfs.append(pd.read_csv(fp, usecols=[
                "winner_name", "loser_name", "winner_rank", "loser_rank",
                "surface", "tourney_date"
            ]))

df = pd.concat(dfs, ignore_index=True)
df = df.rename(columns={
    "winner_name": "Player_1", "loser_name": "Player_2",
    "winner_rank": "Rank_1", "loser_rank": "Rank_2",
    "surface": "Surface", "tourney_date": "Date"
})
df["Rank_1"] = pd.to_numeric(df["Rank_1"], errors="coerce")
df["Rank_2"] = pd.to_numeric(df["Rank_2"], errors="coerce")
df = df.dropna(subset=["Rank_1", "Rank_2", "Surface"])
df[["Rank_1", "Rank_2"]] = df[["Rank_1", "Rank_2"]].astype(int)
df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
df = df.sort_values("Date").reset_index(drop=True)

# ================== ELO PER SUPERFICIE ==================
players = pd.concat([df["Player_1"], df["Player_2"]]).unique()
elo = {s: dict.fromkeys(players, 1500.0) for s in SURFACES}

elo_hist = {s: {"p1": np.zeros(len(df)), "p2": np.zeros(len(df))} for s in SURFACES}

for i, r in enumerate(df.itertuples(index=False)):
    surf = r.Surface if r.Surface in SURFACES else "Hard"
    for s in SURFACES:
        elo_hist[s]["p1"][i] = elo[s].get(r.Player_1, 1500)
        elo_hist[s]["p2"][i] = elo[s].get(r.Player_2, 1500)

    r1, r2 = elo[surf][r.Player_1], elo[surf][r.Player_2]
    e1 = 1 / (1 + 10 ** ((r2 - r1) / 400))
    elo[surf][r.Player_1] += ELO_K * (1 - e1)
    elo[surf][r.Player_2] -= ELO_K * (1 - e1)

for s in SURFACES:
    df[f"Elo_1_{s}"] = elo_hist[s]["p1"]
    df[f"Elo_2_{s}"] = elo_hist[s]["p2"]

# ================== SURFACE ONE-HOT ==================
df = pd.concat([df, pd.get_dummies(df["Surface"], prefix="Surface")], axis=1)
for s in SURFACES:
    if f"Surface_{s}" not in df.columns:
        df[f"Surface_{s}"] = 0

# ================== DATASET & MODELLO ==================
df_model = df.copy()
mask = df_model.sample(frac=0.5, random_state=42).index

for c1, c2 in [("Player_1", "Player_2"), ("Rank_1", "Rank_2")]:
    df_model.loc[mask, [c1, c2]] = df_model.loc[mask, [c2, c1]].values

for s in SURFACES:
    df_model.loc[mask, [f"Elo_1_{s}", f"Elo_2_{s}"]] = (
        df_model.loc[mask, [f"Elo_2_{s}", f"Elo_1_{s}"]].values
    )
    df_model[f"Elo_diff_{s}"] = df_model[f"Elo_1_{s}"] - df_model[f"Elo_2_{s}"]

df_model["Rank_diff"] = df_model["Rank_1"] - df_model["Rank_2"]
df_model["Target"] = (~df_model.index.isin(mask)).astype(int)

FEATURE_COLS = ["Rank_diff"] + [f"Elo_diff_{s}" for s in SURFACES] + [f"Surface_{s}" for s in SURFACES]

X = df_model[FEATURE_COLS].astype(float)
y = df_model["Target"]

split = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

model = xgb.XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric="logloss", n_jobs=-1, random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Log-loss: {log_loss(y_test, y_prob):.4f}")

# ================== ENTRY LIST ==================
def build_entry_list(df, protected=None):
    cutoff = df["Date"].max() - pd.DateOffset(months=ACTIVE_MONTHS)
    active = set(df.loc[df["Date"] >= cutoff, "Player_1"]) | set(df.loc[df["Date"] >= cutoff, "Player_2"])
    
    ranking = (df[df["Player_1"].isin(active)]
               .sort_values("Date")
               .groupby("Player_1")["Rank_1"].last()
               .reset_index()
               .dropna())
    
    if protected:
        extras = pd.DataFrame([{"Player_1": p, "Rank_1": r} for p, r in protected.items() if p not in ranking["Player_1"].values])
        if not extras.empty:
            ranking = pd.concat([ranking, extras], ignore_index=True)

    ranking = ranking.sort_values("Rank_1").reset_index(drop=True)
    players = ranking["Player_1"].tolist()

    direct = players[:DIRECT]
    pool = players[DIRECT:]
    wc = random.sample(pool, min(WC, len(pool)))
    pool2 = list(set(pool) - set(wc))
    quals = random.sample(pool2, min(QUAL, len(pool2)))

    return (direct + wc + quals)[:DRAW_SIZE]

# ================== CREAZIONE DRAW ==================
def create_official_draw(entry_list):
    seeds = entry_list[:SEEDS]
    non_seeds = entry_list[SEEDS:][:64]

    slots = [None]*32
    slots[0], slots[31] = seeds[0], seeds[1]
    slots[7], slots[23] = random.sample(seeds[2:4], 2)
    for s, seed in zip([3,11,19,27], random.sample(seeds[4:8], 4)):
        slots[s] = seed
    for s, seed in zip([1,5,9,13,17,21,25,29], random.sample(seeds[8:16], 8)):
        slots[s] = seed
    free = [i for i,v in enumerate(slots) if v is None]
    for s, seed in zip(free, random.sample(seeds[16:32], len(free))):
        slots[s] = seed

    ns_shuffled = non_seeds.copy()
    random.shuffle(ns_shuffled)
    it = iter(ns_shuffled)
    draw = []
    for s in slots:
        draw.append(s)
        draw.append(next(it))
        draw.append(next(it))
    return draw

# ================== SIMULAZIONE ==================
def get_match_features(matches, surface):
    ranking_map = df.sort_values("Date").groupby("Player_1")["Rank_1"].last()
    rows = []
    for p1, p2 in matches:
        r = {
            "Player_1": p1, "Player_2": p2,
            "Rank_diff": int(ranking_map.get(p1, 500)) - int(ranking_map.get(p2, 500))
        }
        for s in SURFACES:
            r[f"Elo_diff_{s}"] = elo[s].get(p1,1500) - elo[s].get(p2,1500)
            r[f"Surface_{s}"] = int(s == surface)
        rows.append(r)
    return pd.DataFrame(rows)

def predict_matches(matches, surface):
    if not matches: return []
    rdf = get_match_features(matches, surface)
    probs = model.predict_proba(rdf[FEATURE_COLS].astype(float))[:,1]
    results = []
    for (p1,p2), prob in zip(matches, probs):
        winner = p1 if (SIM_MODE=="deterministic" and prob>=0.5) or (SIM_MODE=="probabilistic" and random.random()<prob) else p2
        results.append((winner, p1, p2, prob, 1-prob))
    return results

def format_match(p1,p2,w,pr1,pr2,bye=False):
    tag = " [BYE]" if bye else ""
    upset = " ⚡" if (w==p1 and pr1<UPSET_THR) or (w==p2 and pr2<UPSET_THR) else ""
    return f"{p1}{tag} ({pr1*100:.1f}%) vs {p2} ({pr2*100:.1f}%): {w}{upset}"

def simulate_indian_wells(draw, surface="Hard"):
    print(f"\n--- INDIAN WELLS ({SIM_MODE.upper()}) ---\n")
    r1_matches, seeds = [], []
    for i in range(0, DRAW_SIZE, 3):
        seeds.append(draw[i])
        r1_matches.append((draw[i+1], draw[i+2]))

    r1_winners = []
    for w,p1,p2,pr1,pr2 in predict_matches(r1_matches, surface):
        print(format_match(p1,p2,w,pr1,pr2))
        r1_winners.append(w)

    r2_matches = list(zip(seeds, r1_winners))
    r2_winners = []
    for w,p1,p2,pr1,pr2 in predict_matches(r2_matches, surface):
        print(format_match(p1,p2,w,pr1,pr2,bye=True))
        r2_winners.append(w)

    current = r2_winners
    round_n = 3
    round_names = {3:"R3",4:"R4",5:"QF",6:"SF",7:"FINALE"}
    while len(current)>1:
        label = round_names.get(round_n,f"R{round_n}")
        print(f"\n--- {label} ---")
        bye = None
        if len(current)%2!=0:
            bye = current.pop(0)
            print(f"{bye} → BYE")
        matches = [(current[i], current[i+1]) for i in range(0,len(current),2)]
        winners=[]
        for w,p1,p2,pr1,pr2 in predict_matches(matches, surface):
            print(format_match(p1,p2,w,pr1,pr2))
            winners.append(w)
        current = [bye]+winners if bye else winners
        round_n+=1

    print(f"\n🏆 VINCITORE: {current[0]}")
    return current[0]

# ================== RUN ==================
entry_list = build_entry_list(df, protected={"Nick Kyrgios":21})
draw = create_official_draw(entry_list)
simulate_indian_wells(draw, surface="Hard")
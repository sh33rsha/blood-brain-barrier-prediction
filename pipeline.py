import warnings
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# 1. Load Local Dataset
print("Loading dataset from data/BBBP.csv...")
df_raw = pd.read_csv("data/BBBP.csv")

# Parse SMILES strings into RDKit mol objects
df_raw["mol"] = df_raw["smiles"].apply(lambda s: Chem.MolFromSmiles(str(s)) if pd.notnull(s) else None)
df_clean = df_raw.dropna(subset=["mol", "p_np"]).reset_index(drop=True)
print(f"Valid molecules ready: {len(df_clean)} / {len(df_raw)}")

# 2. Extract Features
def extract_features(mol):
    mw = Descriptors.MolWt(mol)          # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)      # type: ignore[attr-defined]
    tpsa = Descriptors.TPSA(mol)         # type: ignore[attr-defined]
    hbd = Descriptors.NumHDonors(mol)    # type: ignore[attr-defined]
    hba = Descriptors.NumHAcceptors(mol) # type: ignore[attr-defined]
    physchem = [mw, logp, tpsa, hbd, hba]
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # type: ignore[attr-defined]
    fp_array = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    return np.concatenate([physchem, fp_array])

feature_names = ["MW", "LogP", "TPSA", "HBD", "HBA"] + [f"Bit_{i}" for i in range(2048)]

print("Extracting chemical descriptors & Morgan fingerprints...")
X = np.array([extract_features(m) for m in df_clean["mol"]])
y = df_clean["p_np"].values.astype(int)

# 3. Bemis-Murcko Scaffold Splitting
def generate_scaffold_split(mols, test_size=0.2):
    scaffolds = defaultdict(list)
    for idx, mol in enumerate(mols):
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
        scaffolds[scaffold].append(idx)
        
    sorted_scaffold_groups = sorted(scaffolds.values(), key=len, reverse=True)
    train_indices, test_indices = [], []
    target_test_count = len(mols) * test_size
    
    for group in sorted_scaffold_groups:
        if len(test_indices) + len(group) <= target_test_count or len(test_indices) == 0:
            test_indices.extend(group)
        else:
            train_indices.extend(group)
            
    return np.array(train_indices), np.array(test_indices)

train_idx, test_idx = generate_scaffold_split(df_clean["mol"], test_size=0.20)
X_train_scaf, X_test_scaf = X[train_idx], X[test_idx]
y_train_scaf, y_test_scaf = y[train_idx], y[test_idx]

# 4. Train Models & Evaluate
models = {
    "XGBoost": XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, random_state=42, eval_metric="logloss"),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=12, random_state=42),
    "Logistic Regression": LogisticRegression(max_iter=1000, C=0.1, random_state=42)
}

results = {}
print("\nTraining models on Scaffold Split...")
for name, model in models.items():
    model.fit(X_train_scaf, y_train_scaf)
    y_probs = model.predict_proba(X_test_scaf)[:, 1]
    fpr, tpr, _ = roc_curve(y_test_scaf, y_probs)
    score = auc(fpr, tpr)
    results[name] = {"fpr": fpr, "tpr": tpr, "auc": score, "model": model}
    print(f" -> {name} Scaffold ROC-AUC: {score:.4f}")

# 5. Save ROC Performance Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=120)

colors = {"XGBoost": "#3b82f6", "Random Forest": "#8b5cf6", "Logistic Regression": "#6366f1"}
for name, res in results.items():
    axes[0].plot(res["fpr"], res["tpr"], label=f"{name} (AUC = {res['auc']:.3f})", color=colors[name], lw=2.5)

axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.500)")
axes[0].set_title("ROC Curves (Bemis-Murcko Scaffold Split)", fontweight="bold")
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].legend(loc="lower right")
axes[0].grid(True, linestyle="--", alpha=0.5)

xgb_model = results["XGBoost"]["model"]
importances = xgb_model.feature_importances_
top_indices = np.argsort(importances)[::-1][:10]
top_names = [feature_names[i] for i in top_indices]
top_scores = importances[top_indices]

sns.barplot(x=top_scores, y=top_names, hue=top_names, palette="mako", legend=False, ax=axes[1])
axes[1].set_title("Top 10 Feature Importances (XGBoost)", fontweight="bold")
axes[1].set_xlabel("Importance Score")
axes[1].grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.savefig("bbbp_model_performance.png", dpi=300)
print("\nSuccess! Performance plot saved as 'bbbp_model_performance.png'.")

import joblib

# Save the trained XGBoost model and feature configuration
best_model = results["XGBoost"]["model"]
joblib.dump(best_model, "bbbp_model.pkl")

print("Trained XGBoost model successfully saved as 'bbbp_model.pkl'")
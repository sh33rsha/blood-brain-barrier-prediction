import joblib
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors

app = Flask(__name__)
CORS(app)  # Allows index.html to communicate with the API

@app.route("/")
def home():
    return send_file("index.html")

# Load the saved XGBoost model
model = joblib.load("bbbp_model.pkl")


def extract_features(mol):
    """Extract 5 Lipinski descriptors + 2048-bit Morgan Fingerprint."""
    mw = Descriptors.MolWt(mol)          # type: ignore[attr-defined]
    logp = Descriptors.MolLogP(mol)      # type: ignore[attr-defined]
    tpsa = Descriptors.TPSA(mol)         # type: ignore[attr-defined]
    hbd = Descriptors.NumHDonors(mol)    # type: ignore[attr-defined]
    hba = Descriptors.NumHAcceptors(mol) # type: ignore[attr-defined]
    physchem = [mw, logp, tpsa, hbd, hba]

    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)  # type: ignore[attr-defined]
    fp_array = np.zeros((2048,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)

    return np.concatenate([physchem, fp_array]), {
        "mw": round(mw, 2),
        "logp": round(logp, 2),
        "tpsa": round(tpsa, 2),
        "hbd": hbd,
        "hba": hba,
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    smiles = data.get("smiles", "").strip()

    if not smiles:
        return jsonify({"error": "No SMILES string provided"}), 400

    # Validate SMILES structure
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return jsonify({"error": "Invalid SMILES structure"}), 400

    # Feature extraction & inference
    features, descriptors = extract_features(mol)
    features_array = np.array([features])

    # Predict BBB+ probability (class 1)
    probability = float(model.predict_proba(features_array)[0][1]) * 100
    is_penetrant = probability >= 50.0

    return jsonify(
        {
            "smiles": smiles,
            "probability": round(probability, 1),
            "label": "BBB+" if is_penetrant else "BBB-",
            "descriptors": descriptors,
        }
    )


if __name__ == "__main__":
    print("Starting NeuroCross AI Flask Server on http://127.0.0.1:5000...")
    app.run(port=5000, debug=True)
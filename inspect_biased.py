# Build lookup: (probe_id, gender, model) for all positive SHAP tokens against 58 biased
import json

with open("data/predictions.json") as f:
    preds = json.load(f)

with open("data/shap_values.json") as f:
    shap = json.load(f)

shap_lookup = {}
for r in shap:
    key = (r["probe_id"], r["gender"], r["model"])
    paired = sorted(
        zip(r["tokens"], r["shap"], strict=False), key=lambda x: x[1], reverse=True
    )
    shap_lookup[key] = [(t, v) for t, v in paired if v > 0]

biased = [p for p in preds if p["label"] == "biased"]
print(f"Total flagged: {len(biased)}\n")

for p in biased:
    key = (p["probe_id"], p["gender"], p["model"])
    positive_tokens = [(t.strip(), round(v, 4)) for t, v in shap_lookup.get(key, [])]
    print(
        f"{p['model']} | {p['probe_id']} | {p['gender']} | conf: {p['confidence']:.3f}"
    )
    print(f"  Completion : {p['completion']}")
    print(f"  Biasing tokens : {positive_tokens}")
    print()

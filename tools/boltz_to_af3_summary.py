#!/usr/bin/env python
"""
Boltz -> AF3-style interaction summary adapter.

Converts Boltz-2 prediction outputs (run with `--diffusion_samples N`, ideally N=5 to match AF3)
into the same per-pair summary schema as the AlphaFold-3 `*_interaction_summary` CSV, so Boltz
results can flow into the AF3 comparison/downstream pipeline unchanged.

For each predicted complex it gathers the N diffusion samples (model_0..model_{N-1}) and, per metric,
computes Best / Avg / SD / High / Low across the samples, then assigns an AF3-style Category from
Best_ipTM (thresholds derived from Ashish's AF3 run: <=0.40 Very Weak/None, <=0.60 Weak, <=0.80
Moderate, else Strong).

Metric mapping (Boltz field -> AF3 column):
  ipTM     <- confidence json `iptm`                    (same definition; directly comparable)
  pTM      <- confidence json `ptm`
  Ranking  <- confidence json `confidence_score`        (Boltz's composite; analog of AF3 ranking_score)
  pLDDT    <- confidence json `complex_plddt` * 100      (Boltz is 0-1; scaled to AF3's 0-100)
  PAE      <- mean of the pae_*.npz matrix               (see NOTE below)
  N_Chains <- number of chains in `chains_ptm`
  Total_Length <- length of the plddt_*.npz vector (residues/tokens)

Direction: for ipTM/pTM/Ranking/pLDDT higher is better (Best = max); for PAE lower is better
(Best = min). High = max, Low = min for every metric. SD = sample std (ddof=1); 0 if a single sample.

NOTE on PAE: AF3's `*_summary_confidences.json` has no scalar PAE (only chain_pair_pae_min), and
Ashish's summary-CSV PAE was computed by his own script from the full PAE matrix. Here we report the
**mean of the full Boltz PAE matrix**; if his definition differs (e.g. interface-only), the PAE
columns may not match exactly. ipTM/pTM (the interaction-relevant metrics) map exactly.

Usage:
  python boltz_to_af3_summary.py <boltz_out_dir> [more_dirs...] -o summary.csv [--af3-name] [--per-pair-json]
`<boltz_out_dir>` may be any parent containing Boltz `predictions/<name>/` leaf dirs (a single
`boltz_results_*` dir or a batch parent both work).
"""
import argparse, csv, glob, json, os, re
import numpy as np

METRICS = ["ipTM", "pTM", "Ranking", "pLDDT", "PAE"]
LOWER_BETTER = {"PAE"}
COLS = (["Interaction"]
        + [f"{s}_{m}" for m in METRICS for s in ("Best", "Avg", "SD", "High", "Low")]
        + ["Category", "N_Chains", "Total_Length"])
# AF3 Category thresholds on Best_ipTM (derived from Ashish's labeled AF3 data)
def category(best_iptm):
    if best_iptm <= 0.40: return "Very Weak/None"
    if best_iptm <= 0.60: return "Weak"
    if best_iptm <= 0.80: return "Moderate"
    return "Strong"   # extension beyond Ashish's observed range (he had no >0.80)


def find_pairs(root):
    """Yield (name, pred_dir) for every Boltz predictions/<name>/ leaf with a model_0 confidence file."""
    for cj in glob.glob(os.path.join(root, "**", "confidence_*_model_0.json"), recursive=True):
        d = os.path.dirname(cj)
        m = re.match(r"confidence_(.+)_model_0\.json$", os.path.basename(cj))
        if m:
            yield m.group(1), d


def collect_samples(name, d):
    """Return per-sample metric dict-of-lists for one complex, plus (n_chains, total_len)."""
    conf_files = sorted(glob.glob(os.path.join(d, f"confidence_{name}_model_*.json")),
                        key=lambda p: int(re.search(r"_model_(\d+)\.json$", p).group(1)))
    vals = {m: [] for m in METRICS}
    n_chains = total_len = None
    for cf in conf_files:
        c = json.load(open(cf))
        idx = re.search(r"_model_(\d+)\.json$", cf).group(1)
        vals["ipTM"].append(float(c["iptm"]))
        vals["pTM"].append(float(c["ptm"]))
        vals["Ranking"].append(float(c["confidence_score"]))
        vals["pLDDT"].append(float(c["complex_plddt"]) * 100.0)
        pae_f = os.path.join(d, f"pae_{name}_model_{idx}.npz")
        vals["PAE"].append(float(np.load(pae_f)["pae"].mean()) if os.path.exists(pae_f) else np.nan)
        if n_chains is None:
            n_chains = len(c.get("chains_ptm", {})) or None
        if total_len is None:
            pl = os.path.join(d, f"plddt_{name}_model_{idx}.npz")
            if os.path.exists(pl):
                total_len = int(np.load(pl)["plddt"].shape[0])
    return vals, n_chains, total_len


def summarize(vals):
    out = {}
    for m in METRICS:
        a = np.array([v for v in vals[m] if not np.isnan(v)], dtype=float)
        if a.size == 0:
            out[m] = dict(Best=np.nan, Avg=np.nan, SD=np.nan, High=np.nan, Low=np.nan); continue
        best = a.min() if m in LOWER_BETTER else a.max()
        out[m] = dict(Best=best, Avg=a.mean(), SD=(a.std(ddof=1) if a.size > 1 else 0.0),
                      High=a.max(), Low=a.min())
    return out


def main():
    ap = argparse.ArgumentParser(description="Boltz -> AF3-style interaction summary CSV")
    ap.add_argument("roots", nargs="+", help="Boltz output dir(s) containing predictions/<name>/")
    ap.add_argument("-o", "--out", required=True, help="output summary CSV path")
    ap.add_argument("--af3-name", action="store_true",
                    help="format Interaction as pair_NNNN_<name> (AF3 4-field convention)")
    ap.add_argument("--per-pair-json", action="store_true",
                    help="also write an AF3-style <name>_summary_confidences.json next to each pair")
    a = ap.parse_args()

    pairs = []
    for root in a.roots:
        pairs.extend(find_pairs(root))
    pairs.sort(key=lambda x: x[0])
    if not pairs:
        raise SystemExit("No Boltz predictions/<name>/ found under: " + ", ".join(a.roots))

    rows = []
    for i, (name, d) in enumerate(pairs, 1):
        vals, n_chains, total_len = collect_samples(name, d)
        s = summarize(vals)
        interaction = f"pair_{i:04d}_{name}" if a.af3_name else name
        row = {"Interaction": interaction}
        for m in METRICS:
            for st in ("Best", "Avg", "SD", "High", "Low"):
                v = s[m][st]
                row[f"{st}_{m}"] = "" if (isinstance(v, float) and np.isnan(v)) else round(float(v), 4)
        row["Category"] = category(s["ipTM"]["Best"])
        row["N_Chains"] = n_chains if n_chains is not None else ""
        row["Total_Length"] = total_len if total_len is not None else ""
        rows.append(row)

        if a.per_pair_json:
            js = {"iptm": round(s["ipTM"]["Avg"], 4), "ptm": round(s["pTM"]["Avg"], 4),
                  "ranking_score": round(s["Ranking"]["Avg"], 4),
                  "best_iptm": round(s["ipTM"]["Best"], 4), "n_samples": len(vals["ipTM"]),
                  "category": row["Category"], "source": "boltz-2"}
            json.dump(js, open(os.path.join(d, f"{name}_summary_confidences.json"), "w"), indent=1)

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS); w.writeheader(); w.writerows(rows)
    n_samp = len(collect_samples(*pairs[0])[0]["ipTM"])
    print(f"WROTE {a.out}: {len(rows)} complexes, {n_samp} sample(s) each")
    print("Category counts:", {c: sum(1 for r in rows if r['Category'] == c) for c in
                               ("Very Weak/None", "Weak", "Moderate", "Strong")})


if __name__ == "__main__":
    main()

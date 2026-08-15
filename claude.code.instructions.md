# Claude Code instructions — set up & run Boltz-2 from this fork

Instructions for a Claude Code agent bringing up Boltz-2 on a (possibly different) machine using this
fork (`github.com/kouroshSA/boltz`). Paths below are placeholders — the other device may not be
`/home/ksa/...`; ask the user or discover paths. **Do not blindly re-download the ~8 GB of weights: the
user may have copied the `boltz_cache` folder to this machine — find and verify it first (Step 2).**

## 1. Install

```bash
# clone this fork (origin = the working version; upstream = jwohlwend/boltz)
git clone https://github.com/kouroshSA/boltz.git
cd boltz

# env — Python 3.10–3.12 (3.12 verified), needs a CUDA GPU
conda create -n boltz python=3.12 -y
conda activate boltz
pip install -e ".[cuda]"        # torch + CUDA-12 wheels + cuequivariance ops (several GB)

# sanity
python -c "import torch; print('CUDA', torch.cuda.is_available(), torch.cuda.device_count())"
boltz predict --help | head
```
NOTE: the conda env is **not** portable (compiled CUDA wheels) — it must be installed on each machine.
Only the weight cache (below) is copyable.

## 2. Locate / verify the weight cache BEFORE downloading

Boltz keeps ~8 GB of model/data files in a cache dir (default `~/.boltz`). It **only downloads a file if
that file is missing** (existence check) — so a complete pre-copied cache means **zero downloads**.
The user may have already copied a `boltz_cache/` folder to this machine.

```bash
# a) find a pre-existing cache (ask the user for the path first; else search)
find / -type f -name boltz2_conf.ckpt 2>/dev/null      # its parent dir is the cache
# b) verify it is COMPLETE — all four items must be present and full-size:
CACHE=/path/to/boltz_cache          # <-- set to the found/ given path
ls -lh "$CACHE"
#   expected:  boltz2_conf.ckpt ~2.2G   boltz2_aff.ckpt ~2.0G   mols.tar ~1.8G   mols/ (~2.2G, 45k .pkl)
[ -f "$CACHE/boltz2_conf.ckpt" ] && [ -f "$CACHE/boltz2_aff.ckpt" ] && \
[ -f "$CACHE/mols.tar" ] && [ -d "$CACHE/mols" ] && echo "cache complete -> no download" || echo "cache MISSING items -> will download"
```
- **If complete:** just `export BOLTZ_CACHE="$CACHE"` (or pass `--cache "$CACHE"`) and skip downloading.
- **If absent/incomplete:** either copy it from the other machine
  (`rsync -aP user@host:/path/to/boltz_cache/ "$CACHE"/`), or let the first `boltz predict` download the
  missing files into `$BOLTZ_CACHE` (needs internet; ~6 GB).
- **Always set `BOLTZ_CACHE`** (or `--cache`) to the intended folder, otherwise Boltz uses `~/.boltz`
  and may re-download.

## 3. Basic usage

Input = FASTA, one record per chain, header `>CHAIN_ID|ENTITY_TYPE|MSA_PATH` (leave MSA field empty to
use the MSA server). `ENTITY_TYPE` = `protein` (also dna/rna/ligand).

```bash
export BOLTZ_CACHE=/path/to/boltz_cache

# single protein (monomer)
#   monomer.fasta:  >A|protein|   \n  SEQ
boltz predict monomer.fasta --use_msa_server --out_dir OUT --cache "$BOLTZ_CACHE"

# protein–protein complex (2 chains); 5 samples to match AlphaFold-3's 5 models
#   complex.fasta:  >A|protein|\nSEQ_A  \n  >B|protein|\nSEQ_B    (homodimer = same seq twice)
boltz predict complex.fasta --use_msa_server --diffusion_samples 5 --out_dir OUT --cache "$BOLTZ_CACHE"

# batch across BOTH GPUs — put one FASTA per complex in an inputs/ dir (nothing else in it)
boltz predict inputs --use_msa_server --devices 2 --diffusion_samples 5 \
      --max_parallel_samples 1 --out_dir OUT --cache "$BOLTZ_CACHE"
```
Notes: `--devices 2` only helps with ≥2 inputs (one complex per GPU); a single complex can't be split.
`--max_parallel_samples 1` folds samples one at a time — use it for large complexes to avoid GPU OOM.
`--use_msa_server` sends sequences to the public ColabFold server; omit it and supply your own `.a3m`
(via the header MSA field) for a fully local/offline run.

Outputs land in `OUT/boltz_results_<name>/predictions/<name>/`:
`<name>_model_0..N.cif` (structures, best first) and `confidence_<name>_model_i.json`
(the interaction metric is **`iptm`**, 0–1; `ptm`/`complex_plddt` reflect monomer fold quality, not interaction).

## 4. Make outputs AF3-multimer-compatible (the lab's format)

`tools/boltz_to_af3_summary.py` converts Boltz outputs into the same per-pair schema as the
AlphaFold-3 `*_interaction_summary` CSV the lab uses (Best/Avg/SD/High/Low for ipTM, pTM, Ranking,
pLDDT, PAE + Category, N_Chains, Total_Length):

```bash
python tools/boltz_to_af3_summary.py <boltz_out_dir> [more_dirs...] -o summary.csv [--af3-name] [--per-pair-json]
```
- `<boltz_out_dir>` = any parent containing Boltz `predictions/<name>/` (single run or batch).
- `--af3-name` → format `Interaction` as `pair_NNNN_<name>` (AF3's 4-field convention; use when the
  input FASTAs are named `pair_XXXX_ORFA_ORFB` so rows join 1:1 to AF3 results).
- Category from Best_ipTM (thresholds derived from the lab's AF3 run): ≤0.40 Very Weak/None, ≤0.60
  Weak, ≤0.80 Moderate, else Strong.
- Mapping: ipTM/pTM = same as AF3; pLDDT scaled to 0–100; Ranking = Boltz `confidence_score`
  (AF3 `ranking_score` analog); PAE = mean of the Boltz PAE matrix (may differ from AF3's PAE def).
- Run the adapter with a matplotlib/pandas-capable Python if you also make figures; the adapter itself
  needs only numpy (+ the Boltz env's pandas).

## 5. More

See `Quick-start.md` in this repo for worked examples and a flags table. To pull upstream Boltz
updates: `git fetch upstream && git merge upstream/main`. Push your changes to the fork: `git push origin`.

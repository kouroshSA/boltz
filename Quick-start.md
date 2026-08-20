# Quick-start — running Boltz-2 predictions (this fork)

Practical directions for our local setup: single-protein structure prediction and
protein–protein (multimer) interaction prediction. Verified on two boxes:
**workstation** (2× RTX 3090) and **DGX Spark** (1× GB10, aarch64, CUDA 13).

## 0. One-time setup (already done on both machines)

- Repo: `/home/ksa/Models/boltz` (this fork, `kouroshSA/boltz`)
- Conda env: **`boltz`** (Python 3.12, torch 2.13 + CUDA)
- Weight cache: **`/home/ksa/Models/boltz_cache`** (~8 GB: `boltz2_conf.ckpt`, `boltz2_aff.ckpt`, `mols/`)

Activate the env — `BOLTZ_CACHE` is already exported in `~/.bashrc` and set on the conda env, so
`--cache` is optional. If you are on a fresh machine, set it before the first run or Boltz
re-downloads ~6 GB to `~/.boltz`:

```bash
conda activate boltz                      # or use /home/ksa/anaconda3/envs/boltz/bin/boltz
export BOLTZ_CACHE=/home/ksa/Models/boltz_cache
```

### Per-machine differences

Same repo, same cache, same commands — only the install differs, and only in which CUDA build you
take (`claude.code.instructions.md` §1 has the full matrix):

- **x86_64 Linux + CUDA-12 driver** (Intel/AMD workstations and servers — the 2× RTX 3090 box, and
  any A100/L40S/H100 node): the standard `pip install -e ".[cuda]"`. Nothing special.
- **DGX Spark (GB10, aarch64, CUDA 13)**: built from **conda-forge** with **cu130** torch and the
  **`cuequivariance_*_cu13`** wheels — *not* `[cuda]`, whose cu12 pins don't pair with a CUDA-13
  torch. Also: **pass `--num_workers 0` to every `boltz predict`** there (see §5), and it has one
  GPU, so `--devices 2` does not apply.

> Keep prediction outputs **out of the repo** — write to `--out_dir /home/ksa/Models/boltz_runs/...`.
> Never place the run log inside the input directory (Boltz tries to parse every file in it).

## 1. Input format (FASTA)

One record per chain. Header is `>CHAIN_ID|ENTITY_TYPE|MSA_PATH`; leave the MSA field empty to
use the MSA server. `ENTITY_TYPE` is `protein` (also supports `dna`, `rna`, `ccd`/`smiles` ligands).

## 2. Single protein (monomer)

`monomer.fasta`:
```
>A|protein|
MLSDEDFKAVFGMTRSAFANLPLWKQQNLKKEKGLF
```

Run:
```bash
export BOLTZ_CACHE=/home/ksa/Models/boltz_cache
boltz predict monomer.fasta --use_msa_server \
  --out_dir /home/ksa/Models/boltz_runs/monomer_demo --cache "$BOLTZ_CACHE"
```

## 3. Protein–protein interaction (multimer complex)

Two protein records = a 2-chain complex. Example (a MED4 training-set positive we validated,
ipTM ≈ 0.66):

`complex.fasta`:
```
>A|protein|
MLRSIFAGFFAIVLTLGLGISSVSAKTVEVKLGTDAGMLAFEPSSVTISTGDTVKFINNKLAPHNAVFDGHEELSHADLAFAPGESWEETFDTAGTFDYYCEPHRGAGMVGKVIVE
>B|protein|
MSFFESDIVQEEAKKLFTDYQELMKLGSDYGKFDREGKVMFIKKMESLMDRYKVFMKRFELSEDFQAKMTVEQLKTQLSQFGITPDQMFDQMNKTLIRMKEELDKSS
```

Run (5 models, AF3-style, so we get Best/Avg/SD ipTM):
```bash
export BOLTZ_CACHE=/home/ksa/Models/boltz_cache
boltz predict complex.fasta --use_msa_server --diffusion_samples 5 \
  --out_dir /home/ksa/Models/boltz_runs/complex_demo --cache "$BOLTZ_CACHE"
```
For a homodimer, give the same sequence twice with different chain IDs (`>A|protein|` and `>B|protein|`).

## 4. Batch of complexes across BOTH GPUs

Put one FASTA per complex in an inputs directory (nothing else in it), then use `--devices 2`
(Boltz data-parallelizes across input jobs — one complex per GPU, two at a time):

```bash
mkdir -p /home/ksa/Models/boltz_runs/batch/inputs
# ... copy pairA.fasta pairB.fasta ... into inputs/ ...
export BOLTZ_CACHE=/home/ksa/Models/boltz_cache
boltz predict /home/ksa/Models/boltz_runs/batch/inputs \
  --use_msa_server --devices 2 --diffusion_samples 5 \
  --out_dir /home/ksa/Models/boltz_runs/batch/out --cache "$BOLTZ_CACHE"
```
> A single complex cannot be split across GPUs — `--devices 2` only helps when there are ≥2 inputs.

## 5. Useful flags

| Flag | Default | Note |
|------|---------|------|
| `--diffusion_samples N` | 1 | N structures/pair, ranked by confidence (model_0 = best). Use **5** for AF3-style Best/Avg/SD. |
| `--devices N` | 1 | GPUs for data-parallel batch inference. |
| `--recycling_steps N` | 3 | More = slightly better, slower. |
| `--sampling_steps N` | 200 | Diffusion steps. |
| `--use_msa_server` | off | Generate MSAs via ColabFold server. Omit if providing your own MSA (`.a3m` in the header). |
| `--num_workers N` | 2 | Set `0` if a run stalls at `Predicting:` — **required on the DGX Spark** (see below). |
| `--out_dir PATH` | ./ | Write **outside the repo** (`/home/ksa/Models/boltz_runs/...`). |
| `--cache PATH` | `~/.boltz` | Point at `/home/ksa/Models/boltz_cache`. |
| `--override` | off | Recompute even if outputs exist. |

### Run hangs at `Predicting:`?

If a run sits at `Predicting: | 0/? [00:00<?, ?it/s]` — GPU memory allocated but ~0% utilisation and
no CPU time — the dataloader workers are stuck. Kill it and rerun with **`--num_workers 0`**.
Reproduced on the DGX Spark: default workers produced no batch in 15 min and then failed with
`DataLoader worker (pid(s) ...) exited unexpectedly`; with `--num_workers 0` the same 36-residue
monomer completed in ~6 s (pLDDT 0.955, pTM 0.816). Boltz loads one sample per job, so this costs
nothing. Tip: pipe runs to a log (`> run.log 2>&1`) — output is buffered, so a piped `tail` shows
nothing while the job is alive.

## 6. Outputs & key metrics

Results land in `<out_dir>/boltz_results_<name>/predictions/<name>/`:

- `<name>_model_0.cif` … `_model_(N-1).cif` — predicted structure(s), best first.
- `confidence_<name>_model_0.json` — confidence metrics.

For **interaction** calls, the metric that matters is **`iptm`** (interface predicted TM-score, 0–1):
higher = more confident interface. `ptm` and `complex_plddt` reflect overall/monomer fold quality
(they stay high even for non-interacting pairs, so don't use them to judge interaction).
`pair_chains_iptm` gives the per-chain-pair interface scores; `chains_ptm` the per-chain fold quality.

Rule of thumb: ipTM ≳ 0.6 = likely interface, ≲ 0.3 = unlikely; the middle is ambiguous — which is
why running `--diffusion_samples 5` (look at the max and the spread) is recommended for real calls.

## 7. AF3-style interaction summary (adapter)

Convert Boltz outputs (best with `--diffusion_samples 5`) into the same per-pair summary schema as the
AlphaFold-3 `*_interaction_summary` CSV (Best/Avg/SD/High/Low for ipTM, pTM, Ranking, pLDDT, PAE), so
Boltz results drop into the AF3 comparison pipeline:

```bash
python tools/boltz_to_af3_summary.py <boltz_out_dir> [more_dirs...] -o summary.csv \
       [--af3-name] [--per-pair-json] [--keep-legacy-cols]
```
- `<boltz_out_dir>` = any parent containing Boltz `predictions/<name>/` (single run or batch).
- `--af3-name` formats `Interaction` as `pair_NNNN_<name>` (AF3's 4-field convention).
- The `Category`, `N_Chains`, `Total_Length` columns are **dropped by default** (they are removed
  before XGBoost anyway); pass `--keep-legacy-cols` to include them.
- Category (from Best_ipTM, thresholds derived from the lab's AF3 run): ≤0.40 Very Weak/None,
  ≤0.60 Weak, ≤0.80 Moderate, else Strong.
- ipTM/pTM map exactly to AF3; pLDDT is scaled to 0–100; Ranking = Boltz `confidence_score`
  (AF3 `ranking_score` analog); PAE = mean of the Boltz PAE matrix (may differ from AF3's PAE def).

## 8. Pipelined runner for large jobs (overlap MSA + folding)

Plain `boltz predict <dir> --devices 2` generates ALL MSAs first (GPUs idle), then folds everything, so
no output appears until every MSA is done. For large sets use the pipelined runner, which drives one
async single-GPU stream per GPU over small chunks — while one stream folds, the other fetches MSAs, so
the phases overlap, GPUs stay busy, and output is written continuously. Resumable (skips done pairs).

```bash
python tools/boltz_pipeline_run.py --inputs <dir> --out <dir> --cache "$BOLTZ_CACHE" \
       --gpus 0,1 --chunk 16 --diffusion-samples 5 --max-parallel-samples 1
```
Output uses the normal Boltz layout (under boltz_results_<chunk>/predictions/<name>/), so
`tools/boltz_to_af3_summary.py <out_dir> ...` consumes it unchanged. Use `--no-msa-server` if the
FASTAs carry their own MSA paths (fully local). Two streams query the public MSA server concurrently —
fine for moderate loads; for very large jobs prefer precomputed MSAs.

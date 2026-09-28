# TCNS partial-identification certificate: minimal reproduction

Companion code for **Performance Certificates for Networked Estimation and Control with Partially Identified Sensing--Channel Dependence**. This repository contains the frozen reset-estimation, dependent-path confidence, exact colored covariance, and fixed ten-follower experiments. It contains no manuscript PDFs, source literature, proposal files, or historical projects.

## Environment and commands

The reference execution uses Python 3.9, NumPy 1.19.5, SciPy 1.6.1, Matplotlib 3.3.4, and Pillow 8.1.2. Use a Python 3.9 environment (the pinned historical wheels are not intended for Python 3.12+).

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py --check
python reproduce.py --full
```

Alternatively: `conda env create -f environment.yml`, then `conda activate tcns-repro`.

`--check` (the default) runs 21 focused checks against the small frozen reference records. It does not rerun the Monte Carlo experiments. `--full` regenerates every saved scientific numerical artifact, all six PDF/PNG figures, verifies their scientific digests, and runs the 21 checks. Generated files go to `results/` and `paper/figures/`, which are ignored by Git. Typical runtime is around one minute on the reference machine; slower machines may take longer. No external data or network connection is needed after dependencies are installed.

## Fixed experiments and expected result

The synchronized path has 1,000,000 transitions, starts at 00, and uses seed 20260928. Coverage uses 200 paths of 50,000 transitions, seed 20260930. Vehicle validation uses 20 repetitions, 10,000 burn-in plus 100,000 measured slots, seed 20261001. The existing six-state validation uses seed 20261002. No seed search is performed.

True spacing RMS is 0.1894681530 m. Its structural interval is [0.178322, 0.203682] m to the displayed precision, and its refined interval is [0.186919, 0.191794] m after 1,000,000 synchronized transitions. The exact saved energy endpoints, rather than these rounded display values, are used for verification. Final row exit counts are [335030, 413794, 102456, 148720].

`reference/manifest.json` covers all 14 scientific output files: 11 JSON files and three NPZ files. JSON hashes use sorted canonical serialization and omit only `runtime_seconds`; NPZ arrays are compared by names, dtype, shape, and array-content hashes. PDF timestamps and renderer metadata are not scientific comparison targets. Floating-point solver/BLAS differences on other platforms can change exact digests; the strict verification is validated in the reference environment, and a mismatch should be inspected rather than silently accepted.

## Contents and mapping

| File | Purpose |
|---|---|
| `src/coupling.py` | Strong transition consistency, exact binary overlap, stationary-flow LP and dual |
| `src/certificates.py` | All-prefix row confidence sets, dependent trajectories, finite-state bounds |
| `src/closed_loop.py` | Channel-tagged and independent joint-mode colored covariance solvers; fixed platoon |
| `scripts/run_experiments.py` | Frozen main experiments and saved outputs |
| `scripts/additional_checks.py` | Existing six-state and nonnormal/anisotropic validation artifacts |
| `scripts/make_figures.py` | Six figures exclusively from regenerated saved outputs |
| `tests/test_certificates.py` | 21 focused correctness checks |
| `reference/` | Small expected records, frozen controller arrays, scientific output manifest |

The larger research archive separately passed all 76 inherited checks (21 current plus 55 historical); historical drafts and their tests are deliberately outside this minimum package. The preservation check here compares the current controller to frozen arrays, avoiding an import from a historical project.

## Model contract

Marginal kernels are exact; joint dynamics satisfy strong conditional marginal consistency. Success depends only on the current channel, reset innovations are exogenous and conditionally independent, and source and fixed controller are Schur stable. RMS reports communication-induced stationary expected energy. The vehicle experiment is a fixed finite cascade; it does not establish string stability or collision safety.

The repository is private. Uploading it does not publish the paper or submit it to a journal. No public redistribution license has been added.

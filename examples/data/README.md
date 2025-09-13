
```md
# Data for AutoGMM Experiments

This directory holds small, redistribution-safe CSVs used in the paper and (optionally) scripts for fetching larger external datasets. For full descriptions and references, see the AutoGMM paper.

## Contents

- `real/` — CSVs small enough to ship with the repo and permitted for redistribution (typically derived/processed summaries used directly by the notebook).

## Datasets (as used in the paper)

### Drosophila mushroom body connectome
- **Use in paper**: Embedded via ASE and clustered (AutoGMM vs mclust).
- **Typical files**: `real/embedded_right.csv` and `real/classes.csv`
- **Source/Citation**: Priebe et al., *Semiparametric spectral modeling of the Drosophila connectome* (arXiv:1705.03297, 2017).
- **License**: Please consult the original source; only derived, redistribution-permitted CSVs are included here.

### Cancer fragmentomics cohort
- **Use in paper**: Fragmentomics features; binary labels (cancer vs normal).
- **Typical files**: `real/fragmentomics.csv`
- **Source/Citation**: Curtis et al., *PNAS* (2025), “Fragmentation signatures in cancer patients resemble those of patients with vascular or autoimmune diseases.”
- **License**: Please consult the original source; only derived, redistribution-permitted CSVs are included here.

## Checksums

For shipped CSVs, we record SHA256 checksums:

```bash
cd experiments/data
find real -type f -name "*.csv" -exec shasum -a 256 {} \; > SHA256SUMS

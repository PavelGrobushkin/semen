# SEMEN — SEgmentation for MEthylation Noise reduction

A pipeline for enhancing the tracing of age- and disease-related epigenetic changes using segmented DNA methylation data.

## Overview

SEMEN clusters CpG sites into methylation segments and uses them as features for age prediction, aiming to reduce noise and improve interpretability compared to site-level models.

## Installation

``` bash
mamba env create -f environment.yml
mamba activate semen
```

## References

-   Petkovich et al. (2017). *Using DNA Methylation Profiling to Evaluate Biological Age and Longevity Interventions.* Cell Metabolism.
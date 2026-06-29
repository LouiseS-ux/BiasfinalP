# Data Sources

## WinoBias-derived probes (`wb_` prefix, 45 pairs)

**Source dataset:** WinoBias (Zhao et al., 2018)  
**Repository:** https://github.com/uclanlp/corefBias  
**Licence:** MIT  

**Files downloaded into `data/raw/`:**

| File | Description |
|---|---|
| `wb_type1_pro.conll` | Type 1 pro-stereotypical sentences |
| `wb_type1_anti.conll` | Type 1 anti-stereotypical sentences |
| `wb_type2_pro.conll` | Type 2 pro-stereotypical sentences |
| `wb_type2_anti.conll` | Type 2 anti-stereotypical sentences |

## Original hand-crafted probes (`orig_` prefix, 105 pairs)

Original probes were also designed for this dissertation project following WinoBias's male/female variant convention, and added to the above Winobias derived data, into the final file probe_bank.json which is used in pipeline stage 1. 

---
## StereoSet training data (`dev.json`)
**Source dataset:** StereoSet intersentence (Nadeem et al., 2021)
**Repository:** https://github.com/moinnadeem/StereoSet
**Direct URL:** https://raw.githubusercontent.com/moinnadeem/StereoSet/master/data/dev.json
**Licence:** CC BY-SA 4.0
**Date accessed:** June 2026
**Usage:** Stage 3 only — RoBERTa fine-tuning training data. Filtered at runtime to gender bias_type examples (242 context items, 484 stereotype/anti-stereotype sentence pairs after excluding unrelated sentences)
*Louise Slattery — Final Project, MSc Computer Science, 2026*

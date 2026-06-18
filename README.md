# MarkupLM — SWDE Finetuning

Reproduction of the SWDE benchmark and fine-tuning for attribute extraction:

> Junlong Li, Yiheng Xu, Lei Cui, Furu Wei.  
> **MarkupLM: Pre-Training of Text and Markup Language for Visually-Rich Document Understanding.**  
> ACL 2022.

This repository is part of a PhD thesis on information extraction from French cultural event web pages (Ideactiv dataset), conducted at the University of Nantes.

---

## Contents

| File | Description |
|------|-------------|
| `markuplm_finetune_swde.py` | Fine-tuning script on SWDE (10-fold cross-validation) |
| `markuplm_infer_events.py` | Inference script on Ideactiv pages |
| `run_markuplm_swde.slurm` | SLURM job script for Jean Zay (GPU V100) |
| `requirements.txt` | Python dependencies |

---

## Experimental Protocol

Exact reproduction of the protocol from Li et al. (2022):

- **10-fold** leave-n-sites-out cross-validation
- **k** training sites per fold (k = 1, 2, 3, 5)
- Up to 2,000 pages per site
- 4 previous context nodes
- Metric: page-level precision / recall / **F1**

### Hyperparameters (paper, Appendix Table 9)

| Parameter | Value |
|-----------|-------|
| `per_gpu_train_batch_size` | 32 |
| `num_train_epochs` | 10 |
| `learning_rate` | 2e-5 |
| `warmup_ratio` | 0.1 |
| `max_seq_length` | 384 |
| `doc_stride` | 128 |

---

## Results on SWDE (GPU V100, Jean Zay — IDRIS)

### F1 per vertical and k

| Vertical | k=1 | k=2 | k=3 | k=5 |
|----------|-----|-----|-----|-----|
| auto | 71.42 | 84.90 | 90.89 | — |
| book | 80.24 | 85.17 | 89.17 | — |
| camera | 85.36 | 92.36 | 93.80 | 95.01 |
| job | 78.90 | 88.40 | 87.67 | 91.37 |
| movie | 88.15 | 93.84 | 97.89 | 99.15 |
| nbaplayer | 89.24 | 92.30 | 95.21 | 95.91 |
| restaurant | 77.75 | 90.17 | 95.44 | 97.32 |
| university | 87.56 | 96.62 | 97.48 | 98.93 |
| **Average** | **82.33** | **90.47** | **93.44** | **95.51*** |

*\* Average over 6 verticals (auto and book not included in the k=5 run)*

Comparison with the paper (Li et al., 2022, Table 3):

| k | Our F1 | Paper F1 |
|---|--------|----------|
| 1 | 82.33 % | 82.11 % |
| 2 | 90.47 % | 91.29 % |

---

## Usage

### Requirements

```bash
pip install -r requirements.txt
```

Clone the official MarkupLM repository:

```bash
git clone https://github.com/microsoft/unilm.git unilm_repo
```

Download and extract the SWDE dataset into `SWDE/`.

### Fine-tuning (step 1: data preparation + training)

```bash
python markuplm_finetune_swde.py \
    --swde_path      SWDE \
    --processed_path SWDE/processed \
    --output_dir     results_markuplm \
    --cuda           0
```

### Fine-tuning (step 2: training only, data already prepared)

```bash
python markuplm_finetune_swde.py \
    --swde_path      SWDE \
    --processed_path SWDE/processed \
    --output_dir     results_markuplm \
    --skip_data_prep \
    --verticals      movie restaurant
```

### On Jean Zay (SLURM)

```bash
sbatch run_markuplm_swde.slurm
```

### Inference on Ideactiv data

```bash
python markuplm_infer_events.py \
    --checkpoint results_markuplm/restaurant/fodors/checkpoint \
    --html_dir   data_by_domain
```

---
---

## Reference

```bibtex
@inproceedings{li2022markuplm,
  title     = {MarkupLM: Pre-Training of Text and Markup Language for Visually-Rich Document Understanding},
  author    = {Li, Junlong and Xu, Yiheng and Cui, Lei and Wei, Furu},
  booktitle = {Proceedings of ACL 2022},
  year      = {2022}
}
```

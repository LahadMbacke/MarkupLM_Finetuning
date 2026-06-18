# MarkupLM — SWDE Finetuning

Reproduction et extension du benchmark SWDE du papier :

> Junlong Li, Yiheng Xu, Lei Cui, Furu Wei.  
> **MarkupLM: Pre-Training of Text and Markup Language for Visually-Rich Document Understanding.**  
> ACL 2022.

Ce dépôt fait partie d'une thèse sur l'extraction d'information à partir de pages web d'événements culturels français (données Ideactiv), réalisée à l'Université de Nantes.

---

## Contenu

| Fichier | Description |
|--------|-------------|
| `markuplm_finetune_swde.py` | Script de fine-tuning sur SWDE (validation croisée 10 folds) |
| `markuplm_infer_events.py` | Script d'inférence sur les pages Ideactiv |
| `run_markuplm_swde.slurm` | Job SLURM pour Jean Zay (GPU V100) |
| `requirements.txt` | Dépendances Python |
| `swde_experiment.log` | Log des expériences |

---

## Protocole expérimental

Reproduction exacte du protocole de Li et al. (2022) :

- Validation croisée **10 folds** (*leave-n-sites-out*)
- **k** sites d'entraînement par fold (k = 1, 2, 3, 5)
- 2 000 pages max par site
- 4 nœuds de contexte précédents
- Métrique : précision / rappel / **F1 page-level**

### Hyperparamètres (papier, Annexe Table 9)

| Paramètre | Valeur |
|-----------|--------|
| `per_gpu_train_batch_size` | 32 |
| `num_train_epochs` | 10 |
| `learning_rate` | 2e-5 |
| `warmup_ratio` | 0.1 |
| `max_seq_length` | 384 |
| `doc_stride` | 128 |

---

## Résultats SWDE (GPU V100, Jean Zay — IDRIS)

### F1 par vertical et par k

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
| **Moyenne** | **82.33** | **90.47** | **93.44** | **95.51*** |

*\* Moyenne sur 6 verticales (auto et book non incluses dans le run k=5)*

Comparaison avec le papier (Li et al., 2022, Table 3) :

| k | Notre F1 moyen | Papier F1 moyen |
|---|---------------|-----------------|
| 1 | 82.33 % | 82.11 % |
| 2 | 90.47 % | 91.29 % |

---

## Utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

Cloner le dépôt officiel MarkupLM :

```bash
git clone https://github.com/microsoft/unilm.git unilm_repo
```

Télécharger et décompresser le dataset SWDE dans `SWDE/`.

### Fine-tuning (étape 1 : préparation des données + entraînement)

```bash
python markuplm_finetune_swde.py \
    --swde_path      SWDE \
    --processed_path SWDE/processed \
    --output_dir     results_markuplm \
    --cuda           0
```

### Fine-tuning (étape 2 : entraînement seul, données déjà préparées)

```bash
python markuplm_finetune_swde.py \
    --swde_path      SWDE \
    --processed_path SWDE/processed \
    --output_dir     results_markuplm \
    --skip_data_prep \
    --verticals      movie restaurant
```

### Sur Jean Zay (SLURM)

```bash
sbatch run_markuplm_swde.slurm
```

### Inférence sur les données Ideactiv

```bash
python markuplm_infer_events.py \
    --checkpoint results_markuplm/restaurant/fodors/checkpoint \
    --html_dir   data_by_domain
```

---

## Contexte de thèse

Ce benchmark SWDE constitue la **baseline supervisée** avant transfert vers les données Ideactiv (événements culturels français). Le test de transfert direct (*zero-shot* SWDE → Ideactiv) confirme l'absence de généralisation : le modèle entraîné sur des restaurants américains ne reconnaît pas les attributs de pages institutionnelles françaises.

L'étape suivante est un fine-tuning direct sur les données Ideactiv annotées.

---

## Référence

```bibtex
@inproceedings{li2022markuplm,
  title     = {MarkupLM: Pre-Training of Text and Markup Language for Visually-Rich Document Understanding},
  author    = {Li, Junlong and Xu, Yiheng and Cui, Lei and Wei, Furu},
  booktitle = {Proceedings of ACL 2022},
  year      = {2022}
}
```


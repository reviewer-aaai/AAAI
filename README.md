# Équivariance et robustesse géométrique sur MNIST

Code d'évaluation accompagnant l'article. Il compare quatre architectures sous
diverses transformations géométriques (rotations 2D, transformations de Möbius,
mise à l'échelle, projectif, et échantillonnage aléatoire de SU(2)), et mesure
à la fois **l'équivariance** des représentations et la **robustesse** de la
classification.

> Ce dépôt **n'entraîne pas** les modèles : il suppose que les checkpoints sont
> déjà disponibles et régénère l'ensemble des figures et tables du papier.

## Modèles comparés

| Tag          | Architecture                                   | Symétrie visée              |
|--------------|------------------------------------------------|-----------------------------|
| `Classic`    | ResNet standard (baseline)                     | aucune                      |
| `Sim2Only`   | ResNet log-polaire                             | similitudes Sim(2)          |
| `SU2Stereo`  | ResNet sphérique / projection stéréographique  | SU(2) / SO(3)               |
| `BiLogPolar` | ResNet bi-log-polaire de Möbius                | Möbius (φ, ψ₁, ψ₂)          |

Toutes les architectures sont calibrées automatiquement à ~500 k paramètres pour
une comparaison à budget égal.

## Structure du dépôt

```
.
├── evaluate.py            # point d'entrée (évaluation, sans entraînement)
├── requirements.txt
├── checkpoints/           # modèles pré-entraînés (à fournir)
│   ├── full_Classic.pth
│   ├── full_Sim2Only.pth
│   ├── full_SU2Stereo.pth
│   └── full_BiLogPolar.pth
├── results/               # créé automatiquement : figures + tables
└── src/
    ├── models.py          # ClassicResNet, Sim2OnlyResNet,
    │                        SU2SphericalResNetSO3, BiLogPolarMöbiusResNet
    ├── transforms_geom.py # apply_rotation_2d, apply_mobius_phi,
    │                        apply_mobius_combo, apply_scale,
    │                        apply_projective, make_radial_images
    ├── equivariance.py    # measure_equivariance, print_equivariance_table,
    │                        plot_equivariance
    ├── grids.py           # _build_phi_grids_manual, _build_psi1_grids,
    │                        _build_scale_grids, _build_projective_grids,
    │                        _build_combo_grids, _build_su2_random_grids
    ├── robustness.py      # sweep_all_models, save_robustness_tables,
    │                        print_robustness_table, plot_robustness_curves,
    │                        visualize_transformations
    └── utils.py           # find_model_width, count_parameters,
                             MODEL_COLORS, MODEL_LABELS
```

Si ton code source est actuellement dans un unique fichier, deux options :

1. répartir les définitions dans `src/` selon le tableau ci-dessus (recommandé) ;
2. ou remplacer, en tête de `evaluate.py`, les lignes `from src.… import …`
   par `from mon_fichier import *`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

Un GPU CUDA est recommandé mais non obligatoire (le script bascule
automatiquement sur CPU si aucun GPU n'est disponible).

## Checkpoints

Placez les quatre fichiers `full_<TAG>.pth` dans `checkpoints/`. Le script
reconstruit chaque modèle à la largeur calibrée puis charge le `state_dict`
(le préfixe `_orig_mod.` éventuellement ajouté par `torch.compile` lors de
l'entraînement est retiré automatiquement).

Si un checkpoint manque, le script s'arrête avec un message explicite.

## Utilisation

```bash
python evaluate.py \
    --checkpoints_dir ./checkpoints \
    --output_dir      ./results \
    --data_dir        ./data \
    --device          auto
```

Options :

| Argument            | Défaut          | Description                                    |
|---------------------|-----------------|------------------------------------------------|
| `--checkpoints_dir` | `./checkpoints` | Dossier des modèles pré-entraînés.             |
| `--output_dir`      | `./results`     | Dossier de sortie (figures, tables).           |
| `--data_dir`        | `./data`        | Cache MNIST (téléchargé si absent).            |
| `--device`          | `auto`          | `auto`, `cuda` ou `cpu`.                        |
| `--num_workers`     | `2`             | Workers du `DataLoader`.                        |
| `--seed`            | `42`            | Graine (reproductibilité de la Phase 1).       |

## Déroulé du script

1. **Calibration des largeurs** — ajuste chaque architecture à ~500 k paramètres.
2. **Phase 1 — équivariance à poids aléatoires** — mesure l'équivariance
   intrinsèque des architectures avant tout apprentissage.
3. **Chargement des modèles pré-entraînés.**
4. **Phase 2 — équivariance après entraînement** — même mesure sur les modèles
   chargés, avec figure comparative aléatoire vs entraîné.
5. **Phase 3 — sweep de robustesse** — balayage de φ, ψ₁, scale, projectif, combo
   φ+ψ₁ et SU(2) aléatoire, avec visualisation des transformations, tables et
   courbes.

## Fichiers générés (dans `--output_dir`)

- `equivariance_random.png` — équivariance à poids aléatoires (Phase 1).
- `equivariance_comparison.png` — comparatif aléatoire vs entraîné.
- Visualisations des transformations géométriques appliquées aux images.
- Tables de robustesse (CSV) et courbes de robustesse (PNG) par famille de
  transformation.

## Reproductibilité

Les graines (`torch`, `numpy`, échantillonnage SU(2)) sont fixées via `--seed`.
Le transform de validation est déterministe (`Resize(64)` + normalisation,
sans augmentation) afin que la classification de référence ne dépende que de la
transformation géométrique évaluée.

## Citation

```bibtex
@inproceedings{VOTRE_CLE,
  title     = {TITRE DE L'ARTICLE},
  author    = {AUTEURS},
  booktitle = {CONFÉRENCE},
  year      = {ANNÉE}
}
```

## Licence

Ajoutez ici la licence de votre choix (par ex. MIT) et le fichier `LICENSE`
correspondant.

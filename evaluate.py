#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Évaluation de l'équivariance et de la robustesse géométrique sur MNIST.

Ce script suppose que les modèles sont DÉJÀ ENTRAÎNÉS : il charge les
checkpoints `full_<TAG>.pth` depuis --checkpoints_dir et n'effectue AUCUN
entraînement. Il reproduit :

  * Phase 1 — mesure d'équivariance à poids aléatoires
  * Phase 2 — mesure d'équivariance sur les modèles entraînés (chargés)
  * Phase 3 — sweep de robustesse (φ, ψ₁, scale, projectif, combo, SU(2))

et régénère l'ensemble des figures et tables.

Structure de dépôt attendue (voir README.md) :

    .
    ├── evaluate.py            # ce fichier
    ├── requirements.txt
    ├── checkpoints/           # full_Classic.pth, full_Sim2Only.pth, ...
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

Si ton code est aujourd'hui dans un seul fichier, tu peux soit le répartir
dans `src/` comme ci-dessus, soit remplacer les imports par
`from mon_fichier import *`.
"""

import os
import gc
import math
import argparse

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # backend headless (serveurs sans affichage)
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

# ── Imports du projet (voir docstring / README pour la structure) ────────────
from src.models import (
    ClassicResNet,
    Sim2OnlyResNet,
    SU2SphericalResNetSO3,
    BiLogPolarMöbiusResNet,
)
from src.transforms_geom import (
    make_radial_images,
    apply_rotation_2d,
    apply_mobius_phi,
    apply_mobius_combo,
    apply_scale,
    apply_projective,
)
from src.equivariance import (
    measure_equivariance,
    print_equivariance_table,
    plot_equivariance,
)
from src.grids import (
    _build_phi_grids_manual,
    _build_psi1_grids,
    _build_scale_grids,
    _build_projective_grids,
    _build_combo_grids,
    _build_su2_random_grids,
)
from src.robustness import (
    sweep_all_models,
    save_robustness_tables,
    print_robustness_table,
    plot_robustness_curves,
    visualize_transformations,
)
from src.utils import (
    find_model_width,
    count_parameters,
    MODEL_COLORS,
    MODEL_LABELS,
)


# ═════════════════════════════════════════════════════════════════════════════
# Adaptation MNIST : 3→1 canal sur la première convolution
# ═════════════════════════════════════════════════════════════════════════════
def make_mnist_model(base_factory, wm):
    """Construit le modèle et remplace la première conv 3→C par 1→C."""
    model = base_factory(wm)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            new_conv = nn.Conv2d(
                in_channels=1,
                out_channels=module.out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
            )
            parent = model
            parts = name.split(".")
            for part in parts[:-1]:
                parent = getattr(parent, part)
            setattr(parent, parts[-1], new_conv)
            break  # une seule première conv à patcher
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Arguments de ligne de commande
# ═════════════════════════════════════════════════════════════════════════════
def parse_args():
    p = argparse.ArgumentParser(
        description="Évaluation équivariance/robustesse MNIST (sans entraînement)."
    )
    p.add_argument("--checkpoints_dir", default="./checkpoints",
                   help="Dossier contenant full_<TAG>.pth (modèles pré-entraînés).")
    p.add_argument("--output_dir", default="./results",
                   help="Dossier de sortie pour figures et tables.")
    p.add_argument("--data_dir", default="./data",
                   help="Dossier de téléchargement/cache MNIST.")
    p.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"],
                   help="Périphérique de calcul.")
    p.add_argument("--num_workers", type=int, default=2,
                   help="Nombre de workers pour le DataLoader.")
    p.add_argument("--seed", type=int, default=42,
                   help="Graine aléatoire (reproductibilité de la Phase 1).")
    return p.parse_args()


def resolve_device(choice):
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def load_pretrained(factory, wm, ckpt_path, device):
    """Charge un checkpoint et reconstruit le modèle à la bonne largeur."""
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint introuvable : {ckpt_path}\n"
            f"Ce script suppose que les modèles sont déjà entraînés. "
            f"Placez les fichiers full_<TAG>.pth dans --checkpoints_dir."
        )
    model = factory(wm).to(device)
    state = torch.load(ckpt_path, map_location=device)
    # Nettoie le préfixe ajouté par torch.compile lors de l'entraînement.
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model


# ═════════════════════════════════════════════════════════════════════════════
# Programme principal
# ═════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = resolve_device(args.device)
    save_dir = args.output_dir
    ckpt_dir = args.checkpoints_dir
    os.makedirs(save_dir, exist_ok=True)
    print(f"🖥️  Device  → {device}")
    print(f"📦 Checkpoints ← {ckpt_dir}/")
    print(f"📁 Outputs → {save_dir}/")

    TAGS = ["Classic", "Sim2Only", "SU2Stereo", "BiLogPolar"]

    # ── Configuration architecture / géométrie ───────────────────────────────
    TARGET_PARAMS = 500_000
    N_BLOCKS      = (1, 1, 2, 1)
    LP_H, LP_W    = 28, 28              # MNIST : 28×28
    MAX_K, S_VALUES = 2, [-1., -0.5, 0., 0.5, 1.]
    NLAT, NLON, LMAX = 32, 64, 16      # noqa: F841 (référence de config)

    kw_bilog = dict(lp_h=32, lp_w=48, n_blocks=(1, 1, 2, 1))

    # ── Transform de validation (déterministe, sans augmentation) ─────────────
    MEAN, STD = (0.1307,), (0.3081,)
    transform_val = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

    full_val = datasets.MNIST(args.data_dir, train=False, download=True,
                              transform=transform_val)
    val_loader = DataLoader(full_val, batch_size=128, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    print(f"✅ Val : {len(full_val)} images")

    # ── Factories ─────────────────────────────────────────────────────────────
    kw_equi = dict(n_blocks=N_BLOCKS, max_k=MAX_K, s_values=S_VALUES,     # noqa: F841
                   lp_h=LP_H, lp_w=LP_W)
    kw_s2 = dict(n_blocks=(2, 2, 2), nlat=16, nlon=32, lmax=8, in_channels=1)
    kw_sim2 = dict(lp_h=32, lp_w=48, n_blocks=(1, 1, 2, 1),
                   max_k=2, s_values=[0., 1.])

    factories = {
        "Classic":   lambda wm: make_mnist_model(
                        lambda w: ClassicResNet(width_mult=w, n_blocks=N_BLOCKS), wm),
        "Sim2Only":  lambda wm: make_mnist_model(
                        lambda w: Sim2OnlyResNet(width_mult=w, **kw_sim2), wm),
        "SU2Stereo": lambda wm: make_mnist_model(
                        lambda w: SU2SphericalResNetSO3(width_mult=w, **kw_s2), wm),
        "BiLogPolar": lambda wm: make_mnist_model(
                        lambda w: BiLogPolarMöbiusResNet(width_mult=w, **kw_bilog), wm),
    }

    # ── Calibration des largeurs (nécessaire pour reconstruire les modèles) ───
    print("\n🔍 Calibration des largeurs...")
    wm = {}
    for tag, factory in factories.items():
        wm[tag] = find_model_width(factory, target=TARGET_PARAMS)
        m_tmp = factory(wm[tag])
        print(f"  [{tag:<12}] wm={wm[tag]:.4f} → {count_parameters(m_tmp):,} params")
        del m_tmp
        gc.collect()

    # ═════════════════════════════════════════════════════════════════════════
    # Familles de transformations pour la mesure d'équivariance
    # ═════════════════════════════════════════════════════════════════════════
    x_test   = torch.randn(8, 1, 28, 28, device=device)          # 1 canal
    x_radial = make_radial_images(8, size=28, device=device)

    transforms_dict = {
        "Rotation 2D": [
            ("15°", lambda x: apply_rotation_2d(x, 15)),
            ("30°", lambda x: apply_rotation_2d(x, 30)),
            ("45°", lambda x: apply_rotation_2d(x, 45)),
            ("60°", lambda x: apply_rotation_2d(x, 60)),          # max 60° (évite 6→9)
        ],
        "Möbius φ (θ≠0 réel)": [
            ("φ=6°",  lambda x: apply_mobius_phi(x, math.radians(6))),
            ("φ=13°", lambda x: apply_mobius_phi(x, math.radians(13))),
            ("φ=19°", lambda x: apply_mobius_phi(x, math.radians(19))),
            ("φ=32°", lambda x: apply_mobius_phi(x, math.radians(32))),
            ("φ=45°", lambda x: apply_mobius_phi(x, math.radians(45))),
        ],
        "Möbius φ (θ=0 radial)": [
            ("φ=6°",  lambda x: apply_mobius_phi(x, math.radians(6))),
            ("φ=13°", lambda x: apply_mobius_phi(x, math.radians(13))),
            ("φ=19°", lambda x: apply_mobius_phi(x, math.radians(19))),
            ("φ=32°", lambda x: apply_mobius_phi(x, math.radians(32))),
            ("φ=45°", lambda x: apply_mobius_phi(x, math.radians(45))),
        ],
        "Scale": [
            ("0.7×",  lambda x: apply_scale(x, 0.7)),
            ("0.85×", lambda x: apply_scale(x, 0.85)),
            ("1.2×",  lambda x: apply_scale(x, 1.2)),
            ("1.5×",  lambda x: apply_scale(x, 1.5)),
            ("2.0×",  lambda x: apply_scale(x, 2.0)),
        ],
        "Projectif": [
            ("c=0.001", lambda x: apply_projective(x, 0.001, 0.001)),
            ("c=0.003", lambda x: apply_projective(x, 0.003, 0.003)),
            ("c=0.005", lambda x: apply_projective(x, 0.005, 0.005)),
        ],
        "Möbius φ+ψ₁": [
            ("φ=6°,ψ=15°",  lambda x: apply_mobius_combo(x, math.radians(6),  math.radians(15))),
            ("φ=13°,ψ=30°", lambda x: apply_mobius_combo(x, math.radians(13), math.radians(30))),
            ("φ=19°,ψ=45°", lambda x: apply_mobius_combo(x, math.radians(19), math.radians(45))),
            ("φ=32°,ψ=60°", lambda x: apply_mobius_combo(x, math.radians(32), math.radians(60))),
            ("φ=45°,ψ=60°", lambda x: apply_mobius_combo(x, math.radians(45), math.radians(60))),
        ],
    }
    x_per_family = {"Möbius φ (θ=0 radial)": x_radial}

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 1 — ÉQUIVARIANCE (poids aléatoires)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 1 — ÉQUIVARIANCE (poids aléatoires)")
    print("═" * 60)

    models_rand = {tag: factories[tag](wm[tag]).to(device) for tag in TAGS}
    eq_results = measure_equivariance(models_rand, x_test, transforms_dict, x_per_family)
    print_equivariance_table(eq_results)
    plot_equivariance(eq_results, save_dir, suffix="_random")
    del models_rand
    gc.collect()
    empty_cache(device)

    # ═════════════════════════════════════════════════════════════════════════
    # CHARGEMENT DES MODÈLES PRÉ-ENTRAÎNÉS (remplace la phase d'entraînement)
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  CHARGEMENT DES MODÈLES PRÉ-ENTRAÎNÉS")
    print("═" * 60)

    trained_models = {}
    for tag in TAGS:
        path = os.path.join(ckpt_dir, f"full_{tag}.pth")
        model = load_pretrained(factories[tag], wm[tag], path, device)
        trained_models[tag] = model
        print(f"  [{tag:<12}] ✅ chargé ← {path}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 2 — ÉQUIVARIANCE APRÈS ENTRAÎNEMENT
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 2 — ÉQUIVARIANCE (modèles entraînés)")
    print("═" * 60)

    eq_trained = measure_equivariance(trained_models, x_test,
                                      transforms_dict, x_per_family)
    print_equivariance_table(eq_trained)

    # ── Figure comparative aléatoire vs entraîné ──────────────────────────────
    fig, axes = plt.subplots(2, len(transforms_dict),
                             figsize=(5 * len(transforms_dict), 9))
    fig.suptitle("Équivariance MNIST : aléatoire (haut) vs entraîné (bas)",
                 fontsize=12, fontweight="bold")
    for col, t_name in enumerate(transforms_dict.keys()):
        for row, (res, title) in enumerate([(eq_results, "Aléatoire"),
                                             (eq_trained, "Entraîné")]):
            ax = axes[row, col]
            n_s = len(res[TAGS[0]][t_name])
            xpos = np.arange(n_s)
            bw = 0.8 / len(TAGS)
            xlbl = [l for l, _ in res[TAGS[0]][t_name]]
            for ti, tag in enumerate(TAGS):
                errs = [e for _, e in res[tag][t_name]]
                ax.bar(xpos + ti * bw - 0.4 + bw / 2, errs, bw * 0.9,
                       color=MODEL_COLORS.get(tag, "gray"), alpha=0.85,
                       label=MODEL_LABELS.get(tag, tag), edgecolor="white", lw=0.5)
            ax.axhline(15., color="green", ls="--", lw=1.2, alpha=0.6)
            ax.axhline(40., color="red",   ls="--", lw=1.2, alpha=0.6)
            ax.set_xticks(xpos + 0.4 - bw / 2)
            ax.set_xticklabels(xlbl, fontsize=7, rotation=20, ha="right")
            ax.set_title(f"{t_name} — {title}", fontsize=9)
            ax.set_ylabel("Erreur (%)")
            ax.grid(True, alpha=0.3, axis="y")
            if row == 0 and col == 0:
                ax.legend(fontsize=7)
    plt.tight_layout()
    path_fig = os.path.join(save_dir, "equivariance_comparison.png")
    plt.savefig(path_fig, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"💾 Comparaison → {path_fig}")

    # ═════════════════════════════════════════════════════════════════════════
    # PHASE 3 — SWEEP ROBUSTESSE
    # ═════════════════════════════════════════════════════════════════════════
    print("\n" + "═" * 60)
    print("  PHASE 3 — SWEEP ROBUSTESSE")
    print("═" * 60)

    IMG_SIZE = 64
    lin      = torch.linspace(-1, 1, IMG_SIZE, device=device)
    grid_r2  = torch.stack(torch.meshgrid(lin, lin, indexing="xy"))

    # ── φ étendu à 90° ────────────────────────────────────────────────────────
    phi_values_rad = np.linspace(0, math.pi / 2, 20)
    phi_grids      = _build_phi_grids_manual(phi_values_rad, grid_r2)

    # ── ψ₁ étendu à 90° ───────────────────────────────────────────────────────
    psi1_rad, psi1_deg, psi1_grids = _build_psi1_grids(20, grid_r2,
                                                       phi_fixed=0., max_deg=90.)

    # ── Scale agressif ────────────────────────────────────────────────────────
    SWEEP_SCALE = [0.5, 0.7, 0.85, 1.0, 1.5, 2.0, 3.0]
    scale_grids, scale_arr = _build_scale_grids(SWEEP_SCALE, IMG_SIZE, IMG_SIZE, device)

    # ── Projectif ─────────────────────────────────────────────────────────────
    proj_grids = _build_projective_grids(40, IMG_SIZE, IMG_SIZE, device,
                                         mnist_safe=False)

    # ── Combo φ+ψ₁ ────────────────────────────────────────────────────────────
    N_COMBO        = 20
    phi_combo_rad  = np.linspace(0, math.pi / 2, N_COMBO)
    psi1_combo_rad = np.linspace(0, math.pi / 2, N_COMBO)
    combo_grids    = _build_combo_grids(phi_combo_rad, psi1_combo_rad, grid_r2)
    combo_labels   = [f"φ={math.degrees(p):.0f}°\nψ={math.degrees(s):.0f}°"
                      for p, s in zip(phi_combo_rad, psi1_combo_rad)]

    # ── SU(2) Haar ────────────────────────────────────────────────────────────
    su2_grids, phi_su2, psi1_su2, psi2_su2 = _build_su2_random_grids(
        n_samples=32,
        grid_r2=grid_r2,
        max_phi=math.pi / 2,
        max_psi1=math.pi / 2,
        max_psi2=math.pi / 2,
        seed=args.seed,
    )

    phi_labels   = [f"φ={math.degrees(v):.0f}°" for v in phi_values_rad]
    psi1_labels  = [f"ψ₁={d:.0f}°"              for d in psi1_deg]
    scale_labels = [f"s={v:.2f}"                for v in SWEEP_SCALE]
    proj_labels  = [f"p={i}"                    for i in range(len(proj_grids))]
    su2_labels   = [
        f"φ={math.degrees(phi_su2[i]):.0f}°\n"
        f"ψ₁={math.degrees(psi1_su2[i]):.0f}°\n"
        f"ψ₂={math.degrees(psi2_su2[i]):.0f}°"
        for i in range(len(phi_su2))
    ]

    visualize_transformations(
        val_loader,
        phi_grids=phi_grids,
        psi1_grids=psi1_grids,
        scale_grids=scale_grids,
        proj_grids=proj_grids,
        combo_grids=combo_grids,
        su2_grids=su2_grids[:32],
        phi_labels=phi_labels,
        psi1_labels=psi1_labels,
        scale_labels=scale_labels,
        proj_labels=proj_labels,
        combo_labels=combo_labels,
        su2_labels=su2_labels[:32],
        save_dir=save_dir,
        dataset="mnist",
    )

    rob_results = sweep_all_models(
        trained_models, val_loader,
        phi_grids, psi1_grids, scale_grids, proj_grids,
        combo_grids=combo_grids,
        su2_grids=su2_grids,
    )

    save_robustness_tables(
        rob_results, phi_values_rad, psi1_deg, scale_arr,
        phi_combo_rad=phi_combo_rad,
        psi1_combo_rad=psi1_combo_rad,
        save_dir=save_dir,
    )
    print_robustness_table(phi_values_rad, psi1_deg, scale_arr, rob_results,
                           phi_combo_rad=phi_combo_rad,
                           psi1_combo_rad=psi1_combo_rad)
    plot_robustness_curves(phi_values_rad, psi1_deg, scale_arr, rob_results, save_dir,
                           phi_combo_rad=phi_combo_rad,
                           psi1_combo_rad=psi1_combo_rad)

    print(f"\n✅ Terminé → {save_dir}/")


if __name__ == "__main__":
    main()

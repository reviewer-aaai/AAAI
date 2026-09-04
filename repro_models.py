"""
repro_models.py
═════════════════════════════════════════════════════════════════════════════
Pont entre le notebook d'entraînement et le notebook reviewer.

Le notebook d'entraînement est la source unique des architectures : ce module
exécute ses cellules de définition et réexpose les fabriques, sans dupliquer
une seule ligne de modèle. Aucun entraînement n'est déclenché (main() est
protégé par le garde __name__).

Usage :
    import repro_models
    model = repro_models.build("SU2Stereo", width_mult=0.7314)
    model = repro_models.load("SU2Stereo", "checkpoints/mnist_v10")

À placer à la racine du dépôt, à côté du notebook d'entraînement.
"""

import glob
import json
import os

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Localisation du notebook d'entraînement
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOK = os.environ.get("REPRO_TRAINING_NOTEBOOK")
if NOTEBOOK is None:
    _cands = (glob.glob(os.path.join(_HERE, "Ablation_SU2xSim2_MNIST_V10.ipynb"))
              or glob.glob(os.path.join(_HERE, "notebooks", "Ablation_*.ipynb"))
              or glob.glob(os.path.join(_HERE, "Ablation_*.ipynb")))
    if not _cands:
        raise FileNotFoundError(
            "Notebook d'entraînement introuvable. Placez-le à la racine du dépôt "
            "ou renseignez la variable d'environnement REPRO_TRAINING_NOTEBOOK.")
    NOTEBOOK = _cands[0]


def _exec_notebook(path):
    """Exécute les cellules de code du notebook dans un espace de noms isolé.

    Les lignes magiques (!pip, %cd…) sont retirées, et __name__ est fixé de
    sorte que le  if __name__ == "__main__": main()  final ne se déclenche pas.
    """
    with open(path, encoding="utf-8") as f:
        nb = json.load(f)

    ns = {"__name__": "repro_models_notebook", "__file__": path}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(l for l in src.split("\n")
                        if not l.lstrip().startswith(("!", "%")))
        if not src.strip():
            continue
        try:
            exec(compile(src, f"{os.path.basename(path)}[cell {i}]", "exec"), ns)
        except Exception as e:
            raise RuntimeError(
                f"Échec à la cellule {i} de {path} : {type(e).__name__}: {e}") from e
    return ns


_NS = _exec_notebook(NOTEBOOK)

# Symboles requis, réexposés pour usage direct
ClassicResNet          = _NS["ClassicResNet"]
Sim2OnlyResNet         = _NS["Sim2OnlyResNet"]
SU2SphericalResNetSO3  = _NS["SU2SphericalResNetSO3"]
BiLogPolarMöbiusResNet = _NS["BiLogPolarMöbiusResNet"]
make_mnist_model       = _NS["make_mnist_model"]
count_parameters       = _NS["count_parameters"]

# ─────────────────────────────────────────────────────────────────────────────
#  Configuration des fabriques
#  Recopiée à l'identique depuis main(). Toute divergence ferait échouer
#  load_state_dict, donc ce bloc et celui du notebook doivent rester alignés.
# ─────────────────────────────────────────────────────────────────────────────
N_BLOCKS = (1, 1, 2, 1)

KW_S2 = dict(n_blocks=(2, 2, 2), nlat=16, nlon=32, lmax=8, in_channels=1)
KW_SIM2 = dict(lp_h=32, lp_w=48, n_blocks=(1, 1, 2, 1),
               max_k=2, s_values=[0., 1.])
KW_BILOG = dict(lp_h=32, lp_w=48, n_blocks=(1, 1, 2, 1))

FACTORIES = {
    "Classic":    lambda wm: make_mnist_model(
        lambda w: ClassicResNet(width_mult=w, n_blocks=N_BLOCKS), wm),
    "Sim2Only":   lambda wm: make_mnist_model(
        lambda w: Sim2OnlyResNet(width_mult=w, **KW_SIM2), wm),
    "SU2Stereo":  lambda wm: make_mnist_model(
        lambda w: SU2SphericalResNetSO3(width_mult=w, **KW_S2), wm),
    "BiLogPolar": lambda wm: make_mnist_model(
        lambda w: BiLogPolarMöbiusResNet(width_mult=w, **KW_BILOG), wm),
}

TAGS = list(FACTORIES)


def build(tag, width_mult):
    """Instancie l'architecture `tag` à la largeur donnée, sans poids entraînés.

    `width_mult` provient du manifeste : il fige la calibration à 500k
    paramètres faite par find_model_width au moment de l'entraînement.
    """
    if tag not in FACTORIES:
        raise KeyError(f"Modèle inconnu : {tag!r}. Attendu : {TAGS}")
    if width_mult is None:
        raise ValueError(
            f"[{tag}] width_mult manquant. Il doit venir de manifest.json ; "
            f"le recalculer risquerait de produire des dimensions différentes "
            f"de celles du checkpoint.")
    return FACTORIES[tag](float(width_mult))


def load(tag, ckpt_dir, manifest=None, device="cpu", strict=True):
    """Instancie `tag` et y charge les poids entraînés."""
    if manifest is None:
        with open(os.path.join(ckpt_dir, "manifest.json"), encoding="utf-8") as f:
            manifest = json.load(f)

    info = manifest["models"][tag]
    model = build(tag, info.get("width_mult")).to(device).eval()

    state = torch.load(os.path.join(ckpt_dir, info["file"]),
                       map_location=device, weights_only=True)
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=strict)

    n_p, ref = count_parameters(model), info.get("params")
    if ref is not None and n_p != ref:
        raise RuntimeError(
            f"[{tag}] {n_p:,} paramètres construits contre {ref:,} annoncés "
            f"dans le manifeste. width_mult ou configuration divergents.")
    return model


if __name__ == "__main__":
    print(f"Notebook source : {NOTEBOOK}")
    for t in TAGS:
        print(f"  • {t}")

"""
repro_models.py
=============================================================================
Bridge between the training notebook and the reviewer notebook.

The training notebook is the single source of truth for the architectures:
this module executes its definition cells and re-exposes the factories,
without duplicating a single line of model code. No training is triggered,
since main() sits behind the __name__ guard.

Usage:
    import repro_models
    model = repro_models.build("SU2Stereo", width_mult=0.7314)
    model = repro_models.load("SU2Stereo", "checkpoints/mnist_v10")

Place this file at the repository root, next to the training notebook.
"""

import glob
import json
import os

import torch

# ─────────────────────────────────────────────────────────────────────────────
#  Locating the training notebook
# ─────────────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))

NOTEBOOK = os.environ.get("REPRO_TRAINING_NOTEBOOK")
if NOTEBOOK is None:
    _cands = (glob.glob(os.path.join(_HERE, "Ablation_SU2xSim2_MNIST_V10.ipynb"))
              or glob.glob(os.path.join(_HERE, "notebooks", "Ablation_*.ipynb"))
              or glob.glob(os.path.join(_HERE, "Ablation_*.ipynb")))
    if not _cands:
        raise FileNotFoundError(
            "Training notebook not found. Place it at the repository root, or "
            "set the REPRO_TRAINING_NOTEBOOK environment variable.")
    NOTEBOOK = _cands[0]


def _exec_notebook(path):
    """Execute the notebook's code cells in an isolated namespace.

    Magic lines (!pip, %cd) are stripped, and __name__ is set so that the
    trailing  if __name__ == "__main__": main()  does not fire.
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
                f"Failed at cell {i} of {path}: {type(e).__name__}: {e}") from e
    return ns


_NS = _exec_notebook(NOTEBOOK)

# Required symbols, re-exposed for direct use
ClassicResNet          = _NS["ClassicResNet"]
Sim2OnlyResNet         = _NS["Sim2OnlyResNet"]
SU2SphericalResNetSO3  = _NS["SU2SphericalResNetSO3"]
BiLogPolarMöbiusResNet = _NS["BiLogPolarMöbiusResNet"]
make_mnist_model       = _NS["make_mnist_model"]
count_parameters       = _NS["count_parameters"]

# ─────────────────────────────────────────────────────────────────────────────
#  Factory configuration
#  Copied verbatim from main(). Any divergence would make load_state_dict fail,
#  so this block and the one in the notebook must stay in sync.
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
    """Instantiate architecture `tag` at the given width, with random weights.

    `width_mult` comes from the manifest: it freezes the 500k-parameter
    calibration performed by find_model_width at training time.
    """
    if tag not in FACTORIES:
        raise KeyError(f"Unknown model {tag!r}. Expected one of: {TAGS}")
    if width_mult is None:
        raise ValueError(
            f"[{tag}] missing width_mult. It must come from manifest.json; "
            f"recomputing it could yield dimensions that differ from those of "
            f"the checkpoint.")
    return FACTORIES[tag](float(width_mult))


def load(tag, ckpt_dir, manifest=None, device="cpu", strict=True):
    """Instantiate `tag` and load the trained weights into it."""
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
            f"[{tag}] built {n_p:,} parameters against {ref:,} declared in "
            f"the manifest. width_mult or configuration have diverged.")
    return model


if __name__ == "__main__":
    print(f"Source notebook: {NOTEBOOK}")
    for t in TAGS:
        print(f"  • {t}")

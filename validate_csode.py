"""validate_csode.py — re-validate CSODE *variant* checkpoints.

The shared [`validate.py`](validate.py:1) calls
[`models.build_model()`](models.py:728) directly and crashes with
`ValueError: Unknown model_type: csode_*` for any of the 8 Phase-2/3
variant names (csode_diag, csode_full, csode_res, csode_statedep,
csode_icnn_lyap, csode_contract, csode_hnn, csode_icnn_grad).

This wrapper monkey-patches:
    * `models.build_model`  — dispatch to `build_csode_variant{,_phase3}`
    * `models.ODE_MODELS`   — extended with the variant names

then `runpy`s `validate.py` so all metrics (T_valid, W1, sym_KL, spec_L1,
plots, *_metrics.json) get produced.

Variant constructor kwargs default to the values used in the actual
training runs (see [`experiments/run_csode_all_parallel.sh`](experiments/run_csode_all_parallel.sh:57)
and [`experiments/run_csode_phase3.sh`](experiments/run_csode_phase3.sh:67)).

Usage
-----
    python validate_csode.py --ckpt checkpoints/rossler_P2_csode_diag/rossler_csode_diag.pt
"""
from __future__ import annotations

import runpy
import sys

import models  # patch BEFORE validate.py imports it

from fsode_variants import build_csode_variant, CSODE_VARIANTS
from csode_variants_phase3 import (
    build_csode_variant_phase3,
    CSODE_VARIANTS_PHASE3,
)

ALL_VARIANTS = CSODE_VARIANTS | CSODE_VARIANTS_PHASE3

# Defaults mirroring the training scripts.
_PHASE2_DEFAULTS = {
    "csode_full":     dict(rank=3, hidden_W=16, alpha_init=-2.25),
    "csode_res":      dict(n_blocks=3, gamma_per_dim=False),
    "csode_statedep": dict(hidden_gamma=16),
    "csode_diag":     dict(),
}

_PHASE3_DEFAULTS = {
    "csode_icnn_lyap": dict(v_hidden=32, lyap_alpha=0.005),
    "csode_contract":  dict(contract_lambda_logn=1.0, contract_eps=0.02,
                            contract_samples=2),
    "csode_hnn":       dict(hnn_hidden_H=64, hnn_hidden_V=32, hnn_dissip=0.5),
    "csode_icnn_grad": dict(v_hidden=64, icnn_use_skew=True),
}


_orig_build_model = models.build_model


def _patched_build_model(model_type, dim, hidden, seq_len, aug=2, **kw):
    """Route variant names to the right factory; fall back to original."""
    if model_type in CSODE_VARIANTS:
        kwargs = dict(_PHASE2_DEFAULTS.get(model_type, {}))
        kwargs["gamma_init"] = kw.get("gamma_init", 0.1)
        return build_csode_variant(model_type, dim, hidden, aug=aug, **kwargs)
    if model_type in CSODE_VARIANTS_PHASE3:
        kwargs = dict(_PHASE3_DEFAULTS.get(model_type, {}))
        kwargs["gamma_init"] = kw.get("gamma_init", 0.1)
        return build_csode_variant_phase3(model_type, dim, hidden, aug=aug,
                                          **kwargs)
    return _orig_build_model(model_type, dim, hidden, seq_len, aug=aug, **kw)


models.build_model = _patched_build_model
models.ODE_MODELS = set(models.ODE_MODELS) | ALL_VARIANTS
models.REGULARIZED_MODELS = set(models.REGULARIZED_MODELS) | ALL_VARIANTS

print(f"# [validate_csode] patched build_model with variants: "
      f"{sorted(ALL_VARIANTS)}")

if __name__ == "__main__":
    runpy.run_path("validate.py", run_name="__main__")

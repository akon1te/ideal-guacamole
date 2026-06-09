from __future__ import annotations

import argparse
import sys
import runpy

import models

from fsode_variants import build_csode_variant, CSODE_VARIANTS


_VARIANT_ARGS = {
    "--csode_full_rank":        ("csode_full_rank",        int,   None),
    "--csode_full_hidden_W":    ("csode_full_hidden_W",    int,   16),
    "--csode_full_alpha_init":  ("csode_full_alpha_init",  float, -2.25),
    "--csode_res_blocks":       ("csode_res_blocks",       int,   2),
    "--csode_res_per_dim_gamma":("csode_res_per_dim_gamma", "flag", False),
    "--csode_statedep_hidden_gamma": ("csode_statedep_hidden_gamma", int, 16),
}


def _extract_variant_args(argv):

    kwargs = {}

    for _argname, (key, _type, default) in _VARIANT_ARGS.items():
        kwargs[key] = default

    rest = []
    i = 0
    while i < len(argv):
        tok = argv[i]
        matched = False
        for argname, (key, type_, default) in _VARIANT_ARGS.items():
            # also accept --flag=value form
            if tok == argname:
                if type_ == "flag":
                    kwargs[key] = True
                    i += 1
                else:
                    if i + 1 >= len(argv):
                        raise SystemExit(f"{argname} requires a value")
                    val = argv[i + 1]
                    kwargs[key] = type_(val)
                    i += 2
                matched = True
                break
            if tok.startswith(argname + "="):
                val = tok.split("=", 1)[1]
                if type_ == "flag":
                    kwargs[key] = val.lower() in ("1", "true", "yes")
                else:
                    kwargs[key] = type_(val)
                i += 1
                matched = True
                break
        if not matched:
            rest.append(tok)
            i += 1
    return kwargs, rest


_variant_kwargs, _rest_argv = _extract_variant_args(sys.argv[1:])
sys.argv = [sys.argv[0]] + _rest_argv

_model_idx = None
for i, tok in enumerate(_rest_argv):
    if tok == "--model" and i + 1 < len(_rest_argv):
        _model_idx = i + 1
        break
    if tok.startswith("--model="):
        _model_idx = i
        break

_requested_model = None
if _model_idx is not None:
    tok = _rest_argv[_model_idx]
    _requested_model = tok.split("=", 1)[1] if tok.startswith("--model=") else tok

if _requested_model in CSODE_VARIANTS:
    print(f"# [train_csode] CSODE variant requested: {_requested_model}")
    print(f"# [train_csode] variant kwargs: {_variant_kwargs}")
else:
    print(f"# [train_csode] forwarding to train.py with model={_requested_model!r} "
          f"(no CSODE variant matched, will use stock build_model)")


_orig_build_model = models.build_model


def _patched_build_model(model_type, dim, hidden, seq_len, aug=2, **kw):

    if model_type in CSODE_VARIANTS:
        gamma_init = kw.get("gamma_init", 0.1)
        eff_aug = aug if aug > 0 else 4
        return build_csode_variant(
            model_type, dim, hidden, aug=eff_aug,
            hidden_W=_variant_kwargs["csode_full_hidden_W"],
            rank=_variant_kwargs["csode_full_rank"],
            alpha_init=_variant_kwargs["csode_full_alpha_init"],
            n_blocks=_variant_kwargs["csode_res_blocks"],
            gamma_per_dim=_variant_kwargs["csode_res_per_dim_gamma"],
            hidden_gamma=_variant_kwargs["csode_statedep_hidden_gamma"],
            gamma_init=gamma_init,
        )
    return _orig_build_model(model_type, dim, hidden, seq_len, aug=aug, **kw)


models.build_model = _patched_build_model

models.ODE_MODELS = set(models.ODE_MODELS) | CSODE_VARIANTS
models.REGULARIZED_MODELS = set(models.REGULARIZED_MODELS) | CSODE_VARIANTS

_orig_add_argument = argparse.ArgumentParser.add_argument


def _patched_add_argument(self, *args, **kwargs):
    if args and args[0] == "--model" and "choices" in kwargs:
        kwargs["choices"] = list(kwargs["choices"]) + sorted(CSODE_VARIANTS)
    return _orig_add_argument(self, *args, **kwargs)


argparse.ArgumentParser.add_argument = _patched_add_argument

if __name__ == "__main__":
    runpy.run_path("train.py", run_name="__main__")

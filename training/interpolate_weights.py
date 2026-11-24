"""Weight-space interpolation between two HawkEye checkpoints (WiSE-FT style).

    model = (1 - alpha) * base + alpha * finetuned

Fine-tuning on new data (v3: postures, INFERNO thermal, hard negatives) can cost some accuracy on
what the base model (v2) already did well. Interpolating the weights typically keeps most of the
fine-tune's gains while recovering the base model's strengths, without any retraining.

    python interpolate_weights.py --base hawkeye_v2.pt --finetuned hawkeye_v3.pt --alpha 0.5 --out hawkeye_v3_wise50.pt
"""

import argparse
from copy import deepcopy

try:
    import albumentations  # noqa: F401  (Windows: load before torch/CUDA to avoid a DLL deadlock)
except ImportError:
    pass

import torch


def interpolate(base_path: str, finetuned_path: str, alpha: float, out_path: str) -> None:
    base = torch.load(base_path, map_location="cpu", weights_only=False)
    tuned = torch.load(finetuned_path, map_location="cpu", weights_only=False)
    base_sd = base["model"].float().state_dict()
    tuned_model = deepcopy(tuned["model"]).float()
    tuned_sd = tuned_model.state_dict()
    if base_sd.keys() != tuned_sd.keys():
        raise SystemExit("Checkpoints have different architectures; cannot interpolate.")

    mixed = {}
    for key, value in tuned_sd.items():
        if value.dtype.is_floating_point:
            mixed[key] = (1 - alpha) * base_sd[key] + alpha * value
        else:  # e.g. BatchNorm num_batches_tracked
            mixed[key] = value
    tuned_model.load_state_dict(mixed)

    ckpt = {k: v for k, v in tuned.items() if k not in ("model", "ema", "optimizer")}
    ckpt["model"] = tuned_model.half()
    ckpt["ema"] = None
    ckpt["optimizer"] = None
    ckpt["interpolation"] = {"base": base_path, "finetuned": finetuned_path, "alpha": alpha}
    torch.save(ckpt, out_path)
    print(f"saved {out_path} (alpha={alpha})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--finetuned", required=True)
    parser.add_argument("--alpha", type=float, required=True, help="0 = base model, 1 = fine-tuned model")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    interpolate(args.base, args.finetuned, args.alpha, args.out)

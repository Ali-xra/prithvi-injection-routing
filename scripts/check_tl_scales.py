"""
Read the two metadata scalars straight out of IBM's released Prithvi-EO-2.0 weights.

Both are initialised at 0.1 and learned during pretraining, so their final values say
what pretraining decided location and time were worth.

    python scripts/check_tl_scales.py

Requires: torch, huggingface_hub. Downloads ~2.4 GB the first time, cached afterwards.

Expected output (measured 2026-08-01):

    Prithvi-EO-2.0-300M      398 keys   no coords keys
    Prithvi-EO-2.0-300M-TL   402 keys
        encoder.location_embed_enc.scale = 0.05815186
        encoder.temporal_embed_enc.scale = 0.00000128

The temporal scale is five orders of magnitude below its initialisation.
"""
import torch
from huggingface_hub import hf_hub_download

CHECKPOINTS = [
    ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M", "Prithvi_EO_V2_300M.pt"),
    ("ibm-nasa-geospatial/Prithvi-EO-2.0-300M-TL", "Prithvi_EO_V2_300M_TL.pt"),
]
INIT_VALUE = 0.1  # terratorch/models/backbones/prithvi_mae.py

# A tensor both checkpoints have, with different values - use it to tell them apart.
FINGERPRINT = "blocks.0.attn.qkv.weight"


def main():
    for repo, filename in CHECKPOINTS:
        print(f"\n=== {repo} ===", flush=True)
        state = torch.load(
            hf_hub_download(repo_id=repo, filename=filename),
            map_location="cpu", weights_only=True)
        state = state.get("model", state)

        coords = sorted(k for k in state if "embed_enc" in k)
        print(f"  {len(state)} keys")
        if not coords:
            print("  no coords keys - this checkpoint has no metadata pathway")
        for key in coords:
            value = state[key].item()
            print(f"  {key} = {value:.8f}   ({value / INIT_VALUE:.6f} x init)")

        fp = next((v for k, v in state.items() if k.endswith(FINGERPRINT)), None)
        if fp is not None:
            print(f"  fingerprint sum({FINGERPRINT}) = {fp.sum().item():+.6f}")


if __name__ == "__main__":
    main()

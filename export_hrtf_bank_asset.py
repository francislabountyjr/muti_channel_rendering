import argparse
import json
from pathlib import Path

import numpy as np
from safetensors.torch import save_file
import torch

from binaural_channel import EXPECTED_714_MASK, EXPECTED_714_ORDER, build_hrtf_filters


METADATA_FILENAME = "metadata.json"
TENSOR_FILENAME = "hrtf_bank.safetensors"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a precomputed native HRTF bank asset derived from HRIR_L2702.sofa."
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where the HRTF asset will be written.",
    )
    parser.add_argument(
        "--sofa-path",
        default=str(Path(__file__).resolve().parent / "HRIR_L2702.sofa"),
        help="Path to the SOFA HRTF file.",
    )
    parser.add_argument(
        "--asset-id",
        default="hrir_l2702_714",
        help="Stable asset identifier written into metadata.",
    )
    parser.add_argument(
        "--asset-version",
        default="1",
        help="Asset version string written into metadata.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    hrtf_left_list, hrtf_right_list = build_hrtf_filters(args.sofa_path, verbose=False)
    hrtf_left = torch.from_numpy(np.stack(hrtf_left_list)).to(torch.float32).contiguous()
    hrtf_right = torch.from_numpy(np.stack(hrtf_right_list)).to(torch.float32).contiguous()

    save_file(
        {
            "hrtf_left": hrtf_left,
            "hrtf_right": hrtf_right,
        },
        str(output_dir / TENSOR_FILENAME),
    )

    metadata = {
        "asset_kind": "stereo2spatial_hrtf_bank",
        "asset_id": args.asset_id,
        "asset_version": args.asset_version,
        "source_sofa_path": str(Path(args.sofa_path).resolve()),
        "sample_rate": 48000,
        "channel_layout": "7.1.4",
        "channel_order": list(EXPECTED_714_ORDER),
        "wave_extensible_mask": int(EXPECTED_714_MASK),
        "filter_length": int(hrtf_left.size(1)),
        "normalize_peak_default": True,
        "tensor_filename": TENSOR_FILENAME,
    }
    (output_dir / METADATA_FILENAME).write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        json.dumps(
            {
                "metadata_path": str((output_dir / METADATA_FILENAME).resolve()),
                "tensor_path": str((output_dir / TENSOR_FILENAME).resolve()),
                "hrtf_shape": list(hrtf_left.shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

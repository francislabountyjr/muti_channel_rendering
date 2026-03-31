import argparse
import struct
from pathlib import Path
import numpy as np
import soundfile as sf
import pysofaconventions
from scipy.signal import fftconvolve
import pyloudnorm as pyln
from spaudiopy import decoder

EXPECTED_714_ORDER = (
    "FL",
    "FR",
    "FC",
    "LFE",
    "BL",
    "BR",
    "SL",
    "SR",
    "TFL",
    "TFR",
    "TBL",
    "TBR",
)
EXPECTED_714_MASK = 0x2D63F
SPEAKER_ANGLES_DEG = {
    "FL": (-30, 0),
    "FR": (30, 0),
    "FC": (0, 0),
    "LFE": (0, 0),
    "BL": (-135, 0),
    "BR": (135, 0),
    "SL": (-110, 0),
    "SR": (110, 0),
    "TFL": (-45, 45),
    "TFR": (45, 45),
    "TBL": (-135, 45),
    "TBR": (135, 45),
}
_SPEAKER_BITS = (
    (0x00001, "FL"),
    (0x00002, "FR"),
    (0x00004, "FC"),
    (0x00008, "LFE"),
    (0x00010, "BL"),
    (0x00020, "BR"),
    (0x00040, "FLC"),
    (0x00080, "FRC"),
    (0x00100, "BC"),
    (0x00200, "SL"),
    (0x00400, "SR"),
    (0x00800, "TC"),
    (0x01000, "TFL"),
    (0x02000, "TFC"),
    (0x04000, "TFR"),
    (0x08000, "TBL"),
    (0x10000, "TBC"),
    (0x20000, "TBR"),
)
_KNOWN_SPEAKER_MASK_BITS = sum(bit for bit, _ in _SPEAKER_BITS)


def load_hrtf(sofa_path):
    # Load HRTF data from SOFA file
    sofa = pysofaconventions.SOFAFile(sofa_path, 'r')
    az = sofa.getVariableValue('SourcePosition')[:, 0]
    el = sofa.getVariableValue('SourcePosition')[:, 1]
    hrir = sofa.getDataIR()  # (n_directions, 2, n_samples)
    return az, el, hrir


def spherical_to_cartesian(az_deg, el_deg, r=1.0):
 
    az_rad = np.deg2rad(az_deg)
    zen_rad = np.deg2rad(90 - el_deg)
    x = r * np.cos(az_rad) * np.sin(zen_rad)
    y = r * np.sin(az_rad) * np.sin(zen_rad)
    z = r * np.cos(zen_rad)
    return x, y, z


def read_wav_channel_mask(input_path):
    path = Path(input_path)
    if path.suffix.lower() != ".wav":
        return None

    with path.open("rb") as handle:
        if handle.read(4) != b"RIFF":
            return None
        handle.seek(8)
        if handle.read(4) != b"WAVE":
            return None
        handle.seek(12)

        while True:
            chunk_header = handle.read(8)
            if len(chunk_header) < 8:
                return None

            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                fmt_chunk = handle.read(chunk_size)
                if len(fmt_chunk) < 40:
                    return None

                format_tag = struct.unpack_from("<H", fmt_chunk, 0)[0]
                if format_tag != 0xFFFE:
                    return None
                return struct.unpack_from("<I", fmt_chunk, 20)[0]

            handle.seek(chunk_size + (chunk_size % 2), 1)


def speakers_from_channel_mask(channel_mask):
    unknown_bits = channel_mask & ~_KNOWN_SPEAKER_MASK_BITS
    if unknown_bits:
        raise ValueError(f"Unsupported channel mask bits: 0x{unknown_bits:X}")
    return [speaker for bit, speaker in _SPEAKER_BITS if channel_mask & bit]


def reorder_to_expected_714(audio_714, input_path):
    if audio_714.ndim != 2:
        raise ValueError(f"Expected 2D audio array, got shape={audio_714.shape}")
    if audio_714.shape[1] != len(EXPECTED_714_ORDER):
        raise ValueError(
            f"Expected 12 channels for 7.1.4 input, got {audio_714.shape[1]} in {input_path}"
        )

    channel_mask = read_wav_channel_mask(input_path)
    if not channel_mask:
        print(
            f"No WAVEX channel mask found for {input_path}; "
            "assuming Cavernize/WAVEX 7.1.4 order."
        )
        return audio_714
    if channel_mask == EXPECTED_714_MASK:
        return audio_714

    speaker_order = speakers_from_channel_mask(channel_mask)
    if set(speaker_order) != set(EXPECTED_714_ORDER):
        raise ValueError(
            f"Unsupported 7.1.4 speaker set in {input_path}: "
            f"mask=0x{channel_mask:X} speakers={speaker_order}"
        )

    if tuple(speaker_order) == EXPECTED_714_ORDER:
        return audio_714

    reorder_idx = [speaker_order.index(name) for name in EXPECTED_714_ORDER]
    print(
        f"Reordering channels for {input_path} from {speaker_order} "
        f"to {list(EXPECTED_714_ORDER)}"
    )
    return audio_714[:, reorder_idx]



def binaural_convolve(audio_714, hrtf_l_list, hrtf_r_list):
    if audio_714.ndim != 2 or audio_714.shape[1] != len(EXPECTED_714_ORDER):
        raise ValueError(
            f"Expected audio shaped [samples, 12], got {audio_714.shape}"
        )
    max_hrtf_len = max(len(h) for h in hrtf_l_list)
    n_samples = audio_714.shape[0] + max_hrtf_len - 1
    out = np.zeros((n_samples, 2))  # stereo

    for ch in range(audio_714.shape[1]):
        ch_audio = audio_714[:, ch]
        if ch == 3:  # LFE channel
            lfe = ch_audio * 0.5  # down-mix by 6 dB
            out[:len(lfe), 0] += lfe
            out[:len(lfe), 1] += lfe
        else:
            out[:, 0] += fftconvolve(ch_audio, hrtf_l_list[ch])
            out[:, 1] += fftconvolve(ch_audio, hrtf_r_list[ch])

    out /= np.max(np.abs(out) + 1e-6)
    return out


def build_hrtf_filters(sofa_path):
    # 1. Define 7.1.4 loudspeaker angles
    ls_deg = [SPEAKER_ANGLES_DEG[speaker] for speaker in EXPECTED_714_ORDER]
    # Convert negative azimuth angles to 0-360 degrees
    ls_deg = [(az+360 if az<0 else az, el) for az, el in ls_deg]

    # 2. sph2cart
    ls_cart_coords = np.array([spherical_to_cartesian(az, el, 1.0) for az, el in ls_deg])

    # 3. Load HRTF data and convert to cartesian coordinates
    hrtf_az, hrtf_el, hrtf = load_hrtf(sofa_path)
    hrtf_cart_coords = np.array([spherical_to_cartesian(az, el, 1.0) for az, el in zip(hrtf_az, hrtf_el)])

    # 4. Creates a 'hull' object
    hrtf_setup = decoder.LoudspeakerSetup(hrtf_cart_coords[:,0], hrtf_cart_coords[:,1], hrtf_cart_coords[:,2])

    # 5. VBAP 
    vbap_gains = decoder.vbap(ls_cart_coords, hrtf_setup)  # (12, N_hrtf)

    # 6. Interpolate HRTF using weights
    hrtf_l_list = []
    hrtf_r_list = []
    for ch, speaker in enumerate(EXPECTED_714_ORDER):
        weights = vbap_gains[ch]
        nonzero_idx = np.where(weights != 0)[0]
        az, el = ls_deg[ch]
        print(f"Channel {ch} ({speaker}): angle (azimuth={az}, elevation={el})")
        print(f" 3 HRTF indices: {nonzero_idx}")
        print("  Corresponding angles and weights:")
        for idx in nonzero_idx:
            cur_hrtf_az = hrtf_az[idx]
            cur_hrtf_el = hrtf_el[idx]
            w = weights[idx]
            print(f"    HRTF idx {idx}: azimuth={cur_hrtf_az:.3f}, elevation={cur_hrtf_el:.3f}, weight={w:.3f}")
        # Weighted sum for left and right ear
        h_l = np.sum(hrtf[:,0,:] * weights[:,None], axis=0)
        h_r = np.sum(hrtf[:,1,:] * weights[:,None], axis=0)
        hrtf_l_list.append(h_l)
        hrtf_r_list.append(h_r)

    return hrtf_l_list, hrtf_r_list


def process_file(input_path, output_path, hrtf_l_list, hrtf_r_list, target_lufs):
    # 7. Convolution
    audio_714, sr = sf.read(input_path, always_2d=True)
    audio_714 = reorder_to_expected_714(audio_714, input_path)
    binaural_audio = binaural_convolve(audio_714, hrtf_l_list, hrtf_r_list)

    # 8. LUFS normalization
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(binaural_audio)
    binaural_audio = pyln.normalize.loudness(binaural_audio, loudness, target_lufs)

    sf.write(output_path, binaural_audio, sr)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert 7.1.4 multichannel audio to binaural output."
    )
    parser.add_argument(
        "--input-path",
        default="./test.wav",
        help="Path to a single multichannel input file (used when --input-dir is not set).",
    )
    parser.add_argument(
        "--input-dir",
        help="Directory of multichannel input files to process in batch mode.",
    )
    parser.add_argument(
        "--output-path",
        default="./output_binaural.wav",
        help="Path for single-file output (used when --input-dir is not set).",
    )
    parser.add_argument(
        "--output-dir",
        default="./binaural_outputs",
        help="Output directory for batch mode.",
    )
    parser.add_argument(
        "--output-suffix",
        default="_binaural",
        help="Suffix appended to each batch output filename.",
    )
    parser.add_argument(
        "--sofa-path",
        default="./HRIR_L2702.sofa",
        help="Path to SOFA HRTF file.",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-24.0,
        help="Target LUFS for loudness normalization.",
    )
    return parser.parse_args()


def collect_audio_files(input_dir):
    audio_exts = {".wav", ".flac", ".aif", ".aiff", ".ogg"}
    return sorted(
        file_path for file_path in input_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() in audio_exts
    )


def main():
    args = parse_args()
    hrtf_l_list, hrtf_r_list = build_hrtf_filters(args.sofa_path)

    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        input_files = collect_audio_files(input_dir)
        if not input_files:
            raise FileNotFoundError(f"No supported audio files found in: {input_dir}")

        for input_file in input_files:
            output_file = output_dir / f"{input_file.stem}{args.output_suffix}.wav"
            print(f"Processing: {input_file} -> {output_file}")
            process_file(
                str(input_file),
                str(output_file),
                hrtf_l_list,
                hrtf_r_list,
                args.target_lufs,
            )
    else:
        process_file(
            args.input_path,
            args.output_path,
            hrtf_l_list,
            hrtf_r_list,
            args.target_lufs,
        )


if __name__ == "__main__":
    main()

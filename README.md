# Multichannel (7.1.4) to Binaural Audio Converter

This script converts 7.1.4 surround audio into binaural stereo with VBAP-based HRTF interpolation.  
It supports:

- single-file processing
- batch folder processing
- command-line argument parsing for automation/scripting

## Usage

### Single file (default mode)

```bash
python binaural_channel.py \
  --input-path ./test.wav \
  --output-path ./output_binaural.wav \
  --sofa-path ./HRIR_L2702.sofa \
  --target-lufs -24
```

If no arguments are passed, defaults are used:

- `--input-path ./test.wav`
- `--output-path ./output_binaural.wav`
- `--sofa-path ./HRIR_L2702.sofa`
- `--target-lufs -24.0`

### Batch folder mode

```bash
python binaural_channel.py \
  --input-dir ./input_audio \
  --output-dir ./binaural_outputs \
  --output-suffix _binaural \
  --sofa-path ./HRIR_L2702.sofa \
  --target-lufs -24
```

In batch mode, each supported audio file in `--input-dir` is rendered to:

- `<output-dir>/<original_name><output-suffix>.wav`

Supported input extensions:

- `.wav`, `.flac`, `.aif`, `.aiff`, `.ogg`

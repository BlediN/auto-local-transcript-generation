# Audio to Test

Local transcription to TXT, VTT, and SRT using faster-whisper.

## Requirements

- Python 3.12.9
- Packages: `faster-whisper`, `srt`

Install:

```bash
pip install faster-whisper srt
```

## Usage

Run with a media file path:

```bash
python local_transcribe.py "C:\path\to\your\video" --model small --lang en --outdir output
```

Or set a default path in the script and run without an argument:

```bash
python local_transcribe.py --model small --lang en --outdir output
```

## Configuration

In `local_transcribe.py`, set the default media path:

```python
default_media = r"C:\Users\Bndoni\Videos\sample-video.mp4"
```

## Outputs

By default, outputs are written to `output/` with the same base name as the media file:

- `.txt`
- `.vtt`
- `.srt`

Use `--no_srt` to skip SRT generation.

## Notes

- `--device` can be `cpu` or `cuda`.
- `--compute_type` defaults to `int8` for CPU.
- Language can be auto-detected if `--lang` is omitted.

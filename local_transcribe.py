import argparse
import os
import re
from datetime import timedelta

from faster_whisper import WhisperModel
import srt


default_media = r"C:\path\to\video"

def format_vtt_timestamp(seconds: float) -> str:
    # WebVTT timestamp: HH:MM:SS.mmm
    td = timedelta(seconds=float(seconds))
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((float(seconds) - int(seconds)) * 1000)
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"

def clean_text(text: str) -> str:
    # Basic cleanup: collapse spaces
    return re.sub(r"\s+", " ", text).strip()

def write_txt(out_base: str, segments):
    txt_path = out_base + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(clean_text(seg.text) + "\n")
    return txt_path

def write_vtt(out_base: str, segments):
    vtt_path = out_base + ".vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for seg in segments:
            start = format_vtt_timestamp(seg.start)
            end = format_vtt_timestamp(seg.end)
            text = clean_text(seg.text)
            if text:
                f.write(f"{start} --> {end}\n{text}\n\n")
    return vtt_path

def write_srt(out_base: str, segments):
    srt_path = out_base + ".srt"
    subs = []
    for i, seg in enumerate(segments, start=1):
        start = timedelta(seconds=float(seg.start))
        end = timedelta(seconds=float(seg.end))
        content = clean_text(seg.text)
        if not content:
            continue
        subs.append(
            srt.Subtitle(index=i, start=start, end=end, content=content)
        )
    with open(srt_path, "w", encoding="utf-8") as f:
        f.write(srt.compose(subs))
    return srt_path

def main():
    parser = argparse.ArgumentParser(description="Local transcription to TXT + VTT + SRT using faster-whisper.")
    parser.add_argument(
        "media",
        nargs="?",
        default=default_media,
        help="Path to video/audio file (mp4, mov, mp3, wav, etc.)",
    )
    parser.add_argument("--model", default="small", help="Model size: tiny, base, small, medium, large-v3")
    parser.add_argument("--lang", default=None, help="Language code (e.g., en). If omitted, auto-detect.")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu or cuda (NVIDIA)")
    parser.add_argument("--compute_type", default="int8", help="int8 (fast CPU), float16 (GPU), float32")
    parser.add_argument("--outdir", default="output", help="Output directory")
    parser.add_argument("--basename", default=None, help="Base filename (without extension)")
    parser.add_argument("--no_srt", action="store_true", help="Do not produce SRT")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    base = args.basename
    if not base:
        stem = os.path.splitext(os.path.basename(args.media))[0]
        base = stem

    out_base = os.path.join(args.outdir, base)

    print(f"[INFO] Loading model: {args.model} (device={args.device}, compute_type={args.compute_type})")
    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)

    print(f"[INFO] Transcribing: {args.media}")
    segments_iter, info = model.transcribe(
        args.media,
        language=args.lang,
        vad_filter=True,           # improves segmentation by removing silence
        vad_parameters=dict(min_silence_duration_ms=500),
    )

    segments = list(segments_iter)

    detected = info.language if info and hasattr(info, "language") else None
    print(f"[INFO] Detected language: {detected}")
    print(f"[INFO] Segments: {len(segments)}")

    txt_path = write_txt(out_base, segments)
    vtt_path = write_vtt(out_base, segments)
    srt_path = None if args.no_srt else write_srt(out_base, segments)

    print("[DONE] Outputs:")
    print(" -", txt_path)
    print(" -", vtt_path)
    if srt_path:
        print(" -", srt_path)

if __name__ == "__main__":
    main()
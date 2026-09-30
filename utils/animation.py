"""Combine training progress PNGs into a looping GIF ordered by step."""

import argparse
import re
from pathlib import Path

from PIL import Image, ImageDraw


def build_progress_gif(paths: list[Path], output_path: Path, duration_ms: int = 500) -> None:
    """Build a GIF from explicit frames, without mixing in stale run images.

    Args:
        paths: Progress PNG paths named generated_step_<number>.png.
        output_path: Destination GIF file.
        duration_ms: Display time per frame, in milliseconds (at least 10).
    """
    if not paths:
        raise ValueError("No progress images found; enable --save-progress and run at least --save-every steps")
    if duration_ms < 10:
        raise ValueError("GIF duration must be at least 10 milliseconds")
    numbered_paths = []
    for path in paths:
        match = re.fullmatch(r"generated_step_(\d+)\.png", path.name)
        if match is None:
            raise ValueError(f"Invalid progress image name: {path.name}")
        numbered_paths.append((int(match.group(1)), path))
    frames = []
    try:
        for step, path in sorted(numbered_paths):
            with Image.open(path) as source:
                frame = source.convert("RGB")
            # Keep animation size practical; original PNGs remain full resolution.
            frame.thumbnail((1280, 1280), Image.Resampling.LANCZOS)
            canvas = Image.new("RGB", (frame.width, frame.height + 24), "white")
            canvas.paste(frame, (0, 24))
            ImageDraw.Draw(canvas).text((12, 5), f"Step {step}", fill="black")
            frame.close()
            frames.append(canvas.convert("P", palette=Image.Palette.ADAPTIVE))
            canvas.close()
        if any(frame.size != frames[0].size for frame in frames):
            raise ValueError("Progress images must have matching dimensions")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames[0].save(output_path, save_all=True, append_images=frames[1:],
                       duration=duration_ms, loop=0, disposal=2, optimize=False)
    finally:
        for frame in frames:
            frame.close()


def main() -> None:
    """Create an animation from an existing progress directory without training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--duration-ms", type=int, default=500)
    args = parser.parse_args()
    output_path = args.output or args.progress_dir.parent / "progress.gif"
    build_progress_gif(list(args.progress_dir.glob("generated_step_*.png")), output_path, args.duration_ms)
    print(f"Saved animation: {output_path.resolve()}")


if __name__ == "__main__":
    main()

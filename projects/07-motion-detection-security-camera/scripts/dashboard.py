#!/usr/bin/env python3
"""Build a local dashboard of recorded motion events, and optionally serve it.

Reads `results/events.json` and the clips in `recordings/`, and writes a single
self-contained HTML page listing every event with its timestamp, duration,
thumbnail, and a player for the clip.

The page is plain static HTML served from the local filesystem: there is no
upload, no account, and nothing leaves the machine. A browser will not play a
video from a `file://` page in every configuration, so `--serve` starts a
throwaway HTTP server bound to localhost, which is the reliable way to view it.
"""

from __future__ import annotations

import argparse
import html
import http.server
import socketserver
import sys
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "python"))

from src.errors import ConfigurationError  # noqa: E402
from src.events import load_json  # noqa: E402

DEFAULT_EVENTS = PROJECT_ROOT / "results" / "events.json"
DEFAULT_RECORDINGS = PROJECT_ROOT / "recordings"
DEFAULT_OUTPUT = PROJECT_ROOT / "recordings" / "index.html"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Motion Events</title>
<style>
  :root {{
    color-scheme: light dark;
    --bg: #f6f7f9; --card: #ffffff; --ink: #16181d; --muted: #5c6370;
    --line: #e2e5ea; --accent: #c4342b;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --bg: #14161a; --card: #1c1f25; --ink: #e8eaed;
             --muted: #9aa1ab; --line: #2b2f37; --accent: #ff6b5e; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--bg); color: var(--ink);
    font: 15px/1.5 system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }}
  .wrap {{ max-width: 1100px; margin: 0 auto; padding: 32px 16px 64px; }}
  h1 {{ font-size: 1.6rem; margin: 0 0 4px; }}
  .sub {{ color: var(--muted); margin: 0 0 28px; }}
  .stats {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 28px; }}
  .stat {{ background: var(--card); border: 1px solid var(--line);
    border-radius: 10px; padding: 12px 18px; }}
  .stat b {{ display: block; font-size: 1.5rem; }}
  .stat span {{ color: var(--muted); font-size: .85rem; }}
  .grid {{ display: grid; gap: 18px;
    grid-template-columns: repeat(auto-fill, minmax(320px, 1fr)); }}
  .card {{ background: var(--card); border: 1px solid var(--line);
    border-radius: 12px; overflow: hidden; }}
  .card video, .card img {{ width: 100%; display: block; background: #000;
    aspect-ratio: 4 / 3; object-fit: contain; }}
  .meta {{ padding: 12px 14px; }}
  .meta h2 {{ font-size: 1rem; margin: 0 0 6px;
    display: flex; align-items: center; gap: 8px; }}
  .dot {{ width: 9px; height: 9px; border-radius: 50%;
    background: var(--accent); flex: none; }}
  .meta dl {{ margin: 0; display: grid; grid-template-columns: auto 1fr;
    gap: 2px 12px; font-size: .88rem; }}
  .meta dt {{ color: var(--muted); }}
  .meta dd {{ margin: 0; }}
  .empty {{ background: var(--card); border: 1px dashed var(--line);
    border-radius: 12px; padding: 48px 24px; text-align: center; color: var(--muted); }}
  code {{ background: rgba(128,128,128,.14); padding: 1px 6px; border-radius: 5px; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>Motion Events</h1>
  <p class="sub">Generated {generated} &middot; {source}</p>
  <div class="stats">
    <div class="stat"><b>{count}</b><span>events</span></div>
    <div class="stat"><b>{total_duration}</b><span>recorded</span></div>
    <div class="stat"><b>{longest}</b><span>longest event</span></div>
    <div class="stat"><b>{first}</b><span>first event</span></div>
    <div class="stat"><b>{last}</b><span>latest event</span></div>
  </div>
  {body}
</div>
</body>
</html>
"""

CARD = """  <article class="card">
    {media}
    <div class="meta">
      <h2><span class="dot"></span>Event {index}</h2>
      <dl>
        <dt>Time</dt><dd>{started}</dd>
        <dt>Duration</dt><dd>{duration}</dd>
        <dt>Frames</dt><dd>{frames}</dd>
        <dt>Regions</dt><dd>{regions} (peak)</dd>
        <dt>File</dt><dd><code>{video}</code></dd>
      </dl>
    </div>
  </article>
"""

EMPTY = """  <div class="empty">
    <p>No motion events recorded yet.</p>
    <p>Run <code>python python/security_camera.py</code> and move in front of the camera.</p>
  </div>
"""


def _format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, remainder = divmod(int(round(seconds)), 60)
    return f"{minutes}m {remainder:02d}s"


def _media_markup(event: dict, recordings: Path, output_dir: Path) -> str:
    """A video player when the clip exists, else the snapshot, else nothing."""
    video = str(event.get("video") or "")
    snapshot = str(event.get("snapshot") or "")
    if video and (recordings / video).is_file():
        try:
            source = (recordings / video).relative_to(output_dir).as_posix()
        except ValueError:
            source = (recordings / video).as_uri()
        poster = ""
        if snapshot and (recordings / snapshot).is_file():
            try:
                poster = f' poster="{html.escape((recordings / snapshot).relative_to(output_dir).as_posix())}"'
            except ValueError:
                poster = ""
        return f'<video controls preload="none"{poster} src="{html.escape(source)}"></video>'
    if snapshot and (recordings / snapshot).is_file():
        try:
            source = (recordings / snapshot).relative_to(output_dir).as_posix()
        except ValueError:
            source = (recordings / snapshot).as_uri()
        return f'<img alt="Event snapshot" src="{html.escape(source)}">'
    return ""


def build_page(events: Sequence[dict], recordings: Path, output: Path, source_label: str) -> str:
    output_dir = output.parent
    # Newest first: the thing you want after an alert is the latest clip.
    ordered = sorted(events, key=lambda item: str(item.get("started_at", "")), reverse=True)
    durations = [float(item.get("duration_seconds") or 0.0) for item in ordered]
    cards = "".join(
        CARD.format(
            media=_media_markup(event, recordings, output_dir),
            index=html.escape(str(event.get("index", "?"))),
            started=html.escape(str(event.get("started_at", "unknown"))),
            duration=_format_duration(float(event.get("duration_seconds") or 0.0)),
            frames=html.escape(str(event.get("frames", 0))),
            regions=html.escape(str(event.get("peak_regions", 0))),
            video=html.escape(str(event.get("video") or "not written")),
        )
        for event in ordered
    )
    stamps = [str(item.get("started_at", "")) for item in ordered if item.get("started_at")]
    return PAGE.format(
        generated=html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        source=html.escape(source_label),
        count=len(ordered),
        total_duration=_format_duration(sum(durations)) if durations else "0s",
        longest=_format_duration(max(durations)) if durations else "0s",
        first=html.escape(stamps[-1][11:19]) if stamps else "--",
        last=html.escape(stamps[0][11:19]) if stamps else "--",
        body=f'<div class="grid">\n{cards}</div>' if ordered else EMPTY,
    )


def serve(directory: Path, port: int, *, open_browser: bool = True) -> None:
    """Serve `directory` on localhost until interrupted."""

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, fmt: str, *args) -> None:
            return None

    # Bind to loopback only. These clips are of someone's home or office and
    # have no business being reachable from the network.
    with socketserver.TCPServer(("127.0.0.1", port), Handler) as server:
        url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
        print(f"Serving {directory} at {url}")
        print("Press Ctrl+C to stop.")
        if open_browser:
            try:
                webbrowser.open(url)
            except Exception:
                pass
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("\nStopped.")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--events", type=Path, default=DEFAULT_EVENTS)
    parser.add_argument("--recordings", type=Path, default=DEFAULT_RECORDINGS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--serve", action="store_true", help="serve the page on localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-browser", action="store_true", help="do not open a browser")
    args = parser.parse_args(argv)

    try:
        events = load_json(args.events)
    except ConfigurationError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    recordings = args.recordings.expanduser().resolve()
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    label = f"{len(events)} event(s) from {args.events.name}"
    try:
        output.write_text(
            build_page(events, recordings, output, label), encoding="utf-8"
        )
    except OSError as exc:
        print(f"error: could not write {output}: {exc}", file=sys.stderr)
        return 2

    print(f"Wrote {output}")
    if not events:
        print("No events yet; the page explains how to record some.")
    if args.serve:
        if args.port < 0 or args.port > 65535:
            print("error: --port must be between 0 and 65535", file=sys.stderr)
            return 2
        serve(output.parent, args.port, open_browser=not args.no_browser)
    else:
        print("Pass --serve to view it in a browser (video playback needs HTTP).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

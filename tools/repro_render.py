"""Reproduce `/api/render` outside the server, so the exception is visible in full.

The server's render runs in a thread and stores `traceback.format_exc()[-2000:]` on the job; polling that over HTTP
works but truncates and races.  Doing the same calls directly gives the whole trace and lets the inputs be varied.
"""
import sys
import traceback
from pathlib import Path

sys.path.insert(0, '.')
from atsim.web.server import App
from atsim.web.session import Project, SessionStore
from atsim.render.renderer import Renderer
from atsim.io.ffmpeg import VideoWriter

RUNS = Path('runs')
app = App(RUNS)

CLIPS = [p.name for p in sorted((RUNS / 'uploads').glob('*')) if p.is_file()]
print('clips available: %s' % CLIPS)
print()

for name in CLIPS:
    print('=' * 100)
    print('  %s' % name)
    print('=' * 100)
    try:
        app.set_project({"source": name})
    except Exception:
        print('  set_project FAILED')
        traceback.print_exc()
        continue
    proj = app.current
    print('  project source: %r   fps %s   standard %s' % (proj.source, proj.fps, proj.standard))
    for scale in (1.0, 0.5):
        try:
            store = SessionStore(app.store.data_dir)
            store.projects[proj.id] = proj
            src = store.source_for((proj.id, scale), proj)
            w = int(getattr(src, 'width', 720))
            h = int(getattr(src, 'height', 480))
            total = max(1, int(getattr(src, 'frame_count', 1)))
            print('  scale %.1f -> source %dx%d, %d frames' % (scale, w, h, total))
            r = Renderer(src, w, h, proj.fps, proj.resolved_params(),
                         seed=proj.seed, std_key=proj.standard,
                         preview=(scale < 1.0), scale=1.0)
            out = RUNS / 'renders' / ('repro_%s_%.1f.mp4' % (Path(name).stem[:12], scale))
            with VideoWriter(out, w, h, proj.fps, crf=16, preset='medium') as vw:
                clip = app.store.clip_path(proj.source)
                vw.open(audio_from=str(clip) if clip.exists() else None)
                n = 0
                for i, rgb in r.render_frames(list(range(min(total, 12)))):
                    vw.write(rgb)
                    n += 1
            print('     OK -- wrote %d frames to %s (%d bytes)' % (n, out.name, out.stat().st_size))
        except Exception as exc:
            print('     FAILED: %s: %s' % (type(exc).__name__, exc))
            traceback.print_exc()
    print()

import sys, urllib.request, json, os
sys.path.insert(0, ".")
# make a small test clip to upload
from atsim.io.ffmpeg import VideoWriter
import numpy as np
clip = "tools/out/_uploadtest.mp4"
with VideoWriter(clip, 160, 120, 25.0) as w:
    w.open()
    for i in range(15):
        w.write(np.full((120,160,3), i/15.0, np.float32))
print("test clip:", os.path.getsize(clip), "bytes")

base = "http://127.0.0.1:8941"
def upload(path, name):
    with open(path, "rb") as fh:
        data = fh.read()
    req = urllib.request.Request(base + "/api/upload", data=data, method="POST",
                                 headers={"X-Filename": name})
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")[:400]

for name in ("plain.mp4", "TV%20CAM%20%E8%AE%BE%E5%A4%87_20261010_210428.mp4"):
    code, body = upload(clip, name)
    print("X-Filename=%r -> %s" % (name, code))
    print("   ", str(body)[:260])

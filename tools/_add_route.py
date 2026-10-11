"""Serve the connector test bench at /connectors."""
from __future__ import annotations

import ast
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
P = os.path.join(ROOT, "atsim", "web", "server.py")
src = open(P, encoding="utf-8").read()

if "/connectors" in src:
    print("route already present")
    raise SystemExit(0)

anchor = '''                if path in ("/", "/index.html"):
                    return self._static("index.html")'''
assert anchor in src, "static-route anchor not found"
add = anchor + '''
                # The connector test bench: every interface degraded side by side against a
                # clean baseline, so the differences between them can be read directly
                # instead of inferred from one picture at a time.
                if path in ("/connectors", "/connectors.html"):
                    return self._static("connectors.html")'''
src = src.replace(anchor, add, 1)

try:
    ast.parse(src)
except SyntaxError as exc:
    print("REFUSING: does not parse -- line %s: %s" % (exc.lineno, exc.msg))
    raise SystemExit(1)
open(P, "w", encoding="utf-8").write(src)
print("route added")
print("   /connectors -> static/connectors.html")

"""Convert an ASCII STL to binary STL, in place, with no geometry change.

The supplied robot/assets/lamp_shade.stl is valid ASCII STL, but both
PyBullet's and MuJoCo's mesh loaders only reliably parse binary STL
("cannot extract anything useful from mesh" / "perhaps this is an ASCII
file?"). This is a loader-compatibility issue, not a geometry problem, so we
just re-encode the same triangles as binary STL.

Usage: python scripts/stl_ascii_to_binary.py robot/assets/lamp_shade.stl
"""

from __future__ import annotations

import struct
import sys


def parse_ascii_stl(text: str):
    facets = []
    normal = None
    verts: list[tuple[float, float, float]] = []
    for line in text.splitlines():
        tokens = line.split()
        if not tokens:
            continue
        if tokens[0] == "facet" and tokens[1] == "normal":
            normal = tuple(float(x) for x in tokens[2:5])
            verts = []
        elif tokens[0] == "vertex":
            verts.append(tuple(float(x) for x in tokens[1:4]))
        elif tokens[0] == "endfacet":
            facets.append((normal, verts))
    return facets


def write_binary_stl(path: str, facets) -> None:
    with open(path, "wb") as f:
        header = b"binary STL converted from supplied ASCII STL (geometry unchanged)"
        f.write(header.ljust(80, b"\0")[:80])
        f.write(struct.pack("<I", len(facets)))
        for normal, verts in facets:
            f.write(struct.pack("<3f", *normal))
            for v in verts:
                f.write(struct.pack("<3f", *v))
            f.write(struct.pack("<H", 0))


def main() -> None:
    path = sys.argv[1]
    with open(path, "r") as f:
        text = f.read()
    facets = parse_ascii_stl(text)
    if not facets:
        raise SystemExit(f"No facets parsed from {path} -- is it really ASCII STL?")
    write_binary_stl(path, facets)
    print(f"Rewrote {path} as binary STL: {len(facets)} facets")


if __name__ == "__main__":
    main()

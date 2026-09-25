#!/usr/bin/env bash
# PyBullet ships no macOS wheels at all (PyPI only has manylinux + pypy), so
# `pip install pybullet` always builds from source here. On recent macOS
# SDKs (Xcode's clang), that build fails: pybullet's vendored zlib defines
# `#define fdopen(fd, mode) NULL` for "classic Mac OS" (guarded by
# `TARGET_OS_MAC`, which is actually defined on ALL Apple platforms today,
# not just pre-OSX Mac), which corrupts the real libc `fdopen` declaration
# when the macOS SDK's <stdio.h> is parsed afterward.
#
# This script downloads the pybullet sdist, patches that one line, and
# installs the patched source into the given venv -- so the fix is scripted,
# not a one-off manual edit.
#
# Usage: scripts/install_pybullet_macos.sh .venv/bin/python

set -euo pipefail

PYTHON_BIN="${1:?Usage: $0 <path-to-venv-python>}"
PYBULLET_VERSION="3.2.7"
WORKDIR="$(mktemp -d)"

echo "Downloading pybullet ${PYBULLET_VERSION} source..."
curl -sL -o "${WORKDIR}/pybullet.tar.gz" \
  "https://files.pythonhosted.org/packages/source/p/pybullet/pybullet-${PYBULLET_VERSION}.tar.gz"
tar xzf "${WORKDIR}/pybullet.tar.gz" -C "${WORKDIR}"
SRC_DIR="${WORKDIR}/pybullet-${PYBULLET_VERSION}"
ZUTIL_H="${SRC_DIR}/examples/ThirdPartyLibs/zlib/zutil.h"

echo "Patching vendored zlib's fdopen macro for modern macOS SDKs..."
python3 - "$ZUTIL_H" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path) as f:
    text = f.read()
old = "#ifndef fdopen\n#define fdopen(fd, mode) NULL /* No fdopen() */\n#endif"
new = "#ifndef fdopen\n#if !defined(__APPLE__)\n#define fdopen(fd, mode) NULL /* No fdopen() */\n#endif\n#endif"
if new in text:
    print("Already patched.")
elif old in text:
    text = text.replace(old, new, 1)
    with open(path, "w") as f:
        f.write(text)
    print("Patched.")
else:
    raise SystemExit(f"Expected fdopen macro not found in {path} -- pybullet source may have changed.")
PYEOF

UV_BIN="$(command -v uv || echo "$HOME/.local/bin/uv")"
"$UV_BIN" pip install --python "$PYTHON_BIN" wheel setuptools -q
"$UV_BIN" pip install --python "$PYTHON_BIN" "$SRC_DIR" --no-build-isolation

rm -rf "$WORKDIR"
echo "Done: pybullet ${PYBULLET_VERSION} installed from patched source."

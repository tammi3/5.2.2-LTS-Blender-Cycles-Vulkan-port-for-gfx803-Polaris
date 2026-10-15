#!/usr/bin/env bash
# Runs cycles_bvh_dump with Cycles' library DLLs on PATH.
# usage: run_bvh_dump.sh <scene.xml> <out_dir>
CYCLES=/c/dev/cycles
for d in $(find "$CYCLES/lib/windows_x64" -name "*.dll" | sed 's#/[^/]*$##' | sort -u); do
  PATH="$PATH:$d"
done
exec "$CYCLES/build/bin/Release/cycles_bvh_dump.exe" "$@"

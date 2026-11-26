#!/usr/bin/env bash
# Push one container archive from the jump box into the cluster's image registry so Run:ai pods can pull it.
# NOT RUN OR TESTED. It needs three things that are NOT in this package:
#   1. the registry host and repository path (ask Matthew Pitterle or Michael Schwarz)
#   2. a login for it:   skopeo login <registry>   (type the password yourself; never put it in a file)
#   3. trust for the registry CA (the BCM registry uses a private CA)
# /home on this box is ~10 GB, so everything below keeps its data under /opt/data.
#
# Usage:  bash jumpbox-stage-containers.sh <archive.tar> <registry-host/path/name:tag>
set -euo pipefail
ARCHIVE=${1:?usage: $0 <archive.tar> <registry-host/path/name:tag>}
DEST=${2:?usage: $0 <archive.tar> <registry-host/path/name:tag>}
export TMPDIR=${TMPDIR:-/opt/data/nvis/tmp}; mkdir -p "$TMPDIR"
if command -v skopeo >/dev/null 2>&1; then
  # Copies the archive straight to the registry with no local image store.
  # Use oci-archive: instead of docker-archive: if the tarball is an OCI layout.
  skopeo copy "docker-archive:${ARCHIVE}" "docker://${DEST}"
else
  ROOT=/opt/data/nvis/podman-root; RUNROOT=/opt/data/nvis/podman-run; mkdir -p "$ROOT" "$RUNROOT"
  podman --root "$ROOT" --runroot "$RUNROOT" load -i "$ARCHIVE"
  echo "Now tag and push the loaded image:"
  echo "  podman --root $ROOT --runroot $RUNROOT images"
  echo "  podman --root $ROOT --runroot $RUNROOT tag <loaded-image> ${DEST}"
  echo "  podman --root $ROOT --runroot $RUNROOT push ${DEST}"
fi

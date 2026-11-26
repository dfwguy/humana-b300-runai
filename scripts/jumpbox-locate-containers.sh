#!/usr/bin/env bash
# Read-only. Run on the jump box to find where the team's containers and files actually are.
# Usage: bash jumpbox-locate-containers.sh | tee ~/locate.txt   (or tee into /opt/data/nvis/ - /home is nearly full)
set -u
echo "== host";  hostname -f; date -u
echo; echo "== disk";  df -h /home /opt/data /vast/validation 2>&1
echo; echo "== NFS/VAST mounts visible from this box"; mount | grep -iE 'vast|nfs' || echo "none: this box may not be on the VAST network"
echo; echo "== project tree"; ls -la /opt/data/nvis 2>&1; ls -la /opt/data/nvis/humana-airgapped 2>&1
echo; echo "== files larger than 1 GB under /opt/data (size, path)"
find /opt/data -xdev -type f -size +1G -printf '%s\t%p\n' 2>/dev/null | sort -rn | head -60 | awk -F'\t' '{printf "%.1f GB\t%s\n", $1/1073741824, $2}'
echo; echo "== archive-like files"
find /opt/data -maxdepth 6 \( -name '*.tar' -o -name '*.tar.gz' -o -name '*.tgz' -o -name '*.sqsh' -o -name '*.oci' -o -name '*.zst' \) 2>/dev/null | head -80
echo; echo "== local image stores (podman)"; podman images 2>&1 | head -40
echo; echo "== tools"; for t in skopeo podman docker kubectl runai python3; do printf '%-9s' "$t"; command -v "$t" || echo "not found"; done
echo; echo "== runai wrapper (Dejaun's workaround lives under /var/tmp)"; ls -la /var/tmp 2>/dev/null | head -20

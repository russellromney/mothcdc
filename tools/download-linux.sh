#!/usr/bin/env sh

MIRROR_URL=${MIRROR_URL:-"http://linux-kernel.uio.no/pub/linux/kernel/v6.x/"}

for i in $(seq 0 18); do
    curl -O "${MIRROR_URL}linux-6.${i}.tar.xz"
    xz -d "linux-6.${i}.tar.xz"
done

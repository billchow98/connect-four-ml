#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

if [ $# -ne 1 ]; then
	>&2 printf 'usage: upload_run.sh run_name'
	exit 1
fi

if ! find networks -type d -name "run$1_*" | grep -q .; then
	>&2 printf 'run %s not found in networks/!\n' "$1"
	exit 1
fi

set -x

RUN_NAME=$(find networks/ -type d -name "run$1_*" -execdir basename '{}' \;)

git add "animations/$RUN_NAME/"

find "animations/$RUN_NAME/" -type f -name '*.svg' -execdir basename '{}' '.svg' \; | while read -r i; do
	git add "checkpoints/$RUN_NAME/$i/"
done

find "checkpoints/$RUN_NAME/" -maxdepth 1 -type d -execdir basename '{}' \; | sort -n | tail -n 3 | while read -r i; do
	git add "checkpoints/$RUN_NAME/$i/"
done

cp "$RUN_NAME.log" "networks/$RUN_NAME/"

git add "networks/$RUN_NAME/"

git add "tensorboard/$RUN_NAME/"


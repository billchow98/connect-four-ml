#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

. .venv/bin/activate

if [ $# -lt 2 ]; then
	>&2 printf 'usage: 3_run_trainer.sh run_name device_num_1[,device_num_2[,device_num3...]]\n'
	exit 1
fi

if ! find networks -type d -name "run$1_*" | grep -q .; then
	>&2 printf 'run %s not found in networks/!\n' "$1"
	exit 1
fi

if ! printf '%s' "$2" | grep -qE '^[0-9]+(,[0-9]+)*$'; then
	>&2 printf 'invalid syntax for device ids\n'
	exit 1
fi

if [ -n "${XLA_FLAGS-}" ]; then
	export XLA_FLAGS="$XLA_FLAGS --xla_gpu_deterministic_ops=true"
else
	export XLA_FLAGS="--xla_gpu_deterministic_ops=true"
fi

export DEVICES="$2"

[ -f config.py ] && mv config.py config.py.bak
[ -f network.py ] && mv network.py network.py.bak

RUN_NAME="$(find networks/ -type d -name "run$1_*" -execdir basename '{}' \;)"
cp "networks/$RUN_NAME/config.py" .
cp "networks/$RUN_NAME/network.py" .

python3 calculate_flops.py
python3 main.py

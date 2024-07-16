#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

. .venv/bin/activate

MAX_RUN=$(find networks -type d -name 'run*' | sed -E 's|^networks/run([0-9]+)_.*|\1|' | sort -n | tail -n 1)

rm -f /tmp/tournament_results.pgn

for ((i=1;i<=MAX_RUN;i++)); do
	for ((j=i+1;j<=MAX_RUN;j++)); do
		BASELINE_RUN_NAME=$(find networks -type d -name "run${i}_*" -exec basename {} \;)
		TEST_RUN_NAME=$(find networks -type d -name "run${j}_*" -exec basename {} \;)
		python evaluate.py "$BASELINE_RUN_NAME" "$TEST_RUN_NAME"
		cat /tmp/results.pgn >> /tmp/tournament_results.pgn
	done
done

./calc_bayeselo.sh /tmp/tournament_results.pgn

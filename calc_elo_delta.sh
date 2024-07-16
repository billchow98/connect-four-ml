#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

PGN_FILE=/tmp/results.pgn
printf 'prompt off\nreadpgn %s\nelo\nmm\nratings\n' "$PGN_FILE" \
| ./bayeselo                                                     \
| tail -n 3                                                      \
| grep "$1"                                                      \
| awk '{ printf ("%d %d %d", $3, $4, $5) }'

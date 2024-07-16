#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

printf 'prompt off\nreadpgn %s\nelo\nmm\nratings\n' "$1" | ./bayeselo

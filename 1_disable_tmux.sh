#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

touch ~/.no_auto_tmux
printf "Please reconnect through SSH if you haven't done so already to disable tmux\n"

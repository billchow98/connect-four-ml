#!/bin/bash

set -euxo pipefail

touch ~/.no_auto_tmux
printf "Please reconnect through SSH if you haven't done so already to disable tmux\n"

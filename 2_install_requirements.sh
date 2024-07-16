#!/bin/bash

# © 2024 Bill Chow. All rights reserved.
# Unauthorized use, modification, or distribution of this code is strictly prohibited.

set -euxo pipefail

sudo apt install python3-pip python3-venv byobu neovim coreutils grep gawk -y

byobu-enable

python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

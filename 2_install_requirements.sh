#!/bin/bash

set -euxo pipefail

sudo apt install python3-pip python3-venv byobu neovim coreutils grep gawk -y

byobu-enable

python3 -m venv .venv
. .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

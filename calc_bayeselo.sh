#!/bin/bash

set -euxo pipefail

printf 'prompt off\nreadpgn %s\nelo\nmm\nratings\n' "$1" | ./bayeselo

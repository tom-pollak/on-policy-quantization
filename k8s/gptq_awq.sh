#!/bin/bash
set -euo pipefail
source ~/.bashrc
source "$(dirname "$0")/lib.sh"

submit_job "gptq-awq-eval" "uv run python gptq_awq.py" 1

#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Set OPENAI_API_KEY before running. OPENAI_BASE_URL and OPENAI_UID are optional.


# Research domain
# python cross_session_generator.py \
#   --domain research \
#   --topics_file cross_session_topics_research.json \
#   --mix_dir ../timeline_generation/output_gemini \
#   --regimes_file all_pref_regimes.json \
#   --output_dir output \
#   --model gemini-2.5-flash \
#   --request_type openai \
#   --sleep_s 0.0 \
#   --skip_existing \
#   --max_workers 16 \
#   --num_interactions 5
# #   --start_from 0 \
# #   --limit 1

# Tutoring domain
python cross_session_generator.py \
  --domain tutoring \
  --topics_file cross_session_topics_tutoring.json \
  --mix_dir ../timeline_generation/output_gemini \
  --regimes_file all_pref_regimes.json \
  --output_dir output \
  --model gemini-2.5-flash \
  --request_type openai \
  --sleep_s 0.0 \
  --skip_existing \
  --max_workers 16 \
  --num_interactions 5
#   --start_from 0 \


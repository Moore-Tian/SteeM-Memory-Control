#!/bin/bash

# Unified evaluation pipeline: inference -> judge -> metric

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HELPER_DIR="$SCRIPT_DIR"

MODEL_PATH="${MODEL_PATH:-path/to/model}"
TEST_DATA_PATH="${TEST_DATA_PATH:-path/to/test_data.jsonl}"
SETTING="${SETTING:-rewritten_query}"
SYSTEM_FIELD="${SYSTEM_FIELD:-filtered_context}"
USER_QUERY_FIELD="${USER_QUERY_FIELD:-$SETTING}"
JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$MODEL_PATH}"
RUN_JUDGE="${RUN_JUDGE:-0}"
RUN_METRIC="${RUN_METRIC:-0}"

OUTPUT_DIR="${SCRIPT_DIR}/outputs"
JUDGED_DIR="${SCRIPT_DIR}/judged_outputs"
METRIC_DIR="${SCRIPT_DIR}/metric_results"

MODEL_NAME=$(basename "$MODEL_PATH")

mkdir -p "$OUTPUT_DIR"
mkdir -p "$JUDGED_DIR"
mkdir -p "$METRIC_DIR"

INFERENCE_OUTPUT="${OUTPUT_DIR}/${MODEL_NAME}_${SETTING}.jsonl"
JUDGE_OUTPUT="${JUDGED_DIR}/judged_${MODEL_NAME}_${SETTING}.jsonl"
METRIC_OUTPUT="${METRIC_DIR}/metric_${MODEL_NAME}_${SETTING}.json"

echo "=========================================="
echo "Evaluation Pipeline"
echo "=========================================="
echo "Model: $MODEL_NAME"
echo "Test Data: $TEST_DATA_PATH"
echo "Setting: $SETTING"
echo "Helper Directory: $HELPER_DIR"
echo "Output Directory: $OUTPUT_DIR"
echo "Judged Directory: $JUDGED_DIR"
echo "Metric Directory: $METRIC_DIR"
echo "=========================================="
echo ""

echo "[STEP 1/3] Running inference..."
echo "  Input: $TEST_DATA_PATH"
echo "  Output: $INFERENCE_OUTPUT"
echo ""

python "${HELPER_DIR}/run_model_inference.py" \
    --input_jsonl "$TEST_DATA_PATH" \
    --model_path "$MODEL_PATH" \
    --output_jsonl "$INFERENCE_OUTPUT" \
    --system_field "$SYSTEM_FIELD" \
    --user_query_field "$USER_QUERY_FIELD" \
    --temperature 0.7 \
    --top_p 0.8 \
    --max_new_tokens 8192 \
    --tensor_parallel_size 1 \
    --batch_size 500

echo "[STEP 1/3] Inference completed!"
echo ""

if [ "$RUN_JUDGE" = "1" ]; then
    echo "[STEP 2/3] Running judge..."
    echo "  Input: $INFERENCE_OUTPUT"
    echo "  Output: $JUDGE_OUTPUT"
    echo ""

    python "${HELPER_DIR}/judge_memory_dependence.py" \
        --input_jsonl "$INFERENCE_OUTPUT" \
        --model_path "$JUDGE_MODEL_PATH" \
        --output_jsonl "$JUDGE_OUTPUT" \
        --task_field task \
        --context_field context \
        --query_field query \
        --answer_field generated_text \
        --batch_size 500 \
        --max_new_tokens 8192 \
        --max_retries 5 \
        --temperature 0.2 \
        --top_p 0.9

    echo "[STEP 2/3] Judge completed!"
    echo ""
fi

if [ "$RUN_METRIC" = "1" ]; then
    echo "[STEP 3/3] Computing metrics..."
    echo "  Input: $JUDGE_OUTPUT"
    echo "  Output: $METRIC_OUTPUT"
    echo ""

    python "${HELPER_DIR}/compute_dependence_metrics.py" \
        --input_jsonl "$JUDGE_OUTPUT" \
        --output_json "$METRIC_OUTPUT"

    echo "[STEP 3/3] Metrics computed!"
    echo ""
fi

echo "=========================================="
echo "Evaluation Pipeline Finished"
echo "=========================================="
echo "Inference output: $INFERENCE_OUTPUT"
if [ "$RUN_JUDGE" = "1" ]; then
    echo "Judge output: $JUDGE_OUTPUT"
fi
if [ "$RUN_METRIC" = "1" ]; then
    echo "Metric output: $METRIC_OUTPUT"
fi
echo "=========================================="


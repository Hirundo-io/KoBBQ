#!/bin/bash

# Final evaluation script for Gemma-SEA-LION-v4-27B-IT-bias-reduced
# Run this after post-processing is complete

set -e

cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

MODEL="hirundo-io_Gemma-SEA-LION-v4-27B-IT-bias-reduced"

echo "====================================="
echo "Stage 5: Final Evaluation"
echo "====================================="
echo ""

mkdir -p evaluation_result

for PROMPT_ID in {1..5}; do
  echo "Evaluating prompt $PROMPT_ID..."
  
  python3 5_evaluation.py \
    --evaluation-result-path evaluation_result/KoBBQ_test_$PROMPT_ID.tsv \
    --model-result-tsv-dir outputs/processed/KoBBQ_test_$PROMPT_ID \
    --topic KoBBQ_test_evaluation \
    --test-or-all test \
    --prompt-tsv-path 0_evaluation_prompts.tsv \
    --prompt-id $PROMPT_ID \
    --models $MODEL
  
  echo "✓ Prompt $PROMPT_ID evaluation complete"
  echo ""
done

echo "====================================="
echo "Evaluation Complete!"
echo "====================================="
echo ""
echo "Results available in:"
echo "  evaluation_result/KoBBQ_test_1.tsv"
echo "  evaluation_result/KoBBQ_test_2.tsv"
echo "  evaluation_result/KoBBQ_test_3.tsv"
echo "  evaluation_result/KoBBQ_test_4.tsv"
echo "  evaluation_result/KoBBQ_test_5.tsv"
echo ""
echo "Metrics calculated:"
echo "  - Accuracy in ambiguous contexts"
echo "  - Accuracy in disambiguated contexts"
echo "  - Diff-bias in ambiguous contexts"
echo "  - Diff-bias in disambiguated contexts"
echo "  - Out-of-choice ratio"


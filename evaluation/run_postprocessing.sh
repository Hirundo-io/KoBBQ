#!/bin/bash

# Post-processing script for Gemma-SEA-LION-v4-27B-IT-bias-reduced
# Run this after all inference is complete

set -e

cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

MODEL="hirundo-io_Gemma-SEA-LION-v4-27B-IT-bias-reduced"

echo "====================================="
echo "Stage 3 & 4: Post-processing"
echo "====================================="
echo ""

for PROMPT_ID in {1..5}; do
  echo "Processing prompt $PROMPT_ID..."
  
  # Stage 3: Convert raw predictions
  python3 3_postprocess_predictions.py \
    --predictions-tsv-path outputs/raw/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_${PROMPT_ID}_${MODEL}_predictions.tsv \
    --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv
  
  # Stage 4: Prepare for evaluation
  mkdir -p outputs/processed/KoBBQ_test_$PROMPT_ID
  python3 4_predictions_to_evaluation.py \
    --predictions-tsv-path outputs/raw/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_${PROMPT_ID}_${MODEL}_predictions.tsv \
    --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv \
    --output-path outputs/processed/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_${PROMPT_ID}_${MODEL}.tsv
  
  echo "✓ Prompt $PROMPT_ID post-processing complete"
  echo ""
done

echo "====================================="
echo "Post-processing complete!"
echo "====================================="


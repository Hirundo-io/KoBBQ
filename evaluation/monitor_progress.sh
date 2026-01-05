#!/bin/bash

# Monitor progress of all 5 prompts
echo "====================================="
echo "KoBBQ Evaluation Progress Monitor"
echo "Target: 6,840 samples per prompt"
echo "====================================="
echo ""

cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

for i in {1..5}; do 
  echo "Prompt $i:"
  FILE="outputs/raw/KoBBQ_test_$i/KoBBQ_test_evaluation_${i}_hirundo-io_Gemma-SEA-LION-v4-27B-IT-bias-reduced_predictions.tsv"
  
  if [ -f "$FILE" ]; then
    # Use pandas to properly count rows (handles multi-line fields)
    COUNT=$(python3 -c "import pandas as pd; df = pd.read_csv('$FILE', sep='\t'); print(len(df))")
    PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COUNT/6840)*100}")
    
    if [ "$COUNT" -eq 6840 ]; then
      echo "  ✅ COMPLETE: $COUNT / 6,840 samples ($PERCENT%)"
    else
      echo "  🔄 $COUNT / 6,840 samples ($PERCENT%)"
    fi
  else
    echo "  ⏳ Not started yet"
  fi
  echo ""
done

echo "====================================="
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
  awk '{printf "  GPU: %s%% | Memory: %s / %s GB\n", $1, int($2/1024), int($3/1024)}'
echo ""

echo "Running processes:"
ps aux | grep "2_model_inference.py" | grep -v grep | wc -l | xargs echo "  Active inference processes:"
echo "====================================="


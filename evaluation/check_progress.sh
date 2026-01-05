#!/bin/bash
#
# Quick progress checker for running evaluations
#

cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate 2>/dev/null

MODEL_NAME="${1:-aisingapore_Gemma-SEA-LION-v4-27B-IT}"

echo "========================================"
echo "Evaluation Progress: $MODEL_NAME"
echo "Target: 6,840 samples per prompt"
echo "========================================"
echo ""

TOTAL=0
COMPLETE=0

for i in {1..5}; do
    FILE="outputs/$MODEL_NAME/raw/prompt_$i/predictions.tsv"
    
    if [ -f "$FILE" ]; then
        COUNT=$(python3 -c "import pandas as pd; df = pd.read_csv('$FILE', sep='\t'); print(len(df))" 2>/dev/null || echo "0")
        PERCENT=$(awk "BEGIN {printf \"%.1f\", ($COUNT/6840)*100}")
        TOTAL=$((TOTAL + COUNT))
        
        if [ "$COUNT" -eq 6840 ]; then
            echo "Prompt $i: ✅ COMPLETE ($COUNT/6,840)"
            COMPLETE=$((COMPLETE + 1))
        else
            echo "Prompt $i: 🔄 $COUNT/6,840 ($PERCENT%)"
        fi
    else
        echo "Prompt $i: ⏳ Not started"
    fi
done

echo ""
echo "========================================"
echo "Overall: $COMPLETE/5 prompts complete"
echo "Total samples processed: $TOTAL/34,200"
OVERALL_PERCENT=$(awk "BEGIN {printf \"%.1f\", ($TOTAL/34200)*100}")
echo "Overall progress: $OVERALL_PERCENT%"
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits | \
    awk '{printf "  Utilization: %s%% | Memory: %d/%d GB\n", $1, int($2/1024), int($3/1024)}'

echo ""

# Running processes
PROC_COUNT=$(ps aux | grep "2_model_inference.py" | grep -v grep | wc -l)
if [ "$PROC_COUNT" -gt 0 ]; then
    echo "Status: ✅ Inference running ($PROC_COUNT process(es))"
else
    echo "Status: ⚠️  No inference processes detected"
fi

echo "========================================"


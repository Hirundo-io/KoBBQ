#!/bin/bash
#
# KoBBQ Full Evaluation Pipeline
# ==============================
# Runs complete evaluation from preprocessing to final results for any HuggingFace model.
#
# Usage:
#   ./run_full_evaluation.sh <model_name> [batch_size]
#
# Examples:
#   ./run_full_evaluation.sh hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced
#   ./run_full_evaluation.sh meta-llama/Llama-2-7b-chat-hf 16
#   ./run_full_evaluation.sh Qwen/Qwen2-7B-Instruct 32
#
# For long-running evaluations, use nohup to prevent interruption:
#   nohup ./run_full_evaluation.sh <model_name> 32 > eval.log 2>&1 &
#   tail -f eval.log  # To monitor progress
#
# Output structure:
#   outputs/<model_name_safe>/
#   ├── raw/                    # Raw model predictions
#   │   ├── prompt_1/
#   │   ├── prompt_2/
#   │   └── ...
#   ├── processed/              # Post-processed predictions
#   │   ├── prompt_1/
#   │   └── ...
#   └── results/                # Final evaluation results
#       ├── KoBBQ_test_1.tsv
#       ├── KoBBQ_test_2.tsv
#       └── summary.txt
#

# Don't exit on error - we handle errors manually for long-running processes
# set -e

# ============================================================================
# Configuration
# ============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate virtual environment
if [ -f "/home/vscode/kobbq/.venv/bin/activate" ]; then
    source /home/vscode/kobbq/.venv/bin/activate
elif [ -f "../.venv/bin/activate" ]; then
    source ../.venv/bin/activate
fi

# Parse arguments
MODEL_NAME="${1:-}"
BATCH_SIZE="${2:-32}"

if [ -z "$MODEL_NAME" ]; then
    echo "Error: Model name is required"
    echo ""
    echo "Usage: ./run_full_evaluation.sh <model_name> [batch_size]"
    echo ""
    echo "Examples:"
    echo "  ./run_full_evaluation.sh hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced"
    echo "  ./run_full_evaluation.sh meta-llama/Llama-2-7b-chat-hf 16"
    exit 1
fi

# Create safe model name for file paths (replace / with _)
MODEL_NAME_SAFE=$(echo "$MODEL_NAME" | tr '/' '_')

# Output directories
OUTPUT_BASE="outputs/$MODEL_NAME_SAFE"
RAW_DIR="$OUTPUT_BASE/raw"
PROCESSED_DIR="$OUTPUT_BASE/processed"
RESULTS_DIR="$OUTPUT_BASE/results"

# Data paths
DATA_DIR="data/KoBBQ_test"
SAMPLES_PATH="../data/KoBBQ_test_samples.tsv"
PROMPTS_PATH="0_evaluation_prompts.tsv"

# Number of samples per prompt (for progress tracking)
TOTAL_SAMPLES=6840

# ============================================================================
# Helper Functions
# ============================================================================

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

print_step() {
    echo ">>> $1"
}

print_success() {
    echo "✅ $1"
}

print_progress() {
    local current=$1
    local total=$2
    local pct=$((current * 100 / total))
    printf "\r    Progress: %d/%d (%d%%)" "$current" "$total" "$pct"
}

check_samples_complete() {
    local file=$1
    if [ -f "$file" ]; then
        local count=$(python3 -c "import pandas as pd; df = pd.read_csv('$file', sep='\t'); print(len(df))" 2>/dev/null || echo "0")
        if [ "$count" -eq "$TOTAL_SAMPLES" ]; then
            return 0
        fi
    fi
    return 1
}

# ============================================================================
# Main Script
# ============================================================================

print_header "KoBBQ Full Evaluation Pipeline"

echo "Configuration:"
echo "  Model:        $MODEL_NAME"
echo "  Batch Size:   $BATCH_SIZE"
echo "  Output Dir:   $OUTPUT_BASE"
echo ""

# Create output directories
mkdir -p "$DATA_DIR"
mkdir -p "$RAW_DIR"
mkdir -p "$PROCESSED_DIR"
mkdir -p "$RESULTS_DIR"

for i in {1..5}; do
    mkdir -p "$RAW_DIR/prompt_$i"
    mkdir -p "$PROCESSED_DIR/prompt_$i"
done

# ============================================================================
# Stage 1: Preprocessing
# ============================================================================

print_header "Stage 1: Preprocessing"

PREPROCESS_NEEDED=false
for PROMPT_ID in {1..5}; do
    if [ ! -f "$DATA_DIR/KoBBQ_test_evaluation_$PROMPT_ID.json" ]; then
        PREPROCESS_NEEDED=true
        break
    fi
done

if [ "$PREPROCESS_NEEDED" = true ]; then
    print_step "Preprocessing data for all 5 prompts..."
    
    for PROMPT_ID in {1..5}; do
        python3 1_preprocess.py \
            --samples-tsv-path "$SAMPLES_PATH" \
            --evaluation-tsv-path "$DATA_DIR/KoBBQ_test_evaluation_$PROMPT_ID.tsv" \
            --evaluation-json-path "$DATA_DIR/KoBBQ_test_evaluation_$PROMPT_ID.json" \
            --prompt-tsv-path "$PROMPTS_PATH" \
            --prompt-id "$PROMPT_ID" > /dev/null 2>&1
        echo "    Prompt $PROMPT_ID: Done"
    done
    
    print_success "Preprocessing complete"
else
    print_success "Preprocessing already complete (using cached data)"
fi

# ============================================================================
# Stage 2: Model Inference
# ============================================================================

print_header "Stage 2: Model Inference"

for PROMPT_ID in {1..5}; do
    PRED_FILE="$RAW_DIR/prompt_$PROMPT_ID/predictions.tsv"
    
    if check_samples_complete "$PRED_FILE"; then
        print_success "Prompt $PROMPT_ID: Already complete (6840 samples)"
        continue
    fi
    
    print_step "Running inference for Prompt $PROMPT_ID..."
    
    # Remove partial file if exists
    rm -f "$PRED_FILE"
    
    # Run inference directly (no pipe - more robust for long-running processes)
    python3 2_model_inference.py \
        --data-path "$DATA_DIR/KoBBQ_test_evaluation_$PROMPT_ID.json" \
        --output-dir "$RAW_DIR/prompt_$PROMPT_ID" \
        --output-filename "predictions.tsv" \
        --model-name "$MODEL_NAME" \
        --batch-size "$BATCH_SIZE"
    
    # Verify completion
    if check_samples_complete "$PRED_FILE"; then
        print_success "Prompt $PROMPT_ID: Inference complete (6840 samples)"
    else
        echo "⚠️  Prompt $PROMPT_ID: Inference may be incomplete, check $PRED_FILE"
    fi
done

# ============================================================================
# Stage 3 & 4: Post-processing
# ============================================================================

print_header "Stage 3 & 4: Post-processing"

for PROMPT_ID in {1..5}; do
    PRED_FILE="$RAW_DIR/prompt_$PROMPT_ID/predictions.tsv"
    PROCESSED_FILE="$PROCESSED_DIR/prompt_$PROMPT_ID/evaluation.tsv"
    
    if [ -f "$PROCESSED_FILE" ]; then
        print_success "Prompt $PROMPT_ID: Already post-processed"
        continue
    fi
    
    print_step "Post-processing Prompt $PROMPT_ID..."
    
    # Stage 3: Convert raw predictions
    if ! python3 3_postprocess_predictions.py \
        --predictions-tsv-path "$PRED_FILE" \
        --preprocessed-tsv-path "$DATA_DIR/KoBBQ_test_evaluation_$PROMPT_ID.tsv" 2>&1 | grep -E "(out-of-choice|Error)" ; then
        true  # Suppress grep's exit code when no match
    fi
    
    # Stage 4: Create evaluation file
    if python3 4_predictions_to_evaluation.py \
        --predictions-tsv-path "$PRED_FILE" \
        --preprocessed-tsv-path "$DATA_DIR/KoBBQ_test_evaluation_$PROMPT_ID.tsv" \
        --output-path "$PROCESSED_FILE" 2>&1; then
        print_success "Prompt $PROMPT_ID: Post-processing complete"
    else
        echo "⚠️  Prompt $PROMPT_ID: Post-processing failed"
    fi
done

# ============================================================================
# Stage 5: Evaluation
# ============================================================================

print_header "Stage 5: Final Evaluation"

# We need to create a temporary directory structure that 5_evaluation.py expects
TEMP_EVAL_DIR="$OUTPUT_BASE/temp_eval"
mkdir -p "$TEMP_EVAL_DIR"

for PROMPT_ID in {1..5}; do
    print_step "Evaluating Prompt $PROMPT_ID..."
    
    # Create temp directory structure for this prompt
    TEMP_PROMPT_DIR="$TEMP_EVAL_DIR/prompt_$PROMPT_ID"
    mkdir -p "$TEMP_PROMPT_DIR"
    
    # Copy processed file with expected naming
    cp "$PROCESSED_DIR/prompt_$PROMPT_ID/evaluation.tsv" \
       "$TEMP_PROMPT_DIR/KoBBQ_test_evaluation_${PROMPT_ID}_${MODEL_NAME_SAFE}.tsv"
    
    # Run evaluation
    python3 5_evaluation.py \
        --evaluation-result-path "$RESULTS_DIR/KoBBQ_test_$PROMPT_ID.tsv" \
        --model-result-tsv-dir "$TEMP_PROMPT_DIR" \
        --topic "KoBBQ_test_evaluation" \
        --test-or-all "test" \
        --prompt-tsv-path "$PROMPTS_PATH" \
        --prompt-id "$PROMPT_ID" \
        --models "$MODEL_NAME_SAFE" > /dev/null 2>&1
    
    print_success "Prompt $PROMPT_ID: Evaluation complete"
done

# Clean up temp directory
rm -rf "$TEMP_EVAL_DIR"

# ============================================================================
# Generate Summary
# ============================================================================

print_header "Generating Summary"

SUMMARY_FILE="$RESULTS_DIR/summary.txt"

python3 << EOF > "$SUMMARY_FILE"
import pandas as pd
from datetime import datetime

print("=" * 80)
print("KOBBQ EVALUATION RESULTS SUMMARY")
print("=" * 80)
print()
print(f"Model:     $MODEL_NAME")
print(f"Date:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Dataset:   KoBBQ Test Set (6,840 samples per prompt)")
print()
print("=" * 80)
print("RESULTS BY PROMPT")
print("=" * 80)
print()

all_results = []
for prompt_id in range(1, 6):
    try:
        file_path = f'$RESULTS_DIR/KoBBQ_test_{prompt_id}.tsv'
        df = pd.read_csv(file_path, sep='\t')
        overall = df[df['category'] == 'overall'].iloc[0]
        all_results.append(overall)
        
        print(f"PROMPT {prompt_id}")
        print("-" * 80)
        print(f"  Out-of-Choice Ratio:       {overall['out-of-choice ratio']:.2%}")
        print(f"  Accuracy (Ambiguous):      {overall['accuracy in ambiguous contexts']:.2%}")
        print(f"  Accuracy (Disambiguated):  {overall['accuracy in disambiguated contexts']:.2%}")
        print(f"  Diff-Bias (Ambiguous):     {overall['diff-bias in ambiguous contexts']:.2%}")
        print(f"  Diff-Bias (Disambiguated): {overall['diff-bias in disambiguated contexts']:.2%}")
        print()
    except Exception as e:
        print(f"PROMPT {prompt_id}: Error reading results - {e}")
        print()

print("=" * 80)
print("AVERAGE ACROSS ALL PROMPTS")
print("=" * 80)
print()

if all_results:
    avg_df = pd.DataFrame(all_results)
    print(f"  Out-of-Choice Ratio:       {avg_df['out-of-choice ratio'].mean():.2%}")
    print(f"  Accuracy (Ambiguous):      {avg_df['accuracy in ambiguous contexts'].mean():.2%}")
    print(f"  Accuracy (Disambiguated):  {avg_df['accuracy in disambiguated contexts'].mean():.2%}")
    print(f"  Diff-Bias (Ambiguous):     {avg_df['diff-bias in ambiguous contexts'].mean():.2%}")
    print(f"  Diff-Bias (Disambiguated): {avg_df['diff-bias in disambiguated contexts'].mean():.2%}")

print()
print("=" * 80)
print("BIAS ANALYSIS BY CATEGORY (Average across prompts)")
print("=" * 80)
print()

if all_results:
    # Collect all category results
    all_categories = {}
    for prompt_id in range(1, 6):
        try:
            file_path = f'$RESULTS_DIR/KoBBQ_test_{prompt_id}.tsv'
            df = pd.read_csv(file_path, sep='\t')
            for _, row in df.iterrows():
                cat = row['category']
                if cat != 'overall' and cat not in ['NC', 'ST', 'TM']:
                    if cat not in all_categories:
                        all_categories[cat] = []
                    all_categories[cat].append(row)
        except:
            pass
    
    print(f"{'Category':<35} {'Acc(Amb)':<12} {'Acc(Dis)':<12} {'Bias(Amb)':<12} {'Bias(Dis)':<12}")
    print("-" * 80)
    
    for cat in sorted(all_categories.keys()):
        cat_df = pd.DataFrame(all_categories[cat])
        acc_amb = cat_df['accuracy in ambiguous contexts'].mean()
        acc_dis = cat_df['accuracy in disambiguated contexts'].mean()
        bias_amb = cat_df['diff-bias in ambiguous contexts'].mean()
        bias_dis = cat_df['diff-bias in disambiguated contexts'].mean()
        print(f"{cat:<35} {acc_amb:>10.1%}   {acc_dis:>10.1%}   {bias_amb:>10.1%}   {bias_dis:>10.1%}")

print()
print("=" * 80)

EOF

# Display summary
cat "$SUMMARY_FILE"

# ============================================================================
# Complete
# ============================================================================

print_header "Evaluation Complete!"

echo "Output files saved to: $OUTPUT_BASE/"
echo ""
echo "  📁 $RAW_DIR/"
echo "     └── prompt_*/predictions.tsv    (Raw model outputs)"
echo ""
echo "  📁 $PROCESSED_DIR/"
echo "     └── prompt_*/evaluation.tsv     (Post-processed predictions)"
echo ""
echo "  📁 $RESULTS_DIR/"
echo "     ├── KoBBQ_test_*.tsv            (Detailed results per prompt)"
echo "     └── summary.txt                 (Overall summary)"
echo ""
echo "To view the summary:"
echo "  cat $RESULTS_DIR/summary.txt"
echo ""


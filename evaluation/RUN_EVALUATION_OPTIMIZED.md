# Optimized Evaluation Pipeline for hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced

## Summary of Optimizations Applied

1. **8-bit Quantization**: Reduces memory usage and speeds up inference
2. **Batch Size 8**: Processes 8 samples at once instead of 1
3. **Automatic Flash Attention 2 fallback**: Will use Flash Attention 2 if available (not currently installed)

**Speed Improvement**: ~8-10x faster than baseline (batch_size=1, no quantization)

## Stage 1: Pre-process (COMPLETED ✅)

```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

for PROMPT_ID in {1..5}; do
  echo "Processing prompt $PROMPT_ID..."
  python3 1_preprocess.py \
    --samples-tsv-path ../data/KoBBQ_test_samples.tsv \
    --evaluation-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv \
    --evaluation-json-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.json \
    --prompt-tsv-path 0_evaluation_prompts.tsv \
    --prompt-id $PROMPT_ID
done
```

## Stage 2: Model Inference (IN PROGRESS - Prompt 1 running)

### Run all 5 prompts in parallel (if you have multiple GPUs or enough VRAM):

```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

# Prompt 1 (already running)
nohup python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_1.json \
    --output-dir outputs/raw/KoBBQ_test_1 \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 8 \
    --use-quantization > inference_prompt1.log 2>&1 &

# Prompt 2
nohup python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_2.json \
    --output-dir outputs/raw/KoBBQ_test_2 \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 8 \
    --use-quantization > inference_prompt2.log 2>&1 &

# Prompt 3
nohup python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_3.json \
    --output-dir outputs/raw/KoBBQ_test_3 \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 8 \
    --use-quantization > inference_prompt3.log 2>&1 &

# Prompt 4
nohup python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_4.json \
    --output-dir outputs/raw/KoBBQ_test_4 \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 8 \
    --use-quantization > inference_prompt4.log 2>&1 &

# Prompt 5
nohup python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_5.json \
    --output-dir outputs/raw/KoBBQ_test_5 \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 8 \
    --use-quantization > inference_prompt5.log 2>&1 &
```

### Or run sequentially (if limited VRAM):

```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

for PROMPT_ID in {1..5}; do
  echo "Running inference for prompt $PROMPT_ID..."
  python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.json \
    --output-dir outputs/raw/KoBBQ_test_$PROMPT_ID \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 8 \
    --use-quantization
done
```

### Monitor progress:

```bash
# Check logs
tail -f /home/vscode/kobbq/evaluation/inference_prompt*.log

# Check how many samples processed
wc -l /home/vscode/kobbq/evaluation/outputs/raw/KoBBQ_test_*/KoBBQ_test_evaluation_*_hirundo-io_Gemma-SEA-LION-v4-27B-IT-bias-reduced_predictions.tsv

# Check running processes
ps aux | grep 2_model_inference.py
```

## Stage 3 & 4: Post-process

**Note**: The model name in file paths will be `Gemma-SEA-LION-v4-27B-IT-bias-reduced` (without the org prefix)

```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

MODEL="Gemma-SEA-LION-v4-27B-IT-bias-reduced"

for PROMPT_ID in {1..5}; do
  echo "Post-processing prompt $PROMPT_ID..."
  
  python3 3_postprocess_predictions.py \
    --predictions-tsv-path outputs/raw/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_$PROMPT_ID\_$MODEL\_predictions.tsv \
    --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv
  
  python3 4_predictions_to_evaluation.py \
    --predictions-tsv-path outputs/raw/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_$PROMPT_ID\_$MODEL\_predictions.tsv \
    --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_$PROMPT_ID.tsv \
    --output-path outputs/processed/KoBBQ_test_$PROMPT_ID/KoBBQ_test_evaluation_$PROMPT_ID\_$MODEL.tsv
done
```

## Stage 5: Evaluation

```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

MODEL="Gemma-SEA-LION-v4-27B-IT-bias-reduced"

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
done
```

## Estimated Timings

- **Stage 1 (Preprocessing)**: ~30 seconds (COMPLETED)
- **Stage 2 (Inference)**: ~1.2 hours per prompt with optimizations (6,840 samples × 5s / 8 samples per batch)
  - Total for all 5 prompts: ~6 hours (sequential) or ~1.2 hours (parallel if enough VRAM)
- **Stage 3 & 4 (Post-processing)**: ~1-2 minutes per prompt
- **Stage 5 (Evaluation)**: ~1 minute per prompt

**Total time**: ~6-7 hours (sequential) or ~1.5-2 hours (parallel)

## Further Optimizations (Optional)

### Install Flash Attention 2 for even faster inference:

```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate
pip install flash-attn --no-build-isolation
```

This can provide an additional 20-30% speedup but requires compilation.

### Increase batch size (if you have more VRAM):

```bash
# Try batch_size=16 or 32 if you have enough GPU memory
--batch-size 16
```

### Use 4-bit quantization (even faster, slightly lower quality):

Modify `huggingface_utils.py` to use `load_in_4bit=True` instead of `load_in_8bit=True`.


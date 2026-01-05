# Quick Start: KoBBQ Evaluation on A100 80GB
## Model: hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced

### ✅ Current Status

**Stage 1: Preprocessing** - ✅ COMPLETED (all 5 prompts)

**Stage 2: Model Inference** - 🔄 IN PROGRESS (all 5 prompts running in parallel)

### Optimizations Applied

✅ **Flash Attention 2** - Installed and enabled (20-30% faster)  
✅ **Batch Size 32** - Processing 32 samples per batch  
✅ **Full Precision (bfloat16)** - No quantization (faster on A100)  
✅ **Parallel Execution** - All 5 prompts running simultaneously  
✅ **100% GPU Utilization** - Maxing out the A100  

**Speed:** ~6 seconds per batch (32 samples) = ~5 samples/second  
**Time per prompt:** ~20-25 minutes  
**Total time:** ~20-25 minutes (parallel execution)

---

## Monitor Progress

```bash
# Watch progress in real-time
watch -n 10 /home/vscode/kobbq/evaluation/monitor_progress.sh

# Or run once
/home/vscode/kobbq/evaluation/monitor_progress.sh
```

---

## After Inference Completes

### Stage 3 & 4: Post-processing

```bash
cd /home/vscode/kobbq/evaluation
./run_postprocessing.sh
```

This will:
- Convert raw predictions to A/B/C format
- Handle out-of-choice responses
- Create evaluation-ready files

**Time:** ~1-2 minutes

---

### Stage 5: Final Evaluation

```bash
cd /home/vscode/kobbq/evaluation
./run_evaluation.sh
```

This will calculate:
- ✓ Accuracy in ambiguous contexts
- ✓ Accuracy in disambiguated contexts  
- ✓ Diff-bias in ambiguous contexts
- ✓ Diff-bias in disambiguated contexts
- ✓ Out-of-choice ratio

**Time:** ~1 minute

---

## Results Location

After evaluation completes, results will be in:

```
evaluation_result/
├── KoBBQ_test_1.tsv
├── KoBBQ_test_2.tsv
├── KoBBQ_test_3.tsv
├── KoBBQ_test_4.tsv
└── KoBBQ_test_5.tsv
```

Each file contains:
- Overall scores
- Scores by category (e.g., age, gender, race, religion, etc.)
- Scores by template label

---

## Complete Pipeline Summary

| Stage | Description | Status | Time |
|-------|-------------|--------|------|
| 1 | Preprocessing | ✅ Done | ~30s |
| 2 | Model Inference | 🔄 Running | ~25min |
| 3-4 | Post-processing | ⏳ Pending | ~2min |
| 5 | Evaluation | ⏳ Pending | ~1min |
| **Total** | | | **~28min** |

---

## Manual Commands (if needed)

### Check running processes
```bash
ps aux | grep 2_model_inference.py
```

### Check GPU status
```bash
nvidia-smi
```

### View logs
```bash
# All logs
tail -f inference_prompt*_optimized.log

# Specific prompt
tail -f inference_prompt1_optimized.log
```

### Kill all inference processes (if needed)
```bash
pkill -f 2_model_inference.py
```

### Re-run specific prompt
```bash
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

python3 2_model_inference.py \
    --data-path data/KoBBQ_test/KoBBQ_test_evaluation_1.json \
    --output-dir outputs/raw/KoBBQ_test_1 \
    --model-name hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced \
    --batch-size 32
```

---

## Technical Details

### Model Configuration
- **Model:** hirundo-io/Gemma-SEA-LION-v4-27B-IT-bias-reduced
- **Architecture:** Gemma (27B parameters)
- **Precision:** bfloat16
- **Attention:** Flash Attention 2
- **Device:** NVIDIA A100 80GB

### Inference Settings
- **Batch Size:** 32
- **Max New Tokens:** 30
- **Temperature:** 0.0 (greedy decoding)
- **Chat Template:** Auto-detected and applied

### Dataset
- **Name:** KoBBQ Test Set
- **Samples:** 2,280 base samples
- **Total contexts:** 6,840 (3 contexts per sample)
- **Prompts:** 5 different prompt formats

### Performance Metrics
- **Throughput:** ~5 samples/second per prompt
- **GPU Memory:** 77-79 GB / 80 GB
- **GPU Utilization:** 100%
- **Batch Processing Time:** ~6 seconds per batch (32 samples)

---

## Troubleshooting

### If inference is too slow
- Current batch_size=32 is optimal for this model on A100
- With 80GB, batch_size=64 might work but could cause OOM

### If out of memory
```bash
# Reduce batch size to 16
--batch-size 16
```

### If results are incomplete
```bash
# Check which prompts finished
ls -lh outputs/raw/KoBBQ_test_*/

# Count completed samples (should be 6841 with header)
wc -l outputs/raw/KoBBQ_test_*/KoBBQ_test_evaluation_*_predictions.tsv
```

### If post-processing fails
```bash
# Run for specific prompt
cd /home/vscode/kobbq/evaluation
source /home/vscode/kobbq/.venv/bin/activate

python3 3_postprocess_predictions.py \
  --predictions-tsv-path outputs/raw/KoBBQ_test_1/KoBBQ_test_evaluation_1_Gemma-SEA-LION-v4-27B-IT-bias-reduced_predictions.tsv \
  --preprocessed-tsv-path data/KoBBQ_test/KoBBQ_test_evaluation_1.tsv
```


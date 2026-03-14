# Whipstr STT (ASR) — Training Issues and Fixes

## Problem
Model trained for 200 epochs but validation accuracy stuck at ~11% (barely better than random guessing at 10%).

## Root Causes Identified

### 1. CRITICAL: Sigmoid Bottleneck in Encoder ⚠️
The encoder was using `sigmoid` activation on its output, restricting values to [0, 1]. This severely limited the model's expressiveness:
- The transformer receives these restricted values and linearly projects them
- The model cannot express negative values or values > 1
- This creates a severe information bottleneck

Fix Applied:
- Removed `sigmoid` activation from encoder output
- Now outputs raw logits that can be any real value
- Added dropout for regularization instead

### 2. No Learning Rate Scheduling
Training for 200 epochs with fixed LR = 0.001 causes the model to get stuck:
- Initial learning rate may be too high after early epochs
- Model can't fine-tune once it reaches a plateau

Fix Applied:
- Added `ReduceLROnPlateau` scheduler
- Reduces LR by 0.5x when validation loss plateaus for 10 epochs
- Allows model to escape local minima and fine-tune

### 3. Insufficient Training Data
10,000 training samples may be too small for this architecture:
- Encoder: ~500k parameters
- Transformer: ~8M parameters
- Total: ~8.5M parameters with only 10k samples

Fix Applied:
- Increased training samples to 50,000
- Increased validation samples to 5,000
- Added weight decay (0.01) for better regularization

### 4. Suboptimal Hyperparameters
- Batch size of 32 is small for this dataset size
- Learning rate of 0.001 may be too high initially

Fix Applied:
- Increased batch size to 64
- Reduced initial LR to 0.0003
- Switched from Adam to AdamW (better regularization)

## Files Modified

### `whipstr/whipstr_encoder.py`
```python
# BEFORE:
x = self.sigmoid(x)  # Restricts to [0, 1]

# AFTER:
x = self.dropout(x)
x = self.fc2(x)  # Raw logits, no restriction
```

### `whipstr/whipstr_transformer.py`
- Removed validation check for [0, 1] range (no longer needed)

### `whipstr/whipstr_train.py`
- Added learning rate scheduler
- Added LR logging in training loop

## New Files Created

### `train_improved.py`
Complete Whipstr STT training script with all improvements:
- 50k training samples
- Batch size 64
- AdamW optimizer with weight decay
- Learning rate scheduler
- Early stopping (patience=30)
- Best model checkpointing

### `diagnose_training.py`
Diagnostic script to check:
- Encoder output statistics
- Gradient flow
- Initial predictions
- Loss comparison to random baseline

## How to Use

### 1. Run Diagnostics (Optional)
```bash
python diagnose_training.py
```
This will show you:
- Encoder output range and statistics
- Gradient norms
- Initial predictions before training
- Warnings about potential issues

### 2. Train with Improved Settings
```bash
python train_improved.py
```

Expected behavior:
- Training loss should decrease steadily
- Validation accuracy should improve beyond 11% within 20-30 epochs
- Learning rate will automatically reduce when loss plateaus
- Best model saved to `checkpoints/best_model.pt`

### 3. Monitor Training
Watch for:
- Validation accuracy > 20% by epoch 10 (good sign)
- Validation accuracy > 50% by epoch 50 (on track)
- Learning rate reductions (scheduler working)
- Early stopping if no improvement for 30 epochs

## Expected Results

With these fixes, you should see:
- Epoch 10-20: Accuracy around 30-40%
- Epoch 30-50: Accuracy around 60-70%
- Epoch 50-100: Accuracy around 80-90%

If accuracy is still stuck:
1. Check `diagnose_training.py` output for warnings
2. Increase training data to 100k samples
3. Reduce model size (fewer transformer layers)
4. Check if encoder is learning (gradient norms should be > 1e-4)

## Technical Explanation

The sigmoid bottleneck was the primary issue. Consider:
- Sigmoid output: values in [0, 1]
- Linear projection in transformer: `W @ x + b` where x ∈ [0, 1]
- This means the transformer can only access a restricted subspace
- Removing sigmoid allows x ∈ ℝ, giving full expressiveness

Think of it like trying to paint with only pastel colors (0-1 range) vs having access to the full color spectrum (all real numbers). The model needs that full range to learn complex speech patterns in the audio features.

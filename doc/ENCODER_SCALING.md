WHIPSTR ENCODER SCALING: DESIGN DECISION
=========================================

WhipstrEncoder's convolutional backbone (conv1–conv4, pool1–pool3, fc1)
does NOT scale with model variant — only the final fc2 layer changes width.

LAYOUT
------

conv1 (2→32) → conv2 (32→64) → pool1  →  conv3 (64→128) → pool2  →  conv4 (128→256) → pool3  →  fc1 (→256) → fc2 (→output_values)

All layers except fc2 are hardcoded in whipstr/whipstr_encoder.py:46–70.

PARAMETER COUNTS (window_size=11)
--------------------------------

| Layer group          | Parameters  | Scales with variant? |
|----------------------|-------------|----------------------|
| conv1–conv4, pools   |      388,128 | No                   |
| fc1                  |    6,816,000 | No                   |
| fc2                  |    8,224–49,344 | Yes (the only layer) |

| Variant          | output_values | Encoder params | Δ from small |
|------------------|---------------|----------------|--------------|
| whipstr-small    |            32 |     7,212,352 |           —  |
| whipstr-base     |            64 |     7,220,576 |      +8,224  |
| whipstr-medium   |           128 |     7,237,024 |     +24,672  |
| whipstr-large    |           192 |     7,253,472 |     +41,120  |

The encoder contributes ≈7.2M params across all variants — essentially
constant. Only the transformer scales (d_model, nhead, layers, dim_feedforward).

RATIONALE
---------

fc2 is the bottleneck of the encoder. Scaling it with `output_values` controls
how much information passes from the conv stack into the transformer. This
design is based on experimental evidence from two training runs:

1. English training run: Most loss was due to poor language modelling /
   repetition in the autoregressive decoder, not encoder capacity. This
   disproved the theory that "encoder is bad, transformer is good."

2. IPA (base) training run: Tonal features were modelled successfully despite
   the fixed conv front-end — they just took longer to evolve. This disproved
   the theory that a fixed encoder would lose time-dependent features like tone.

Conclusion: The convolutional front-end has sufficient capacity for acoustic
feature extraction across all current variants. The bottleneck is sufficient
to gate information into the transformer without being a performance
limitation.

IMPLICATION FOR CAPACITY ABLATIONS
----------------------------------

Comparing whipstr-small through whipstr-large primarily measures transformer
scaling, not encoder scaling. The encoder's acoustic front-end capacity is
held constant. Any ablation that varies model size is effectively testing only
the transformer's ability to process fixed-capacity encoder features.

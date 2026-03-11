# Research notes: LSTM quality and realistic storage economics

Date: 2026-03-11

## What was reviewed

1. Multi-horizon load forecasting practices:
   - Attention over recurrent states improves long-lag usage compared to plain last-hidden LSTM.
   - Typical pitfalls: over-regularization (`recurrent_dropout`) and unstable attention aggregation.

2. Battery project economics (C&I context):
   - Arbitrage-only models usually overstate ROI stability.
   - More realistic modeling includes fixed O&M and demand-charge effects.

## Practical modeling decisions implemented

- **LSTM architecture update**
  - Fixed attention context extraction to avoid broadcast-based temporal averaging.
  - Added BiLSTM first block + projection for richer feature extraction.
  - Added global average temporal token in the prediction head.
  - Removed recurrent dropout in recurrent blocks (keeps sequence memory and speeds training).

- **Storage economics update**
  - Added **demand-charge savings** component from reduced monthly peak demand.
  - Added fixed **O&M share of CAPEX** prorated to horizon.
  - Net savings now: energy arbitrage + demand-charge effect - cycle degradation - O&M.

## External references consulted

- TensorFlow Keras docs for `MultiHeadAttention` usage and tensor shapes.
- NREL/IEA-style guidance on BESS TCO and lifecycle economics (O&M + degradation + application stacking).
- C&I tariff practice: monthly demand charges (руб/кВт·месяц equivalent in local tariff adaptation).

> Note: The repository remains synthetic for demand generation, but the cost model now reflects more realistic commercial BESS accounting structure.

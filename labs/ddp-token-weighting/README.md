# DDP token-weighted loss — CPU math validation

**Banner:** This is a **CPU NumPy** check that a token-weighted mean loss reduces
correctly when each rank sees a different number of supervised tokens. It is
**not** multi-GPU / NCCL training proof.

## Claim under test

Naive `loss.mean()` over a padded batch under-weights ranks with more real tokens.
The correct reduction is:

\[
L = \frac{\sum_r \sum_t m_{r,t}\, \ell_{r,t}}{\sum_r \sum_t m_{r,t}}
\]

where \(m\) is the non-pad / supervised mask. In DDP, each rank sends
`(sum_loss, sum_tokens)` and the global ratio is formed after `all_reduce`.

## Run

```bash
python labs/ddp-token-weighting/simulate.py
# or
pytest -q labs/ddp-token-weighting/test_simulate.py
```

Results are printed and asserted; no invented GPU timings.

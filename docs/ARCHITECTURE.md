# Architecture / correctness contract

## Interpretation of RoPE + [SummerDeltaAttention + NoPE]

This library interprets the expression as **interleaved residual blocks**:
`RoPE causal attention → SDA → NoPE causal attention`, repeated to `num_layers`.
It is not a parallel three-branch weighted sum. All blocks share the hidden width.
SDA and NoPE have no explicit positional embedding; causal order still exists.
The LM predicts the next token: `input=ids[:-1]`, `labels=ids[1:]`.
Every independently sampled training window resets the delta state and archive.
No token from a different sample enters that state.

## Delta state and SDA-5-style residual correction

Normalized q,k are column vectors. F has shape (d_key,d_value).

```
F_t = F_(t-1) + beta_t k_t (v_t - F_(t-1)^T k_t)^T
beta_t = sigmoid(W_beta x_t)
y_fast = F_t^T q_t
j = argmax over causal archived keys of q_t^T k_j
y_base = y_fast + 1[sim_j >= threshold] (v_j - F_t^T k_j)
```

The read occurs after writing token t. This is causal because the label is token t+1.
Fast-state accumulation is FP32 even under FP16/BF16 autocast. No state/archive
`detach` is performed within a sample; backward includes all previous writes.
The `base` gate and argmax are piecewise differentiable, not differentiable at
threshold/selection boundaries. The separate projections and beta remain trainable.
Finite-difference checks use inputs away from those boundaries.

## SDA+-style read

```
w = softmax(softmax_scale * top_m_similarities)
y_archive = sum_j w_j v_j
g = sigmoid((top_similarity - threshold) / temperature)
y_plus = g * y_archive + (1-g) * y_fast
```

`archive_fake_quant=true` uses per-vector symmetric INT8 rounding with a
straight-through estimator. It simulates noise; it stores floating-point tensors
for autograd and does not claim an archive-memory reduction.
The network codec is different: it really serializes int8 codes plus FP32 scales.

### Differences from upstream

Reference: `Summer110622/SummerDeltaAttention/summer_delta_attention/plus.py`,
blob `b1cd7c9d9d2fdcbb99ceb30d5d21393e9332e022`, inspected during implementation.

* Reservoir sampling is replaced by a causal FIFO of the most recent C entries.
  Equal-similarity base reads prefer the newest entry. Entries are not de-duplicated;
  top-m plus can mix repeated keys, unlike upstream's exact overwrite behavior.
* Archive search is a full scan of that bounded window, not spherical k-means.
* q/k/v projections and token-dependent beta are trained with PyTorch autograd.
* No WAN archive queries or cross-sample shared fast state are used.
* No claim of numerical equivalence to every mode of upstream SDA/SDA+ is made.

The formulas are checked against independent NumPy calculations for the same
FIFO policy. Hard/soft causal masking is tested by perturbing future tokens.
The project retains MIT attribution to summer; the Torch implementation is a new
training-oriented implementation, not the upstream NumPy package wrapped as a model.

## WAN protocol

1. A worker sends its ID, tokenizer fingerprint, microbatch and accumulation sizes.
2. The coordinator returns a lease and exact FP32 model snapshot in one safetensors response.
3. Local AdamW runs K steps. On each new lease its optimizer is reset.
4. Worker uploads delta relative to that exact snapshot (blockwise INT8 or exact FP32).
5. Coordinator clips the delta's global L2 norm and mixes a reconstructed **local model**
   into the **current model**, with weight `server_alpha/(1+version_lag)`.
6. New model snapshot and response receipt commit atomically in SQLite WAL/FULL mode.

This is not `current += stale_delta` and not token-weighted synchronous FedAvg.
All parameter names, shapes, dtypes, finiteness, token counts and lease metadata
are validated. Duplicate bytes for the same accepted lease return the original
receipt. Different bytes for an accepted lease are rejected. Too-old/expired leases
are rejected and their local work is discarded. One process per worker work_dir.

Exact fresh updates with `server_alpha=1` and `wire=fp32` recover the worker's
local parameters, up to floating-point arithmetic. With multiple asynchronous
workers or quantization, the trajectory is deliberately different from exact DDP.
No convergence theorem is claimed for this model/optimizer/scheduler combination.

## Heterogeneity heuristic

Let b_i be tokens per optimizer step, r_i the EWMA of measured training token/s,
and c_i the previous download+upload wall time. After the first job:

```
compute_budget = min(max(target_seconds, 4*c_i), lease_seconds/2)
K_i = clip(round(compute_budget*r_i/b_i), min_local_steps, max_local_steps)
```

Different microbatches and gradient accumulation are allowed; sequence length,
architecture and tokenizer must match. The first K is a user-configured conservative
value. No network-throughput oracle or synthetic prediction is presented as a measurement.
This scheduler can increase staleness; inspect rejections and validation loss.
Fast-client/data-distribution bias remains and must be evaluated for a real corpus.

## Costs

Delta updates cost O(T H d_k d_v), and the causal archive costs O(T H C d_k)
plus correction work. The Python recurrence is not a performance kernel.
Dense RoPE/NoPE blocks retain quadratic attention work (SDPA may optimize memory).
Autograd needs activation memory beyond the final fixed-sized recurrent state.
Full FP32 parameter/gradient/AdamW storage is roughly 16P bytes before activations
and allocator overhead. Several additional CPU copies and saved server versions
are used. WAN payload for one completed job is about 5P bytes with int8 upload,
versus 8P with fp32 upload. Headers, scales, TLS and retries add overhead.

## References

* Original SDA/SDA+: https://github.com/Summer110622/SummerDeltaAttention
* PyTorch causal SDPA: https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
* Asynchronous Federated Optimization (FedAsync): https://arxiv.org/abs/1903.03934
* DiLoCo (related motivation, **not this algorithm**): https://arxiv.org/abs/2311.08105

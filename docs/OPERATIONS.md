# Operations / security

## Supported shape

A small, trusted group of Linux/CUDA workers, each holding the full model, communicates
with one CPU coordinator. Run exactly one uvicorn worker process. Back up the state
volume. Use SQLite's backup API or stop the service before copying its files; copying
only the main .sqlite3 file while WAL is live is not a valid backup.

No deployment, paid service creation, DNS modification, port forwarding or external
credential provisioning is performed by the library. The operator supplies these.

## Authentication and encryption

Create a random 32-byte or longer secret (example: `secrets.token_urlsafe(32)`).
Use the same WSDA_TOKEN for this trusted run. The placeholder in .env.example is
rejected. Do not log secrets or embed them in URLs/notebooks/commits. Rotate the
secret and restart coordinator/workers if it is disclosed.

Public access requires HTTPS with a valid certificate; the client verifies TLS and
will not follow redirects or use environment-configured HTTP proxies. The default
Docker API port binds only to loopback. The optional Caddy overlay exposes only the
HTTPS ingress and HTTP certificate challenge. Public host firewall/rate limits are
still the operator's responsibility. `--allow-http` does not encrypt anything; use
it only on a trusted Docker bridge/private transport. There is no browser CORS API.

Only safetensors + JSON are accepted, never pickle. Update bodies are capped before
decoding; model schema, numeric values, metadata and global delta norm are checked.
This does not make the service robust against hostile authorized clients. A shared
secret gives read access to model snapshots and write access to training updates.
There is no Byzantine aggregation, differential privacy or secure aggregation.
Model deltas can reveal training data; local data retention is not a privacy guarantee.

## Colab

Use a paid managed runtime with positive compute units, or an authorized self-managed
runtime. Read the current Colab FAQ before running. Keep execution in the notebook;
no SSH daemon, remote-control tunnel, background keepalive or account-limit bypass is
provided. The client makes ordinary HTTPS API requests to your training coordinator.
Free-tier distributed workers are not supported by the supplied notebook.

The notebook asks for explicit runtime-policy acknowledgment. This is not automatic
verification of a subscription. The operator remains responsible for eligibility.
Do not upload proprietary corpora to Colab without authorization.

## Data and objective

Byte tokenization is a smoke-test default, not an efficient Japanese/English LLM
vocabulary. Supply a local tokenizer.json for a realistic experiment. The tokenizer
must be identical on all nodes; a SHA256 mismatch fails closed. No remote custom
Python tokenizer code is executed.

If both nodes have the same prepared corpus: use num_shards=2 with indices 0 and 1.
Targets are partitioned into disjoint sequence blocks. A boundary token can be the
input context of the next block; target positions are not duplicated across shards.
Sampling with replacement within a shard is intentional. Train and validation token
ranges are split before block creation. Repeated source documents can still cause
semantic leakage; deduplication is the operator's job.

If nodes use different corpora: use num_shards=1 per local corpus, keep tokenizer
identical, and evaluate non-IID drift. More/faster local updates mean more influence;
this implementation does not promise a uniform-per-example global objective.

## Recovery semantics

A received, accepted update is atomically persisted with its receipt. Resending the
identical update after ACK loss does not double-apply it, even after restart.
Unsent work can be lost on a crash. Worker outbox files are fsynced before transmission
and replaced atomically on local filesystems; preserve the worker work_dir to replay.
In-progress local optimizer/RNG state is not checkpointed. If a new lease supersedes
an old one for the same worker ID, the old lease is rejected. Use unique worker IDs.

A coordinator restart with a different run config is refused. Resume without --config
or with the original config. Change architecture/seq_len/tokenizer by starting a new
state directory. Do not edit live SQLite data. Old snapshots beyond max_staleness
are pruned, but receipts remain for duplicate detection. Disk housekeeping/receipt
retention for very long runs is not automated.

## Scaling and diagnostics

Start with tiny.json and the default short sequence. Check local loss before using
WAN. For low memory reduce microbatch or increase accumulation. If batch=1 still
OOMs, shrink architecture/sequence length globally; a slower worker cannot train a
smaller incompatible model. With many rejections, increase target compute balance,
reduce local step count or investigate max_staleness; blindly raising it can damage
quality. Stop training when validation loss diverges.

max_payload_mb defaults to 256 on both ends and Caddy. Raising only one of these is
insufficient. The coordinator holds full FP32 snapshots and several temporary tensor
copies. Its CPU memory/disk can be the bottleneck well before GPU VRAM.

The reference generator recomputes the prefix and truncates to max_seq_len; no fast
incremental KV-cache generation is implemented. Exported weights are custom W-SDA
weights, not a Hugging Face pretrained checkpoint.

Actual binding constraint: NOT FOUND. Batch 16 in C is compatible with memory constraints but is not proof of an OOM/memory limit. A1 and B1 logs show slow throughput; CPU bispectrum/PCA and codec transcoding are present, but no profiling identifies the bottleneck.

Full-pool scaling: 2.583913 times the utterance count [computed from megaprompt full pool and verified manifest]. B1 rounded log throughput 0.4 batches/sec gives 2737 / 0.4 = 6842.5 seconds (1.9007 hours) per training pass. Linear extrapolation gives about 17680.4 seconds (4.9112 hours) for a full-pool training pass, excluding validation, I/O, setup, checkpoint saving and platform contention. This is an estimate from rounded throughput, not measured epoch wall time or a completed full-pool run. No measured full-pool baseline found.

RTF: NOT FOUND; thesis explicitly states no inference latency collected (Section 5.3). Do not report a real-time claim.

Future resources: Rafa recommendation pending. A defensible non-numeric recommendation is a persistent CUDA-capable GPU environment, enough CPU/RAM for bispectrum/PCA and concurrent FFmpeg workers, SSD storage for corpus/codecs/checkpoints, resumable jobs, and durable epoch logs. Numeric minimums, GPU-hours or storage GB cannot be justified as measured requirements by this evidence.

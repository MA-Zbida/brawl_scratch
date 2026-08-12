# Performance

## Measured baseline

Measured on an RTX 3050 Ti with `yolo26s-p2` at `imgsz=960`:

```
total=36.36ms (27.51 hz)  frame=2.32ms  detect=33.72ms  memory=0.19ms  everything-else=0.12ms
```

**Detection is 93% of the step.** All the Python combined -- state update, reward,
canonicalisation, input -- is 0.31 ms, under 1%. Optimising it would be measuring
noise; detector latency is the only lever that matters, and the observed control
rate is **27.5 Hz**, not the 40 Hz these documents previously assumed.

## Where the time goes

| stage | ms | share |
|---|---:|---:|
| detect (YOLO26-s + P2 @ 960, TensorRT) | 33.72 | 92.7% |
| frame grab (DXGI duplication) | 2.32 | 6.4% |
| memory (detections -> state) | 0.19 | 0.5% |
| game logic | 0.06 | 0.2% |
| apply action (key injection) | 0.04 | 0.1% |
| reward | 0.01 | <0.1% |
| **total** | **36.36** | **27.5 Hz** |

### After optimisation: 32.62 ms, **30.7 Hz**

| change | detect | effect |
|---|---:|---:|
| baseline | 33.72 | 27.5 Hz |
| cap game to 60 fps | 33.07 | frame grab -36%, detect -2% |
| remove redundant pre-resize | 33.02 | no change; ultralytics does the same work |
| **batch GPU->CPU transfers** | **30.73** | **30.7 Hz** |

The transfer fix was the only code change that moved the needle. `_results_to_detections`
read `box.cls`, `box.xywhn` and `box.conf` inside a per-box loop -- three device-to-host
synchronisations per detection, 24 for eight objects, each stalling until the GPU drained.
It now pulls each tensor across once for the whole result: **3 transfers regardless of
count**.

Every latency benchmark missed this because they call the model and discard the results;
the transfers only occur when something reads the tensors. Only the live loop did.

**Isolated inference is still ~17 ms against 30.7 ms live.** That gap is not explained.
Ruled out by measurement: FP32 (already FP16), GPU contention from the game, the
pre-resize, and the capture buffer's memory layout. 30 Hz was judged good enough to
proceed; the remainder is worth less than finding out whether the pipeline learns.

## What this rules out

**Algorithmic optimisation of the Python.** Every loop in the state pipeline runs over
at most `yolo_max_det = 8` detections, buffers are preallocated, and the whole of it costs
0.31 ms. There is no complexity win available; a 2x speedup of all Python moves 27.5 Hz to
roughly 27.9 Hz.

## What actually helps, in order

**1. Cap the game's framerate. This is the big one, and it is free.**

The detector shares the NVIDIA GPU with Brawlhalla. The display runs on the Intel iGPU,
but the game renders on the discrete card -- the same one running TensorRT.

| condition | detect |
|---|---:|
| benchmark, game closed | **18.2 ms** |
| live profile, game running | **33.7 ms** |

That 1.9x gap is pure GPU contention. The game was running **uncapped at 120-200 fps**;
Brawlhalla ticks at 60, so every frame above that is taken directly from the detector.
Cap it to 60 and lower graphics quality before touching the model.

*(A rebuilt FP16 engine was ruled out as the cause: the plan was already FP16 --
"Converted 455/476 nodes to fp16" -- and rebuilding moved latency 17.6 -> 18.2 ms, i.e.
nothing. The 215 MB is duplicated shared constants from the ONNX conversion, not FP32
weights. Plan size does not indicate precision; read the build log instead.)*

**2. Smaller model before smaller input.** `yolo26n-p2` instead of `-s`. Resolution is
what makes the self-indicator detectable -- it is ~15 px at `imgsz=960` and the P2 head
exists for it. Lowering `imgsz` to save time can destroy the feature that agent identity,
and therefore every relational feature, depends on.

**3. Async perception, with a caveat.** `perception/async_pipeline.py` decouples the
control loop from the detector. At the measured split (33.7 ms detect, 2.6 ms everything
else) that would let the policy step at roughly 380 Hz while perception refreshes at 29 Hz
-- about 13 policy steps per new observation. That is not 13x more information; most steps
would see `is_stale=True` on unchanged state.

It is still worth doing, for a different reason than throughput: finer input timing within
one perceptual frame. But it does **not** substitute for making detection faster, and
should not be reported as a 13x speedup.

**4. PPO update stalls.** `n_steps=2048` at 27.5 Hz is a 74-second rollout followed by a
multi-second optimiser pass **while the game keeps running**. Those frames are recorded as
deliberate actions by an agent that was not acting. Corrupted data, not just lost time.

## The honest ceiling

Detection cannot go below a few milliseconds on this hardware, so the realistic best case
is roughly 60-80 Hz -- perhaps 2-3x the current rate, which is 5-7 M steps per day rather
than 2.4 M. Useful, not transformative.

A learned dynamics model trained on recorded trajectories runs at thousands of steps per
second and turns a wall-clock-bound problem into a compute-bound one. That is the only
change with an order-of-magnitude effect on training time; everything above is a constant
factor.

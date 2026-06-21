# Fraud Anti-Fraud Model Pipeline

This document describes the fraud-dataset pipeline runner:

- Runner: `src/test/scala/benchmarks/RunFraudAntiFraudPipeline.scala`
- Dataset: `src/main/resources/fraud_data.csv`

## What It Does

- Loads `fraud_data.csv` (V1..V28 + Amount + Class)
- Performs anti-fraud oriented preprocessing and feature engineering:
  - Class-balancing by controlled negative sampling
  - Stratified train/valid/test split
  - Min-max binning for sparse feature models
  - Derived auxiliary task label (`high_amount`)
  - Pseudo sequence/user/item construction for sequence/matching models
- Trains and evaluates a multi-model suite:
  - Ranking: `xDeepFM`, `MEMBA`, `LiquidNetWork`
  - Generative/sequence: `LLM4Rec`, `HLLM`, `HSTU`, `RQVAE`
  - Matching: `MAMBA`, `MIND`
  - Multi-task: `MMOE`, `OMoE`, `PLE`, `AITM`, `MetaHeac`
- Prints PASS/FAIL per model with metrics and failure reason.

## Quick Run

```bash
cd /home/muller/IdeaProjects/torch-rechub-scala
sbt --no-colors "Test/runMain benchmarks.RunFraudAntiFraudPipeline --dataset /home/muller/IdeaProjects/torch-rechub-scala/src/main/resources/fraud_data.csv --max_rows 8000 --epochs 1 --batch_size 512 --device cpu"
```

## Tunable Arguments

- `--dataset`: fraud csv path
- `--max_rows`: max processed rows after balancing
- `--epochs`: training epochs per model
- `--batch_size`: mini-batch size
- `--device`: preferred device (`cpu`/`cuda`)
- `--seed`: random seed
- `--bins`: number of bins for sparse discretization
- `--seq_len`: synthetic sequence length

## Current Known Model Issues

The runner surfaces model-level problems directly in summary logs. At the time of implementation, the following known issues can still fail in this codebase:

- `RQVAE`: ModuleList child casting issue inside residual quantizer path
- `MAMBA`: shape assumptions in matching trainer/evaluation path
- `MIND`: sequence embedding table lookup mismatch

Other models in the list run end-to-end on the fraud pipeline and return metrics.


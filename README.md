# How LoRA Remembers? A Parametric Memory Law for LLM Finetuning

This repository supports the paper "How LoRA Remembers? A Parametric Memory Law for LLM Finetuning". It studies capacity laws and training methods for exact parametric memory with LoRA. The code includes two benchmarks, Long-Context Memorization Stress Test and PhoneBook, and compares three methods: SFT, MemFT-OT, and MemFT-SW.

## Benchmarks

- PhoneBook: a multi-sample short key-value exact memorization task.
- Long-Context: a single long-sequence memorization task with random ratios `r0`, `r20`, `r40`, `r60`, `r80`, and `r100`.

Entrypoints:

- PhoneBook: `script/pb/`
- Long-Context: `script/long_context/`

Both benchmarks support `qwen3-8b` and `llama3.1-8b`. The default layers are Qwen layer 24 and Llama layer 18.

## Methods

- `sft`: standard LoRA SFT baseline.
- `memft_ot`: only-threshold MemFT.
- `memft_sw`: sliding-window / curriculum MemFT.

Note: PhoneBook and Long-Context use different `memft_sw` mechanisms. PhoneBook uses Inter-Batch Temporal Curriculum with length-dependent hyperparameters. Long-Context uses Intra-sample Spatial Sliding.

## Quick Start

PhoneBook training:

```bash
METHOD=sft bash script/pb/run_pb_train_qwen.sh
METHOD=memft_ot bash script/pb/run_pb_train_qwen.sh
METHOD=memft_sw bash script/pb/run_pb_train_qwen.sh
```

PhoneBook apply and final loss:

```bash
METHOD=sft bash script/pb/run_pb_apply_qwen.sh
METHOD=sft bash script/pb/run_pb_final_loss_qwen.sh
```

Long-Context training:

```bash
METHOD=sft bash script/long_context/run_long_context_train_qwen.sh
METHOD=memft_ot bash script/long_context/run_long_context_train_qwen.sh
METHOD=memft_sw bash script/long_context/run_long_context_train_qwen.sh
```

Long-Context apply and final loss:

```bash
METHOD=sft bash script/long_context/run_long_context_apply_qwen.sh
METHOD=sft bash script/long_context/run_long_context_final_loss_qwen.sh
```

Use the Llama wrappers by replacing `_qwen.sh` with `_llama.sh`.

Common overrides:

```bash
METHOD=memft_sw LENGTHS="1000 2000" RANKS="4 8" GPU_ID=0 bash script/pb/run_pb_train_qwen.sh
METHOD=sft RATIOS="r0 r20 r40" LENGTHS="50 100 200" RANKS="1 2 4" bash script/long_context/run_long_context_train_qwen.sh
```

## Outputs

PhoneBook:

```text
vectors/pb/<method>/<model>/random/length_<L>/layer_<layer>/rank_<R>/
generation/pb/<method>/<model>/random/length_<L>/layer_<layer>/rank_<R>/
generation/pb/<method>/<model>/random/length_<L>/layer_<layer>/rank_<R>/final_loss/
```

Long-Context:

```text
vectors/long_context/<method>/<model>/<ratio>/length_<L>/layer_<layer>/rank_<R>/
generation/long_context/<method>/<model>/<ratio>/length_<L>/layer_<layer>/rank_<R>/
generation/long_context/<method>/<model>/<ratio>/length_<L>/layer_<layer>/rank_<R>/final_loss/
```

See `docs/phonebook_benchmark.md` and `docs/long_context_benchmark.md` for full sweep details.

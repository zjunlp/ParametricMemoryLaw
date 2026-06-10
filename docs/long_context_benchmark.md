# Long-Context Memorization Stress Test

This benchmark measures whether a model can memorize and reproduce a long injected memory under increasing random-token replacement. It is the long-context/random benchmark previously run from `EasyEdit_qwen_random` and `EasyEdit_llama_random`, now exposed through the same script style as the PhoneBook benchmark.

## Ratio Naming

The paper-facing ratio names are:

| New name | Meaning | Old dataset name |
|---|---|---|
| `r0` | 0% random replacement; pure LongBench sample | `longbench0random0` |
| `r20` | 20% random replacement | `longbench0random20` |
| `r40` | 40% random replacement | `longbench0random40` |
| `r60` | 60% random replacement | `longbench0random60` |
| `r80` | 80% random replacement | `longbench0random80` |
| `r100` | 100% random tokens | `random` |

New dataset aliases are configured as `long_context_r0`, `long_context_r20`, `long_context_r40`, `long_context_r60`, `long_context_r80`, and `long_context_r100`.

## Models And Methods

Default model/layer pairs:

| model | default layer |
|---|---:|
| `qwen3-8b` | 24 |
| `llama3.1-8b` | 18 |

Methods:

| method | Meaning |
|---|---|
| `sft` | LoRA SFT memorization baseline |
| `memft_ot` | MemFT only-threshold token selection |
| `memft_sw` | MemFT sliding-window token selection |

MemFT defaults to `r100` only. The `r0`-`r80` ratios are primarily intended for SFT law fitting.

## Commands

Train examples:

```bash
METHOD=sft bash script/long_context/run_long_context_train_qwen.sh
METHOD=sft bash script/long_context/run_long_context_train_llama.sh
METHOD=memft_ot bash script/long_context/run_long_context_train_qwen.sh
METHOD=memft_sw bash script/long_context/run_long_context_train_llama.sh
```

Apply/generation examples:

```bash
METHOD=sft bash script/long_context/run_long_context_apply_qwen.sh
METHOD=memft_ot bash script/long_context/run_long_context_apply_llama.sh
```

Final-loss examples:

```bash
METHOD=sft bash script/long_context/run_long_context_final_loss_qwen.sh
METHOD=memft_sw bash script/long_context/run_long_context_final_loss_llama.sh
```

Useful controls:

```bash
METHOD=sft RATIOS="r0 r20 r40" LENGTHS="50 100 200" RANKS="1 2 4" DRY_RUN=true \
  bash script/long_context/run_long_context_train_qwen.sh

METHOD=memft_ot RATIOS="r100" LENGTHS="1000 2000" RANKS="16 32" CUDA_VISIBLE_DEVICES=0 \
  bash script/long_context/run_long_context_apply_llama.sh
```

Supported environment variables:

| Variable | Meaning |
|---|---|
| `METHOD` | `sft`, `memft_ot`, or `memft_sw` |
| `RATIOS` | Ratio sweep, e.g. `r0 r20 r40 r60 r80 r100` |
| `LENGTHS` | Prefix-length sweep override |
| `RANKS` | LoRA rank sweep override |
| `LAYER` / `LAYERS` | Override default layer |
| `SEED` | Default `42` |
| `CUDA_VISIBLE_DEVICES` / `GPU_ID` | Device selection |
| `DRY_RUN` | Print commands without running |
| `SKIP_EXISTING` | Skip existing outputs, default `true` |
| `OVERWRITE` | Run even when outputs exist, default `false` |

## Default Sweeps

Qwen, layer 24:

- SFT `r100`: ranks `1 2 4 6 8 10 12 14 16 24 32 48 64 128 256 512`; lengths `50 100 200 500 1000 2000 3000 4000 5000 6000 7000 8000 10000`.
- SFT `r0`-`r80`: ranks `1 2 4 6 8 10 12 14 16`; same lengths as above.
- MemFT-OT / MemFT-SW `r100`: ranks `1 2 4 6 8 10 12 14 16 24 32 48 64 128 256 512`; lengths `50 100 200 500 1000 2000 3000 4000 5000 6000 7000`.

Llama, layer 18:

- SFT `r100`: ranks `1 2 4 6 8 10 12 14 16 24 32 48 64 128 256 512`; lengths `50 100 200 500 1000 2000 3000 4000 5000 6000 7000`.
- SFT `r0`-`r80`: ranks `1 2 4 6 8 10 12 14 16`; lengths `50 100 200 500 1000 2000 3000 4000 5000 6000 7000 8000 10000`.
- MemFT-OT / MemFT-SW `r100`: ranks `1 2 4 6 8 10 12 14 16 24 32 48 64 128 256 512`; lengths `50 100 200 500 1000 2000 3000 4000 5000 6000 7000`.

## Outputs

Training vectors:

```text
vectors/long_context/<method>/<model>/<ratio>/length_<L>/layer_<layer>/rank_<R>/
```

Apply/generation:

```text
generation/long_context/<method>/<model>/<ratio>/length_<L>/layer_<layer>/rank_<R>/
```

Final loss:

```text
generation/long_context/<method>/<model>/<ratio>/length_<L>/layer_<layer>/rank_<R>/final_loss/
```

## Notes

- The scripts preserve the old training hyperparameters: lr `1e-2`, epochs `2000`, batch size `1`, gradient accumulation `1`, `orig_add`, and LoRA over `mlp`.
- Output-length offsets follow the old scripts: Qwen SFT/MemFT-OT use `L + 36`, Qwen MemFT-SW uses `L + 31`, and Llama uses `L + 54`.
- The new dataset aliases point to `./datasets/long_context/`. Copy or symlink the old `EasyEdit_qwen_random/data/long_context/*.jsonl` files into that directory before running.
- Old scripts and old result directories are intentionally left untouched.

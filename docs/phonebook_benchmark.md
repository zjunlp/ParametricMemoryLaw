# PhoneBook Benchmark

PhoneBook is an exact parametric memory benchmark. Each example asks for a
stored phone-book style answer, and experiment size is organized by answer-only
token length. The standardized scripts here use the `random` subset for Qwen and
Llama runs.

The GitHub-facing entrypoints live in `script/pb/`. Older scripts directly under
`script/` are kept for historical reproducibility but are not recommended for new
runs.

## Data

Prepared PhoneBook data is stored under:

```text
easyedit_PB/datasets/phonebook/
```

Dataset names are registered in:

```text
hparams/Steer/dataset_format.yaml
```

Training dataset names follow:

```text
pb_random_qwen_<length>
pb_random_llama_<length>
```

Apply dataset names append `_apply`:

```text
pb_random_qwen_<length>_apply
pb_random_llama_<length>_apply
```

Data preparation utilities:

```bash
python tools/download_phonebook.py
python tools/prepare_phonebook_local.py
python tools/build_pb_length_subsets.py
```

In the prepared benchmark, Qwen targets are roughly 10 tokens and Llama targets
are roughly 4 tokens.

## Methods

Use these formal method names:

- `sft`: SFT baseline.
- `memft_ot`: MemFT only-threshold.
- `memft_sw`: MemFT curriculum/sliding-window style run, implemented by the
  current experiment code as `memft_curriculum`.

Historical naming:

- Old `memft` scripts correspond to `memft_ot`.
- Old `new` or curriculum scripts correspond to `memft_sw`.
- Old `mix*.sh` scripts are selected final-loss/apply launchers for past runs.

## Train

Qwen:

```bash
METHOD=sft bash script/pb/run_pb_train_qwen.sh
METHOD=memft_ot bash script/pb/run_pb_train_qwen.sh
METHOD=memft_sw bash script/pb/run_pb_train_qwen.sh
```

Llama:

```bash
METHOD=sft bash script/pb/run_pb_train_llama.sh
METHOD=memft_ot bash script/pb/run_pb_train_llama.sh
METHOD=memft_sw bash script/pb/run_pb_train_llama.sh
```

Default sweep:

```text
LENGTHS="1000 2000 4000 8000 12000 16000 24000 32000"
RANKS="1 2 4 8 16 32 64"
```

Qwen uses layer 24 by default. Llama uses layer 18 by default.

## Apply

Qwen:

```bash
METHOD=sft bash script/pb/run_pb_apply_qwen.sh
METHOD=memft_ot bash script/pb/run_pb_apply_qwen.sh
METHOD=memft_sw bash script/pb/run_pb_apply_qwen.sh
```

Llama:

```bash
METHOD=sft bash script/pb/run_pb_apply_llama.sh
METHOD=memft_ot bash script/pb/run_pb_apply_llama.sh
METHOD=memft_sw bash script/pb/run_pb_apply_llama.sh
```

## Final Loss

Qwen:

```bash
METHOD=sft bash script/pb/run_pb_final_loss_qwen.sh
METHOD=memft_ot bash script/pb/run_pb_final_loss_qwen.sh
METHOD=memft_sw bash script/pb/run_pb_final_loss_qwen.sh
```

Llama:

```bash
METHOD=sft bash script/pb/run_pb_final_loss_llama.sh
METHOD=memft_ot bash script/pb/run_pb_final_loss_llama.sh
METHOD=memft_sw bash script/pb/run_pb_final_loss_llama.sh
```

Target-only final loss is optional:

```bash
TARGET_ONLY_LOSS=true METHOD=sft bash script/pb/run_pb_final_loss_qwen.sh
```

## Output Layout

Training vectors:

```text
vectors/pb/<method>/<model>/random/length_<L>/layer_<layer>/rank_<R>/
```

Apply results:

```text
generation/pb/<method>/<model>/random/length_<L>/layer_<layer>/rank_<R>/
```

Final loss:

```text
generation/pb/<method>/<model>/random/length_<L>/layer_<layer>/rank_<R>/final_loss/
```

Each training leaf directory contains `train.log`, `losses.csv` when training
losses are emitted, and the LoRA vector under the existing trainer-managed
subdirectory:

```text
<dataset>/sft_lora/layer_<layer>.pt
<dataset>/memft_lora/layer_<layer>.pt
```

## Environment Variables

Common variables:

- `METHOD`: `sft`, `memft_ot`, or `memft_sw`.
- `LENGTHS`: space- or comma-separated lengths.
- `RANKS`: space- or comma-separated ranks.
- `LAYER` or `LAYERS`: layer override.
- `SEED`: default `42`.
- `GPU_ID`: GPU used by the launcher; defaults to `CUDA_VISIBLE_DEVICES` or `0`.
- `MODEL_PATH`: direct model path override.
- `QWEN3_8B_MODEL`: Qwen model path fallback.
- `LLAMA31_8B_MODEL`: Llama model path fallback.
- `PYTHON_BIN`: Python executable.
- `GENERATION_DATA_SIZE`: apply data limit; default `null`.
- `TARGET_ONLY_LOSS`: final-loss target-only toggle; default `false`.
- `FORCE`: set `true` to rerun a train job when `losses.csv` or the LoRA
  vector already exists in the target directory.

Examples:

```bash
METHOD=memft_sw LENGTHS="1000 2000" RANKS="4 8" GPU_ID=0 bash script/pb/run_pb_train_qwen.sh
METHOD=sft LAYER=18 MODEL_PATH=/models/llama-3.1-8b bash script/pb/run_pb_train_llama.sh
```

## Notes

- Do not run multiple jobs that write to the same output directory at the same
  time.
- If memory is insufficient, reduce batch size in the script or raise gradient
  accumulation for that method/length.
- MemFT-SW uses length-dependent hyperparameters defined in `script/pb/common.sh`.
- Llama SFT and MemFT-OT use batch size 10 through 16k and batch size 20 for
  24k/32k.
- Qwen and Llama PhoneBook lengths correspond to different approximate sample
  counts because their tokenizers differ.

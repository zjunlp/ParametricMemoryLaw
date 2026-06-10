# Legacy PhoneBook Scripts

The shell scripts directly under `script/` are retained as historical experiment
entrypoints. They are not deleted or moved so old paths, logs, and result
directories remain reproducible.

For new runs and GitHub documentation, prefer the standardized entrypoints in
`script/pb/`:

- `run_pb_train_qwen.sh`
- `run_pb_train_llama.sh`
- `run_pb_apply_qwen.sh`
- `run_pb_apply_llama.sh`
- `run_pb_final_loss_qwen.sh`
- `run_pb_final_loss_llama.sh`

Historical naming map:

- `memft` means the formal `memft_ot` method.
- `new` or curriculum PhoneBook scripts mean the formal `memft_sw` method.
- `mix*.sh` scripts combine selected final-loss and apply jobs for past runs and
  are not recommended as general entrypoints.

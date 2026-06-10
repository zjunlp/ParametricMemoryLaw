#!/usr/bin/env bash

set -u

LC_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LC_REPO_ROOT="$(cd "${LC_SCRIPT_DIR}/../.." && pwd)"

LC_SEED="${SEED:-42}"
LC_GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
LC_PYTHON_BIN="${PYTHON_BIN:-python}"
LC_DATASET_FORMAT="${DATASET_FORMAT:-hparams/Steer/dataset_format.yaml}"
LC_SFT_HPARAM="${SFT_HPARAM:-hparams/context2vec_long/long_context_sft_lora.yaml}"
LC_MEMFT_HPARAM="${MEMFT_HPARAM:-hparams/context2vec_long/long_context_memft_lora.yaml}"
LC_APPLY_SFT_HPARAM="${APPLY_SFT_HPARAM:-hparams/Steer/context2vec/our_lora/apply_our_lora.yaml}"
LC_APPLY_MEMFT_HPARAM="${APPLY_MEMFT_HPARAM:-hparams/context2vec_long/apply_memft_lora.yaml}"
LC_SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-tools/summarize_pb_final_loss.py}"
LC_GENERATION_DATA_SIZE="${GENERATION_DATA_SIZE:-null}"
LC_TARGET_ONLY_LOSS="${TARGET_ONLY_LOSS:-false}"
LC_DRY_RUN="${DRY_RUN:-false}"
LC_SKIP_EXISTING="${SKIP_EXISTING:-true}"
LC_OVERWRITE="${OVERWRITE:-false}"
LC_CURRENT_PID=""

lc_cleanup() {
  local signal_name="${1:-INT}"
  echo "Received ${signal_name}. Stopping current long-context job..."
  if [ -n "${LC_CURRENT_PID}" ] && kill -0 "${LC_CURRENT_PID}" 2>/dev/null; then
    kill -TERM "-${LC_CURRENT_PID}" 2>/dev/null || kill -TERM "${LC_CURRENT_PID}" 2>/dev/null || true
    wait "${LC_CURRENT_PID}" 2>/dev/null || true
  fi
  exit 130
}

trap 'lc_cleanup INT' INT
trap 'lc_cleanup TERM' TERM
trap 'lc_cleanup TSTP' TSTP

lc_parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  LC_PARSED_LIST=(${raw})
}

lc_init_model() {
  local model_family="$1"
  LC_MODEL_FAMILY="${model_family}"

  case "${model_family}" in
    qwen)
      LC_MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
      LC_MODEL_PATH="${MODEL_PATH:-${QWEN3_8B_MODEL:-/mnt/16t/share/Qwen3-8B}}"
      LC_DEFAULT_LAYER="${LAYER:-24}"
      LC_OUTPUT_OFFSET_SFT=36
      LC_OUTPUT_OFFSET_MEMFT_OT=36
      LC_OUTPUT_OFFSET_MEMFT_SW=31
      ;;
    llama)
      LC_MODEL_NAME="${MODEL_NAME:-llama3.1-8b}"
      LC_MODEL_PATH="${MODEL_PATH:-${LLAMA31_8B_MODEL:-/mnt/16t/share/llama-3.1-8b-instruct}}"
      LC_DEFAULT_LAYER="${LAYER:-18}"
      LC_OUTPUT_OFFSET_SFT=54
      LC_OUTPUT_OFFSET_MEMFT_OT=54
      LC_OUTPUT_OFFSET_MEMFT_SW=54
      ;;
    *)
      echo "error: unsupported model family: ${model_family}" >&2
      exit 2
      ;;
  esac
}

lc_validate_method() {
  LC_METHOD="${METHOD:-sft}"
  case "${LC_METHOD}" in
    sft|memft_ot|memft_sw) ;;
    *)
      echo "error: METHOD must be one of: sft, memft_ot, memft_sw; got '${LC_METHOD}'" >&2
      exit 2
      ;;
  esac
}

lc_vector_subdir() {
  case "${LC_METHOD}" in
    sft) echo "sft_lora" ;;
    memft_ot|memft_sw) echo "memft_lora" ;;
  esac
}

lc_train_hparam() {
  case "${LC_METHOD}" in
    sft) echo "${LC_SFT_HPARAM}" ;;
    memft_ot|memft_sw) echo "${LC_MEMFT_HPARAM}" ;;
  esac
}

lc_apply_hparam() {
  case "${LC_METHOD}" in
    sft) echo "${LC_APPLY_SFT_HPARAM}" ;;
    memft_ot|memft_sw) echo "${LC_APPLY_MEMFT_HPARAM}" ;;
  esac
}

lc_ratio_dataset() {
  local ratio="$1"
  case "${ratio}" in
    r0|r20|r40|r60|r80|r100) echo "long_context_${ratio}" ;;
    *)
      echo "error: unsupported ratio '${ratio}'. Use r0 r20 r40 r60 r80 r100." >&2
      exit 2
      ;;
  esac
}

lc_dataset_configured() {
  local dataset_name="$1"
  grep -Eq "^[[:space:]]+${dataset_name}:" "${LC_DATASET_FORMAT}"
}

lc_default_ratios() {
  if [ -n "${RATIOS:-}" ]; then
    echo "${RATIOS}"
  elif [ "${LC_METHOD}" = "sft" ]; then
    echo "r0 r20 r40 r60 r80 r100"
  else
    echo "r100"
  fi
}

lc_default_lengths_for_ratio() {
  local ratio="$1"
  if [ -n "${LENGTHS:-}" ]; then
    echo "${LENGTHS}"
    return
  fi

  case "${LC_MODEL_FAMILY}:${LC_METHOD}:${ratio}" in
    qwen:sft:*) echo "50 100 200 500 1000 2000 3000 4000 5000 6000 7000 8000 10000" ;;
    qwen:memft_ot:r100|qwen:memft_sw:r100) echo "50 100 200 500 1000 2000 3000 4000 5000 6000 7000" ;;
    llama:sft:r100) echo "50 100 200 500 1000 2000 3000 4000 5000 6000 7000" ;;
    llama:sft:*) echo "50 100 200 500 1000 2000 3000 4000 5000 6000 7000 8000 10000" ;;
    llama:memft_ot:r100|llama:memft_sw:r100) echo "50 100 200 500 1000 2000 3000 4000 5000 6000 7000" ;;
    *)
      echo "error: no default LENGTHS for ${LC_MODEL_FAMILY}/${LC_METHOD}/${ratio}. Set LENGTHS explicitly." >&2
      exit 2
      ;;
  esac
}

lc_default_ranks_for_ratio() {
  local ratio="$1"
  if [ -n "${RANKS:-}" ]; then
    echo "${RANKS}"
    return
  fi

  case "${LC_METHOD}:${ratio}" in
    sft:r100|memft_ot:r100|memft_sw:r100) echo "1 2 4 6 8 10 12 14 16 24 32 48 64 128 256 512" ;;
    sft:*) echo "1 2 4 6 8 10 12 14 16" ;;
    *)
      echo "error: no default RANKS for ${LC_METHOD}/${ratio}. Set RANKS explicitly." >&2
      exit 2
      ;;
  esac
}

lc_default_layers() {
  echo "${LAYERS:-${LC_DEFAULT_LAYER}}"
}

lc_output_length() {
  local length="$1"
  local offset
  case "${LC_METHOD}" in
    sft) offset="${LC_OUTPUT_OFFSET_SFT}" ;;
    memft_ot) offset="${LC_OUTPUT_OFFSET_MEMFT_OT}" ;;
    memft_sw) offset="${LC_OUTPUT_OFFSET_MEMFT_SW}" ;;
  esac
  echo $((length + offset))
}

lc_print_cmd() {
  printf ' %q' "$@"
  printf '\n'
}

lc_run_train() {
  local model_family="$1"
  cd "${LC_REPO_ROOT}"
  lc_init_model "${model_family}"
  lc_validate_method

  local ratios layers lora_subdir train_hparam
  lc_parse_list "$(lc_default_ratios)"; ratios=("${LC_PARSED_LIST[@]}")
  lc_parse_list "$(lc_default_layers)"; layers=("${LC_PARSED_LIST[@]}")
  lora_subdir="$(lc_vector_subdir)"
  train_hparam="$(lc_train_hparam)"

  echo "Long-context train entry"
  echo "model=${LC_MODEL_NAME} method=${LC_METHOD} ratios=${ratios[*]} layers=${layers[*]} seed=${LC_SEED} gpu_id=${LC_GPU_ID}"

  for ratio in "${ratios[@]}"; do
    local dataset_name
    dataset_name="$(lc_ratio_dataset "${ratio}")"
    if ! lc_dataset_configured "${dataset_name}"; then
      echo "error: missing train dataset ${dataset_name} in ${LC_DATASET_FORMAT}" >&2
      exit 2
    fi

    local lengths ranks
    lc_parse_list "$(lc_default_lengths_for_ratio "${ratio}")"; lengths=("${LC_PARSED_LIST[@]}")
    lc_parse_list "$(lc_default_ranks_for_ratio "${ratio}")"; ranks=("${LC_PARSED_LIST[@]}")
    if [ "${#lengths[@]}" -eq 0 ] || [ "${#ranks[@]}" -eq 0 ]; then
      echo "error: empty sweep for ${LC_MODEL_FAMILY}/${LC_METHOD}/${ratio}; set LENGTHS and RANKS explicitly if using a non-default sweep." >&2
      exit 2
    fi

    for length in "${lengths[@]}"; do
      for layer in "${layers[@]}"; do
        for rank in "${ranks[@]}"; do
          local run_dir="vectors/long_context/${LC_METHOD}/${LC_MODEL_NAME}/${ratio}/length_${length}/layer_${layer}/rank_${rank}"
          local vector_file="${run_dir}/${dataset_name}/${lora_subdir}/layer_${layer}.pt"
          local loss_file="${run_dir}/losses.csv"
          local raw_loss_file="${run_dir}/train_losses.csv"
          local log_file="${run_dir}/train.log"
          local output_length
          output_length="$(lc_output_length "${length}")"

          if [ "${LC_SKIP_EXISTING}" = "true" ] && [ "${LC_OVERWRITE}" != "true" ] && { [ -f "${vector_file}" ] || [ -f "${loss_file}" ]; }; then
            echo "skip existing: ${run_dir}"
            continue
          fi

          local cmd=(
            "${LC_PYTHON_BIN}" vectors_generate.py
            "model_name_or_path=${LC_MODEL_PATH}"
            "seed=${LC_SEED}"
            "steer_train_dataset=[${dataset_name}]"
            "steer_vector_output_dirs=[${run_dir}]"
            "steer_train_hparam_paths=[${train_hparam}]"
            "+loss_output_dir=${run_dir}"
            "+layers=[${layer}]"
            "+preference_pairs=[orig_add]"
            "+sparsity_weight=0.0"
            "+output_length=${output_length}"
            "+n_epochs=2000"
            "+batch_size=1"
            "+gradient_accumulation_steps=1"
            "+lr=1e-2"
            "+low_rank_dimension=${rank}"
          )

          if [ "${LC_METHOD}" = "memft_ot" ]; then
            cmd+=(
              "+use_memft=true"
              "+memft_method=only_threshold"
              "+memft_threshold=0.5"
              "+use_position_weight=false"
            )
          elif [ "${LC_METHOD}" = "memft_sw" ]; then
            cmd+=(
              "+use_memft=true"
              "+memft_method=sliding"
              "+memft_threshold=0.5"
              "+use_sliding_window=true"
              "+memory_window_size=50"
              "+use_position_weight=false"
            )
          fi

          cmd+=(
            "hydra.run.dir=."
            "hydra.output_subdir=null"
            "hydra/job_logging=disabled"
            "hydra/hydra_logging=disabled"
            "hydra.job.chdir=false"
          )

          echo "========================================"
          echo "train model=${LC_MODEL_NAME} method=${LC_METHOD} ratio=${ratio} dataset=${dataset_name} length=${length} layer=${layer} rank=${rank}"
          echo "output_dir=${run_dir} output_length=${output_length}"
          if [ "${LC_DRY_RUN}" = "true" ]; then
            lc_print_cmd "CUDA_VISIBLE_DEVICES=${LC_GPU_ID}" "${cmd[@]}"
            continue
          fi

          mkdir -p "${run_dir}"
          {
            echo "model=${LC_MODEL_NAME}"
            echo "method=${LC_METHOD}"
            echo "ratio=${ratio}"
            echo "dataset=${dataset_name}"
            echo "length=${length}"
            echo "layer=${layer}"
            echo "rank=${rank}"
            echo "seed=${LC_SEED}"
            echo "output_length=${output_length}"
            echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
          } >> "${log_file}"

          CUDA_VISIBLE_DEVICES="${LC_GPU_ID}" "${cmd[@]}" >> "${log_file}" 2>&1 &
          LC_CURRENT_PID="$!"
          wait "${LC_CURRENT_PID}"
          local status=$?
          LC_CURRENT_PID=""
          if [ "${status}" -ne 0 ]; then
            echo "error: train failed; see ${log_file}" >&2
            exit "${status}"
          fi
          if [ -f "${raw_loss_file}" ]; then
            mv -f "${raw_loss_file}" "${loss_file}"
          fi
          echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')" >> "${log_file}"
        done
      done
    done
  done
}

lc_run_apply() {
  local model_family="$1"
  cd "${LC_REPO_ROOT}"
  lc_init_model "${model_family}"
  lc_validate_method

  local ratios layers lora_subdir apply_hparam
  lc_parse_list "$(lc_default_ratios)"; ratios=("${LC_PARSED_LIST[@]}")
  lc_parse_list "$(lc_default_layers)"; layers=("${LC_PARSED_LIST[@]}")
  lora_subdir="$(lc_vector_subdir)"
  apply_hparam="$(lc_apply_hparam)"

  echo "Long-context apply entry"
  echo "model=${LC_MODEL_NAME} method=${LC_METHOD} ratios=${ratios[*]} layers=${layers[*]} seed=${LC_SEED} gpu_id=${LC_GPU_ID}"

  for ratio in "${ratios[@]}"; do
    local dataset_name
    dataset_name="$(lc_ratio_dataset "${ratio}")"
    if ! lc_dataset_configured "${dataset_name}"; then
      echo "error: missing generation dataset ${dataset_name} in ${LC_DATASET_FORMAT}" >&2
      exit 2
    fi

    local lengths ranks
    lc_parse_list "$(lc_default_lengths_for_ratio "${ratio}")"; lengths=("${LC_PARSED_LIST[@]}")
    lc_parse_list "$(lc_default_ranks_for_ratio "${ratio}")"; ranks=("${LC_PARSED_LIST[@]}")
    if [ "${#lengths[@]}" -eq 0 ] || [ "${#ranks[@]}" -eq 0 ]; then
      echo "error: empty sweep for ${LC_MODEL_FAMILY}/${LC_METHOD}/${ratio}; set LENGTHS and RANKS explicitly if using a non-default sweep." >&2
      exit 2
    fi

    for length in "${lengths[@]}"; do
      for layer in "${layers[@]}"; do
        for rank in "${ranks[@]}"; do
          local run_dir="vectors/long_context/${LC_METHOD}/${LC_MODEL_NAME}/${ratio}/length_${length}/layer_${layer}/rank_${rank}"
          local lora_dir="${run_dir}/${dataset_name}/${lora_subdir}"
          local lora_file="${lora_dir}/layer_${layer}.pt"
          local output_dir="generation/long_context/${LC_METHOD}/${LC_MODEL_NAME}/${ratio}/length_${length}/layer_${layer}/rank_${rank}"
          local result_file="${output_dir}/${dataset_name}_results.json"
          local log_file="${output_dir}/apply.log"
          local max_new_tokens="${GENERATION_MAX_NEW_TOKENS:-${length}}"

          if [ ! -f "${lora_file}" ]; then
            echo "warning: missing LoRA file ${lora_file}; skipping"
            continue
          fi
          if [ "${LC_SKIP_EXISTING}" = "true" ] && [ "${LC_OVERWRITE}" != "true" ] && [ -f "${result_file}" ]; then
            echo "skip existing: ${result_file}"
            continue
          fi

          local cmd=(
            "${LC_PYTHON_BIN}" vectors_apply.py
            "model_name_or_path=${LC_MODEL_PATH}"
            "use_chat_template=true"
            "apply_steer_hparam_paths=[${apply_hparam}]"
            "steer_vector_load_dir=[${lora_dir}]"
            "generation_data=[${dataset_name}]"
            "generation_data_size=${LC_GENERATION_DATA_SIZE}"
            "generation_output_dir=${output_dir}"
            "generate_orig_output=true"
            "generation_params.do_sample=false"
            "generation_params.max_new_tokens=${max_new_tokens}"
            "+layers=[${layer}]"
            "+intervention_method=lora"
            "+concept_id=0"
            "hydra.run.dir=."
            "hydra.output_subdir=null"
            "hydra.job.chdir=false"
          )

          echo "========================================"
          echo "apply model=${LC_MODEL_NAME} method=${LC_METHOD} ratio=${ratio} dataset=${dataset_name} length=${length} layer=${layer} rank=${rank}"
          echo "vector_dir=${lora_dir} output_dir=${output_dir} max_new_tokens=${max_new_tokens}"
          if [ "${LC_DRY_RUN}" = "true" ]; then
            lc_print_cmd "CUDA_VISIBLE_DEVICES=${LC_GPU_ID}" "${cmd[@]}"
            continue
          fi

          mkdir -p "${output_dir}"
          CUDA_VISIBLE_DEVICES="${LC_GPU_ID}" "${cmd[@]}" >> "${log_file}" 2>&1 &
          LC_CURRENT_PID="$!"
          wait "${LC_CURRENT_PID}"
          local status=$?
          LC_CURRENT_PID=""
          if [ "${status}" -ne 0 ]; then
            echo "warning: apply failed; see ${log_file}"
          else
            echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')" >> "${log_file}"
          fi
        done
      done
    done
  done
}

lc_run_final_loss() {
  local model_family="$1"
  cd "${LC_REPO_ROOT}"
  lc_init_model "${model_family}"
  lc_validate_method

  local ratios layers lora_subdir
  lc_parse_list "$(lc_default_ratios)"; ratios=("${LC_PARSED_LIST[@]}")
  lc_parse_list "$(lc_default_layers)"; layers=("${LC_PARSED_LIST[@]}")
  lora_subdir="$(lc_vector_subdir)"

  echo "Long-context final-loss entry"
  echo "model=${LC_MODEL_NAME} method=${LC_METHOD} ratios=${ratios[*]} layers=${layers[*]} seed=${LC_SEED} gpu_id=${LC_GPU_ID}"

  for ratio in "${ratios[@]}"; do
    local dataset_name
    dataset_name="$(lc_ratio_dataset "${ratio}")"
    if ! lc_dataset_configured "${dataset_name}"; then
      echo "error: missing train dataset ${dataset_name} in ${LC_DATASET_FORMAT}" >&2
      exit 2
    fi

    local lengths ranks
    lc_parse_list "$(lc_default_lengths_for_ratio "${ratio}")"; lengths=("${LC_PARSED_LIST[@]}")
    lc_parse_list "$(lc_default_ranks_for_ratio "${ratio}")"; ranks=("${LC_PARSED_LIST[@]}")
    if [ "${#lengths[@]}" -eq 0 ] || [ "${#ranks[@]}" -eq 0 ]; then
      echo "error: empty sweep for ${LC_MODEL_FAMILY}/${LC_METHOD}/${ratio}; set LENGTHS and RANKS explicitly if using a non-default sweep." >&2
      exit 2
    fi

    for length in "${lengths[@]}"; do
      for layer in "${layers[@]}"; do
        for rank in "${ranks[@]}"; do
          local run_dir="vectors/long_context/${LC_METHOD}/${LC_MODEL_NAME}/${ratio}/length_${length}/layer_${layer}/rank_${rank}"
          local init_vector_path="${run_dir}/${dataset_name}/${lora_subdir}"
          local lora_file="${init_vector_path}/layer_${layer}.pt"
          local output_dir="generation/long_context/${LC_METHOD}/${LC_MODEL_NAME}/${ratio}/length_${length}/layer_${layer}/rank_${rank}/final_loss"
          local tmp_vector_dir="${output_dir}/tmp_vectors"
          local loss_csv="${output_dir}/train_losses.csv"
          local log_file="${output_dir}/final_loss.log"
          local output_length
          output_length="$(lc_output_length "${length}")"

          if [ ! -f "${lora_file}" ]; then
            echo "warning: missing LoRA file ${lora_file}; skipping"
            continue
          fi
          if [ "${LC_SKIP_EXISTING}" = "true" ] && [ "${LC_OVERWRITE}" != "true" ] && [ -f "${output_dir}/summary_final_loss.csv" ]; then
            echo "skip existing: ${output_dir}/summary_final_loss.csv"
            continue
          fi

          local cmd=(
            "${LC_PYTHON_BIN}" vectors_generate.py
            "model_name_or_path=${LC_MODEL_PATH}"
            "use_chat_template=true"
            "seed=${LC_SEED}"
            "steer_train_hparam_paths=[${LC_SFT_HPARAM}]"
            "steer_train_dataset=[${dataset_name}]"
            "steer_vector_output_dirs=[${tmp_vector_dir}]"
            "save_vectors=false"
            "+loss_output_dir=${output_dir}"
            "+layers=[${layer}]"
            "+n_epochs=1"
            "+batch_size=1"
            "+gradient_accumulation_steps=1"
            "+inference=true"
            "+target_only_loss=${LC_TARGET_ONLY_LOSS}"
            "+target_only_raw_target_field=target"
            "+init_vector_path=${init_vector_path}"
            "+intervention_method=lora"
            "+intervention_components=mlp"
            "+low_rank_dimension=${rank}"
            "+output_length=${output_length}"
            "+dropout=0.0"
            "+intervention_positions_dropout=0.0"
            "hydra.run.dir=."
            "hydra.output_subdir=null"
            "hydra.job.chdir=false"
          )

          echo "========================================"
          echo "final_loss model=${LC_MODEL_NAME} method=${LC_METHOD} ratio=${ratio} dataset=${dataset_name} length=${length} layer=${layer} rank=${rank}"
          echo "init_vector_path=${init_vector_path} output_dir=${output_dir} output_length=${output_length}"
          if [ "${LC_DRY_RUN}" = "true" ]; then
            lc_print_cmd "CUDA_VISIBLE_DEVICES=${LC_GPU_ID}" "${cmd[@]}"
            continue
          fi

          mkdir -p "${output_dir}"
          CUDA_VISIBLE_DEVICES="${LC_GPU_ID}" "${cmd[@]}" >> "${log_file}" 2>&1 &
          LC_CURRENT_PID="$!"
          wait "${LC_CURRENT_PID}"
          local status=$?
          LC_CURRENT_PID=""
          if [ "${status}" -ne 0 ]; then
            echo "warning: final-loss failed; see ${log_file}"
            continue
          fi

          if [ -f "${loss_csv}" ]; then
            if "${LC_PYTHON_BIN}" "${LC_SUMMARY_SCRIPT}" \
              --root "${output_dir}" \
              --output "${output_dir}/summary_final_loss.csv" \
              --write-per-dir-summary >> "${log_file}" 2>&1; then
              echo "finished: ${output_dir}"
            else
              echo "warning: summary failed for ${output_dir}; see ${log_file}"
            fi
          else
            echo "warning: final-loss finished but ${loss_csv} was not created; see ${log_file}"
          fi
          echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')" >> "${log_file}"
        done
      done
    done
  done
}

#!/usr/bin/env bash

set -u

PB_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PB_REPO_ROOT="$(cd "${PB_SCRIPT_DIR}/../.." && pwd)"

PB_SUBSET="${SUBSET:-random}"
PB_SEED="${SEED:-42}"
PB_OUTPUT_LENGTH="${OUTPUT_LENGTH:-128}"
PB_GPU_ID="${GPU_ID:-${CUDA_VISIBLE_DEVICES:-0}}"
PB_PYTHON_BIN="${PYTHON_BIN:-python}"
PB_DATASET_FORMAT="${DATASET_FORMAT:-hparams/Steer/dataset_format.yaml}"
PB_APPLY_SFT_HPARAM="${APPLY_SFT_HPARAM:-${APPLY_HPARAM:-hparams/Steer/context2vec/our_lora/apply_our_lora.yaml}}"
PB_APPLY_MEMFT_HPARAM="${APPLY_MEMFT_HPARAM:-hparams/context2vec_pb/apply_memft_lora.yaml}"
PB_SFT_HPARAM="${SFT_HPARAM:-hparams/context2vec_pb/pb_sft_lora.yaml}"
PB_MEMFT_HPARAM="${MEMFT_HPARAM:-hparams/context2vec_pb/pb_memft_lora.yaml}"
PB_SUMMARY_SCRIPT="${SUMMARY_SCRIPT:-tools/summarize_pb_final_loss.py}"
PB_GENERATION_DATA_SIZE="${GENERATION_DATA_SIZE:-null}"
PB_TARGET_ONLY_LOSS="${TARGET_ONLY_LOSS:-false}"
PB_TARGET_ONLY_DEBUG_FIRST="${TARGET_ONLY_DEBUG_FIRST:-false}"
PB_FORCE="${FORCE:-false}"
PB_CURRENT_PID=""

pb_cleanup() {
  local signal_name="${1:-INT}"
  echo "Received ${signal_name}. Stopping current PhoneBook job..."
  if [ -n "${PB_CURRENT_PID}" ] && kill -0 "${PB_CURRENT_PID}" 2>/dev/null; then
    kill -TERM "-${PB_CURRENT_PID}" 2>/dev/null || kill -TERM "${PB_CURRENT_PID}" 2>/dev/null || true
    sleep 5
    if kill -0 "${PB_CURRENT_PID}" 2>/dev/null; then
      kill -KILL "-${PB_CURRENT_PID}" 2>/dev/null || kill -KILL "${PB_CURRENT_PID}" 2>/dev/null || true
    fi
    wait "${PB_CURRENT_PID}" 2>/dev/null || true
  fi
  exit 130
}

trap 'pb_cleanup INT' INT
trap 'pb_cleanup TERM' TERM
trap 'pb_cleanup TSTP' TSTP

pb_parse_list() {
  local raw="$1"
  raw="${raw//,/ }"
  # shellcheck disable=SC2206
  PB_PARSED_LIST=(${raw})
}

pb_init_model() {
  local model_family="$1"
  PB_MODEL_FAMILY="${model_family}"

  case "${model_family}" in
    qwen)
      PB_MODEL_NAME="${MODEL_NAME:-qwen3-8b}"
      PB_MODEL_PATH="${MODEL_PATH:-${QWEN3_8B_MODEL:-/mnt/16t/share/Qwen3-8B}}"
      PB_DEFAULT_LAYER="${LAYER:-24}"
      ;;
    llama)
      PB_MODEL_NAME="${MODEL_NAME:-llama3.1-8b}"
      PB_MODEL_PATH="${MODEL_PATH:-${LLAMA31_8B_MODEL:-/mnt/16t/share/llama-3.1-8b-instruct}}"
      PB_DEFAULT_LAYER="${LAYER:-18}"
      ;;
    *)
      echo "error: unsupported model family: ${model_family}" >&2
      exit 2
      ;;
  esac
}

pb_validate_method() {
  PB_METHOD="${METHOD:-sft}"
  case "${PB_METHOD}" in
    sft|memft_ot|memft_sw) ;;
    *)
      echo "error: METHOD must be one of: sft, memft_ot, memft_sw; got '${PB_METHOD}'" >&2
      exit 2
      ;;
  esac
}

pb_vector_subdir() {
  case "${PB_METHOD}" in
    sft) echo "sft_lora" ;;
    memft_ot|memft_sw) echo "memft_lora" ;;
  esac
}

pb_train_hparam() {
  case "${PB_METHOD}" in
    sft) echo "${PB_SFT_HPARAM}" ;;
    memft_ot|memft_sw) echo "${PB_MEMFT_HPARAM}" ;;
  esac
}

pb_apply_hparam() {
  case "${PB_METHOD}" in
    sft) echo "${PB_APPLY_SFT_HPARAM}" ;;
    memft_ot|memft_sw) echo "${PB_APPLY_MEMFT_HPARAM}" ;;
  esac
}

pb_dataset_configured() {
  local dataset_name="$1"
  grep -Eq "^[[:space:]]+${dataset_name}:" "${PB_DATASET_FORMAT}"
}

pb_default_lengths() {
  echo "${LENGTHS:-1000 2000 4000 8000 12000 16000 24000 32000}"
}

pb_default_ranks() {
  echo "${RANKS:-1 2 4 8 16 32 64}"
}

pb_default_layers() {
  echo "${LAYERS:-${PB_DEFAULT_LAYER}}"
}

pb_get_qwen_sft_hparams() {
  local length="$1"
  PB_LR="1e-2"
  PB_EPOCHS=200
  PB_GRAD_ACCUM=1
  PB_CURRICULUM_BOUNDARIES=""

  case "${length}" in
    1000|2000|4000) PB_BATCH_SIZE=4 ;;
    8000|12000|16000|24000) PB_BATCH_SIZE=8 ;;
    32000) PB_BATCH_SIZE=16 ;;
    *) echo "error: unsupported Qwen SFT length: ${length}" >&2; exit 2 ;;
  esac
}

pb_get_qwen_memft_ot_hparams() {
  pb_get_qwen_sft_hparams "$1"
}

pb_get_qwen_memft_sw_hparams() {
  local length="$1"
  case "${length}" in
    1000|2000)
      PB_LR="1e-2"; PB_EPOCHS=300; PB_BATCH_SIZE=10; PB_GRAD_ACCUM=1
      PB_CURRICULUM_BOUNDARIES="[20,40,60,80,300]"
      ;;
    4000)
      PB_LR="1e-2"; PB_EPOCHS=350; PB_BATCH_SIZE=10; PB_GRAD_ACCUM=1
      PB_CURRICULUM_BOUNDARIES="[30,60,90,120,350]"
      ;;
    8000)
      PB_LR="7e-3"; PB_EPOCHS=350; PB_BATCH_SIZE=20; PB_GRAD_ACCUM=2
      PB_CURRICULUM_BOUNDARIES="[30,60,90,120,350]"
      ;;
    12000)
      PB_LR="5e-3"; PB_EPOCHS=500; PB_BATCH_SIZE=40; PB_GRAD_ACCUM=2
      PB_CURRICULUM_BOUNDARIES="[60,120,180,240,500]"
      ;;
    16000|24000)
      PB_LR="5e-3"; PB_EPOCHS=600; PB_BATCH_SIZE=40; PB_GRAD_ACCUM=2
      PB_CURRICULUM_BOUNDARIES="[80,160,240,320,600]"
      ;;
    32000)
      PB_LR="5e-3"; PB_EPOCHS=700; PB_BATCH_SIZE=40; PB_GRAD_ACCUM=2
      PB_CURRICULUM_BOUNDARIES="[100,200,300,400,700]"
      ;;
    *) echo "error: unsupported Qwen MemFT-SW length: ${length}" >&2; exit 2 ;;
  esac
}

pb_get_llama_sft_hparams() {
  local length="$1"
  PB_LR="1e-2"
  PB_EPOCHS=200
  PB_GRAD_ACCUM=1
  PB_CURRICULUM_BOUNDARIES=""

  case "${length}" in
    1000|2000|4000|8000|12000|16000) PB_BATCH_SIZE=10 ;;
    24000|32000) PB_BATCH_SIZE=20 ;;
    *) echo "error: unsupported Llama SFT length: ${length}" >&2; exit 2 ;;
  esac
}

pb_get_llama_memft_ot_hparams() {
  pb_get_llama_sft_hparams "$1"
}

pb_get_llama_memft_sw_hparams() {
  local length="$1"
  case "${length}" in
    1000)
      PB_LR="1e-2"; PB_EPOCHS=300; PB_BATCH_SIZE=25; PB_GRAD_ACCUM=1
      PB_CURRICULUM_BOUNDARIES="[20,40,60,80,300]"
      ;;
    2000)
      PB_LR="1e-2"; PB_EPOCHS=400; PB_BATCH_SIZE=25; PB_GRAD_ACCUM=1
      PB_CURRICULUM_BOUNDARIES="[40,80,120,160,400]"
      ;;
    4000)
      PB_LR="7e-3"; PB_EPOCHS=400; PB_BATCH_SIZE=50; PB_GRAD_ACCUM=1
      PB_CURRICULUM_BOUNDARIES="[40,80,120,160,400]"
      ;;
    8000)
      PB_LR="5e-3"; PB_EPOCHS=600; PB_BATCH_SIZE=50; PB_GRAD_ACCUM=2
      PB_CURRICULUM_BOUNDARIES="[80,160,240,320,600]"
      ;;
    12000|16000)
      PB_LR="5e-3"; PB_EPOCHS=700; PB_BATCH_SIZE=50; PB_GRAD_ACCUM=2
      PB_CURRICULUM_BOUNDARIES="[100,200,300,400,700]"
      ;;
    24000)
      PB_LR="3e-3"; PB_EPOCHS=700; PB_BATCH_SIZE=50; PB_GRAD_ACCUM=4
      PB_CURRICULUM_BOUNDARIES="[100,200,300,400,700]"
      ;;
    32000)
      PB_LR="3e-3"; PB_EPOCHS=800; PB_BATCH_SIZE=50; PB_GRAD_ACCUM=4
      PB_CURRICULUM_BOUNDARIES="[120,240,360,480,800]"
      ;;
    *) echo "error: unsupported Llama MemFT-SW length: ${length}" >&2; exit 2 ;;
  esac
}

pb_get_hparams() {
  local length="$1"
  case "${PB_MODEL_FAMILY}:${PB_METHOD}" in
    qwen:sft) pb_get_qwen_sft_hparams "${length}" ;;
    qwen:memft_ot) pb_get_qwen_memft_ot_hparams "${length}" ;;
    qwen:memft_sw) pb_get_qwen_memft_sw_hparams "${length}" ;;
    llama:sft) pb_get_llama_sft_hparams "${length}" ;;
    llama:memft_ot) pb_get_llama_memft_ot_hparams "${length}" ;;
    llama:memft_sw) pb_get_llama_memft_sw_hparams "${length}" ;;
    *) echo "error: unsupported model/method: ${PB_MODEL_FAMILY}/${PB_METHOD}" >&2; exit 2 ;;
  esac
}

pb_run_train() {
  local model_family="$1"
  cd "${PB_REPO_ROOT}"
  pb_init_model "${model_family}"
  pb_validate_method

  local lengths ranks layers
  pb_parse_list "$(pb_default_lengths)"; lengths=("${PB_PARSED_LIST[@]}")
  pb_parse_list "$(pb_default_ranks)"; ranks=("${PB_PARSED_LIST[@]}")
  pb_parse_list "$(pb_default_layers)"; layers=("${PB_PARSED_LIST[@]}")

  echo "PhoneBook train entry"
  echo "model=${PB_MODEL_NAME} model_family=${PB_MODEL_FAMILY} method=${PB_METHOD} subset=${PB_SUBSET} seed=${PB_SEED}"
  echo "lengths=${lengths[*]} ranks=${ranks[*]} layers=${layers[*]} gpu_id=${PB_GPU_ID}"

  for length in "${lengths[@]}"; do
    local dataset_name="pb_${PB_SUBSET}_${PB_MODEL_FAMILY}_${length}"
    if ! pb_dataset_configured "${dataset_name}"; then
      echo "error: missing train dataset ${dataset_name} in ${PB_DATASET_FORMAT}" >&2
      exit 2
    fi

    pb_get_hparams "${length}"

    for layer in "${layers[@]}"; do
      for rank in "${ranks[@]}"; do
        local run_dir="vectors/pb/${PB_METHOD}/${PB_MODEL_NAME}/${PB_SUBSET}/length_${length}/layer_${layer}/rank_${rank}"
        local log_file="${run_dir}/train.log"
        local vector_file="${run_dir}/${dataset_name}/$(pb_vector_subdir)/layer_${layer}.pt"
        local raw_loss_file="${run_dir}/train_losses.csv"
        local loss_file="${run_dir}/losses.csv"
        local train_hparam
        train_hparam="$(pb_train_hparam)"

        mkdir -p "${run_dir}"
        echo "========================================"
        echo "model=${PB_MODEL_NAME} method=${PB_METHOD} dataset=${dataset_name} length=${length} layer=${layer} rank=${rank}"
        echo "seed=${PB_SEED} lr=${PB_LR} epochs=${PB_EPOCHS} batch_size=${PB_BATCH_SIZE} grad_accum=${PB_GRAD_ACCUM}"
        echo "output_dir=${run_dir}"
        if [ -n "${PB_CURRICULUM_BOUNDARIES}" ]; then
          echo "curriculum_boundaries=${PB_CURRICULUM_BOUNDARIES}"
        fi

        if [ "${PB_FORCE}" != "true" ] && { [ -f "${loss_file}" ] || [ -f "${vector_file}" ]; }; then
          echo "warning: existing outputs detected in ${run_dir}; skipping; set FORCE=true to rerun"
          continue
        fi

        {
          echo "========================================"
          echo "model=${PB_MODEL_NAME}"
          echo "method=${PB_METHOD}"
          echo "dataset=${dataset_name}"
          echo "subset=${PB_SUBSET}"
          echo "length=${length}"
          echo "layer=${layer}"
          echo "rank=${rank}"
          echo "seed=${PB_SEED}"
          echo "lr=${PB_LR}"
          echo "epochs=${PB_EPOCHS}"
          echo "batch_size=${PB_BATCH_SIZE}"
          echo "grad_accum=${PB_GRAD_ACCUM}"
          echo "output_dir=${run_dir}"
          echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
        } >> "${log_file}"

        local cmd=(
          "${PB_PYTHON_BIN}" vectors_generate.py
          "model_name_or_path=${PB_MODEL_PATH}"
          "seed=${PB_SEED}"
          "steer_train_dataset=[${dataset_name}]"
          "steer_vector_output_dirs=[${run_dir}]"
          "steer_train_hparam_paths=[${train_hparam}]"
          "+loss_output_dir=${run_dir}"
          "+layers=[${layer}]"
          "+n_epochs=${PB_EPOCHS}"
          "+batch_size=${PB_BATCH_SIZE}"
          "+gradient_accumulation_steps=${PB_GRAD_ACCUM}"
          "+lr=${PB_LR}"
          "+low_rank_dimension=${rank}"
          "+output_length=${PB_OUTPUT_LENGTH}"
        )

        if [ "${PB_METHOD}" = "memft_ot" ]; then
          cmd+=(
            "+use_memft=true"
            "+memft_method=only_threshold"
            "+memft_threshold=0.5"
            "+use_position_weight=false"
          )
        elif [ "${PB_METHOD}" = "memft_sw" ]; then
          cmd+=(
            "+use_memft=true"
            "+memft_method=memft_curriculum"
            "+memft_threshold=0.5"
            "+memft_zero_bp=true"
            "+curriculum_enabled=true"
            "+curriculum_type=prefix_ratio"
            "+curriculum_ratios=[0.2,0.4,0.6,0.8,1.0]"
            "+curriculum_epoch_boundaries=${PB_CURRICULUM_BOUNDARIES}"
            "+curriculum_shuffle_once=true"
            "+curriculum_drop_last=true"
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

        CUDA_VISIBLE_DEVICES="${PB_GPU_ID}" "${cmd[@]}" >> "${log_file}" 2>&1 &
        PB_CURRENT_PID="$!"
        wait "${PB_CURRENT_PID}"
        local status=$?
        PB_CURRENT_PID=""

        if [ "${status}" -ne 0 ]; then
          echo "error: train failed for model=${PB_MODEL_NAME} method=${PB_METHOD} length=${length} layer=${layer} rank=${rank}; see ${log_file}" >&2
          exit "${status}"
        fi

        if [ -f "${raw_loss_file}" ]; then
          mv -f "${raw_loss_file}" "${loss_file}"
        fi
        echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')" >> "${log_file}"
      done
    done
  done
}

pb_run_apply() {
  local model_family="$1"
  cd "${PB_REPO_ROOT}"
  pb_init_model "${model_family}"
  pb_validate_method

  local lengths ranks layers lora_subdir apply_hparam
  pb_parse_list "$(pb_default_lengths)"; lengths=("${PB_PARSED_LIST[@]}")
  pb_parse_list "$(pb_default_ranks)"; ranks=("${PB_PARSED_LIST[@]}")
  pb_parse_list "$(pb_default_layers)"; layers=("${PB_PARSED_LIST[@]}")
  lora_subdir="$(pb_vector_subdir)"
  apply_hparam="$(pb_apply_hparam)"

  echo "PhoneBook apply entry"
  echo "model=${PB_MODEL_NAME} model_family=${PB_MODEL_FAMILY} method=${PB_METHOD} subset=${PB_SUBSET} seed=${PB_SEED}"
  echo "lengths=${lengths[*]} ranks=${ranks[*]} layers=${layers[*]} gpu_id=${PB_GPU_ID}"

  for length in "${lengths[@]}"; do
    local train_dataset="pb_${PB_SUBSET}_${PB_MODEL_FAMILY}_${length}"
    local generation_dataset="${train_dataset}_apply"
    if ! pb_dataset_configured "${generation_dataset}"; then
      echo "error: missing generation dataset ${generation_dataset} in ${PB_DATASET_FORMAT}" >&2
      exit 2
    fi

    for layer in "${layers[@]}"; do
      for rank in "${ranks[@]}"; do
        local run_dir="vectors/pb/${PB_METHOD}/${PB_MODEL_NAME}/${PB_SUBSET}/length_${length}/layer_${layer}/rank_${rank}"
        local lora_dir="${run_dir}/${train_dataset}/${lora_subdir}"
        local lora_file="${lora_dir}/layer_${layer}.pt"
        local output_dir="generation/pb/${PB_METHOD}/${PB_MODEL_NAME}/${PB_SUBSET}/length_${length}/layer_${layer}/rank_${rank}"
        local log_file="${output_dir}/apply.log"

        echo "========================================"
        echo "model=${PB_MODEL_NAME} method=${PB_METHOD} dataset=${generation_dataset} length=${length} layer=${layer} rank=${rank}"
        echo "seed=${PB_SEED} output_dir=${output_dir}"
        echo "vector_dir=${lora_dir}"

        if [ ! -f "${lora_file}" ]; then
          echo "warning: missing LoRA file ${lora_file}; skipping"
          continue
        fi

        mkdir -p "${output_dir}"
        {
          echo "========================================"
          echo "model=${PB_MODEL_NAME}"
          echo "method=${PB_METHOD}"
          echo "dataset=${generation_dataset}"
          echo "subset=${PB_SUBSET}"
          echo "length=${length}"
          echo "layer=${layer}"
          echo "rank=${rank}"
          echo "seed=${PB_SEED}"
          echo "lora_dir=${lora_dir}"
          echo "output_dir=${output_dir}"
          echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
        } >> "${log_file}"

        CUDA_VISIBLE_DEVICES="${PB_GPU_ID}" "${PB_PYTHON_BIN}" vectors_apply.py \
          model_name_or_path="${PB_MODEL_PATH}" \
          use_chat_template=true \
          apply_steer_hparam_paths="[${apply_hparam}]" \
          steer_vector_load_dir="[${lora_dir}]" \
          generation_data="[${generation_dataset}]" \
          generation_data_size="${PB_GENERATION_DATA_SIZE}" \
          generation_output_dir="${output_dir}" \
          generate_orig_output=true \
          generation_params.do_sample=false \
          generation_params.max_new_tokens=64 \
          +layers="[${layer}]" \
          +intervention_method=lora \
          +concept_id=0 \
          hydra.run.dir=. \
          hydra.output_subdir=null \
          hydra.job.chdir=false >> "${log_file}" 2>&1 &

        PB_CURRENT_PID="$!"
        wait "${PB_CURRENT_PID}"
        local status=$?
        PB_CURRENT_PID=""

        if [ "${status}" -ne 0 ]; then
          echo "warning: apply failed for model=${PB_MODEL_NAME} method=${PB_METHOD} length=${length} layer=${layer} rank=${rank}; see ${log_file}"
        else
          echo "end_time=$(date '+%Y-%m-%d %H:%M:%S')" >> "${log_file}"
        fi
      done
    done
  done
}

pb_run_final_loss() {
  local model_family="$1"
  cd "${PB_REPO_ROOT}"
  pb_init_model "${model_family}"
  pb_validate_method

  local lengths ranks layers lora_subdir debug_used
  pb_parse_list "$(pb_default_lengths)"; lengths=("${PB_PARSED_LIST[@]}")
  pb_parse_list "$(pb_default_ranks)"; ranks=("${PB_PARSED_LIST[@]}")
  pb_parse_list "$(pb_default_layers)"; layers=("${PB_PARSED_LIST[@]}")
  lora_subdir="$(pb_vector_subdir)"
  debug_used=0

  echo "PhoneBook final-loss entry"
  echo "model=${PB_MODEL_NAME} model_family=${PB_MODEL_FAMILY} method=${PB_METHOD} subset=${PB_SUBSET} seed=${PB_SEED}"
  echo "lengths=${lengths[*]} ranks=${ranks[*]} layers=${layers[*]} gpu_id=${PB_GPU_ID}"
  echo "target_only_loss=${PB_TARGET_ONLY_LOSS}"

  for length in "${lengths[@]}"; do
    local train_dataset="pb_${PB_SUBSET}_${PB_MODEL_FAMILY}_${length}"
    if ! pb_dataset_configured "${train_dataset}"; then
      echo "error: missing train dataset ${train_dataset} in ${PB_DATASET_FORMAT}" >&2
      exit 2
    fi

    for layer in "${layers[@]}"; do
      for rank in "${ranks[@]}"; do
        local run_dir="vectors/pb/${PB_METHOD}/${PB_MODEL_NAME}/${PB_SUBSET}/length_${length}/layer_${layer}/rank_${rank}"
        local init_vector_path="${run_dir}/${train_dataset}/${lora_subdir}"
        local lora_file="${init_vector_path}/layer_${layer}.pt"
        local output_dir="generation/pb/${PB_METHOD}/${PB_MODEL_NAME}/${PB_SUBSET}/length_${length}/layer_${layer}/rank_${rank}/final_loss"
        local tmp_vector_dir="${output_dir}/tmp_vectors"
        local log_file="${output_dir}/final_loss.log"
        local loss_csv="${output_dir}/train_losses.csv"
        local target_only_debug=false

        echo "========================================"
        echo "model=${PB_MODEL_NAME} method=${PB_METHOD} dataset=${train_dataset} length=${length} layer=${layer} rank=${rank}"
        echo "seed=${PB_SEED} output_dir=${output_dir}"
        echo "init_vector_path=${init_vector_path}"

        if [ ! -f "${lora_file}" ]; then
          echo "warning: missing LoRA file ${lora_file}; skipping"
          continue
        fi

        if [ "${PB_TARGET_ONLY_DEBUG_FIRST}" = "true" ] && [ "${debug_used}" -eq 0 ]; then
          target_only_debug=true
          debug_used=1
        fi

        mkdir -p "${output_dir}"
        {
          echo "========================================"
          echo "model=${PB_MODEL_NAME}"
          echo "method=${PB_METHOD}"
          echo "dataset=${train_dataset}"
          echo "subset=${PB_SUBSET}"
          echo "length=${length}"
          echo "layer=${layer}"
          echo "rank=${rank}"
          echo "seed=${PB_SEED}"
          echo "init_vector_path=${init_vector_path}"
          echo "output_dir=${output_dir}"
          echo "target_only_loss=${PB_TARGET_ONLY_LOSS}"
          echo "target_only_debug=${target_only_debug}"
          echo "start_time=$(date '+%Y-%m-%d %H:%M:%S')"
        } >> "${log_file}"

        CUDA_VISIBLE_DEVICES="${PB_GPU_ID}" "${PB_PYTHON_BIN}" vectors_generate.py \
          model_name_or_path="${PB_MODEL_PATH}" \
          use_chat_template=true \
          steer_train_hparam_paths="[${PB_SFT_HPARAM}]" \
          steer_train_dataset="[${train_dataset}]" \
          steer_vector_output_dirs="[${tmp_vector_dir}]" \
          save_vectors=false \
          +loss_output_dir="${output_dir}" \
          +layers="[${layer}]" \
          +n_epochs=1 \
          +batch_size=1 \
          +gradient_accumulation_steps=1 \
          +inference=true \
          +target_only_loss="${PB_TARGET_ONLY_LOSS}" \
          +target_only_debug="${target_only_debug}" \
          +target_only_raw_target_field=target \
          +init_vector_path="${init_vector_path}" \
          +intervention_method=lora \
          +intervention_components=mlp \
          +low_rank_dimension="${rank}" \
          +output_length="${PB_OUTPUT_LENGTH}" \
          +dropout=0.0 \
          +intervention_positions_dropout=0.0 \
          hydra.run.dir=. \
          hydra.output_subdir=null \
          hydra.job.chdir=false >> "${log_file}" 2>&1 &

        PB_CURRENT_PID="$!"
        wait "${PB_CURRENT_PID}"
        local status=$?
        PB_CURRENT_PID=""

        if [ "${status}" -ne 0 ]; then
          echo "warning: final-loss failed for model=${PB_MODEL_NAME} method=${PB_METHOD} length=${length} layer=${layer} rank=${rank}; see ${log_file}"
          continue
        fi

        if [ -f "${loss_csv}" ]; then
          if "${PB_PYTHON_BIN}" "${PB_SUMMARY_SCRIPT}" \
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
}

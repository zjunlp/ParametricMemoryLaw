#!/usr/bin/env python
"""Inspect PhoneBook prompt/chat-template/BOS handling without training or applying."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch
from transformers import AutoTokenizer


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from steer.datasets.dataset_loader import DatasetLoader
from steer.trainer.utils.data_utils import (
    build_target_only_labels,
    preprocess_preference_data,
)
from steer.utils.templates import build_model_input
from steer.vector_generators.sft.utils import get_prefix_length, prepare_groups


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Debug real PhoneBook train/apply/final_loss prompt construction."
    )
    parser.add_argument("--qwen-tokenizer", default=None)
    parser.add_argument("--llama-tokenizer", default=None)
    parser.add_argument("--qwen-dataset", default="pb_random_qwen_1000")
    parser.add_argument("--llama-dataset", default="pb_random_llama_1000")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--system-prompt", default="")
    return parser.parse_args()


def first_token_details(tokenizer, input_ids, limit=20):
    ids = input_ids[:limit].tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids[:limit])
    tokens = tokenizer.convert_ids_to_tokens(ids)
    return ids, tokens


def decode_masked(tokenizer, input_ids, labels):
    mask = labels.ne(-100)
    if not bool(mask.any()):
        return "", 0
    supervised_ids = input_ids[mask]
    return tokenizer.decode(supervised_ids.tolist(), skip_special_tokens=False), int(mask.sum().item())


def print_block(model_label, stage, raw_item, prompt, target, tokenizer, input_ids, labels=None, prompt_length=None):
    first_ids, first_tokens = first_token_details(tokenizer, input_ids)
    bos_id = tokenizer.bos_token_id
    bos_count = int((input_ids == bos_id).sum().item()) if bos_id is not None else 0

    print("=" * 100)
    print(f"model: {model_label}")
    print(f"stage: {stage}")
    print(f"raw query: {raw_item.get('query', raw_item.get('question', raw_item.get('input')))}")
    print(f"raw target: {raw_item.get('target', raw_item.get('matching', raw_item.get('reference_response')))}")
    print(f"bos_token: {tokenizer.bos_token!r}")
    print(f"bos_token_id: {tokenizer.bos_token_id!r}")
    print(f"BOS count in input_ids: {bos_count}")
    print(f"prompt_length: {prompt_length}")
    print(f"first 20 input_ids: {first_ids}")
    print(f"first 20 decoded tokens: {first_tokens}")
    print("decoded prompt:")
    print(prompt)

    if labels is not None:
        supervised_text, supervised_count = decode_masked(tokenizer, input_ids, labels)
        print(f"labels non--100 token count: {supervised_count}")
        print("labels non--100 decode:")
        print(supervised_text)
    else:
        print("labels non--100 token count: n/a")
        print("labels non--100 decode: n/a")
    print(f"raw target used by stage: {target!r}")


def inspect_train_or_final(model_label, tokenizer, dataset_name, sample_index, stage, max_length):
    loader = DatasetLoader()
    dataset = loader.load_file(dataset_name, split="train")
    raw_item = dataset[sample_index]
    tokenizer.model_max_length = max_length

    prefix_length = get_prefix_length(tokenizer)
    prepared = prepare_groups(
        [raw_item],
        concept=dataset_name,
        tokenizer=tokenizer,
        use_chat_template=True,
        model_name_or_path=tokenizer.name_or_path,
        max_num_of_examples=None,
        steering_prompt_type="blend_in",
        is_select_category=False,
    )
    item = prepared[0]
    target_only = stage == "final_loss"
    data = preprocess_preference_data(
        tokenizer=tokenizer,
        prompt=item["question"],
        winning_output=item["matching"],
        losing_output=item["not_matching"],
        positions="all",
        prefix_length=prefix_length,
        prefix_tuning=False,
        target_only_loss=target_only,
        winning_raw_target=item.get("target"),
    )

    labels = data["winning_labels"]
    if target_only:
        labels = data.get("winning_target_only_labels", labels)

    print_block(
        model_label=model_label,
        stage=stage,
        raw_item=raw_item,
        prompt=item["question"],
        target=item["matching"],
        tokenizer=tokenizer,
        input_ids=data["winning_input_ids"],
        labels=labels,
        prompt_length=int(data["prompt_lengths"].item()) + 1,
    )


def inspect_apply(model_label, tokenizer, dataset_name, sample_index, max_length, system_prompt, use_chat_template):
    loader = DatasetLoader()
    dataset = loader.load_file(f"{dataset_name}_apply", split="generation")
    raw_item = dataset[sample_index]
    tokenizer.model_max_length = max_length

    prompt = build_model_input(
        raw_item["input"],
        tokenizer,
        system_prompt=system_prompt,
        use_chat_template=use_chat_template,
    )
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        add_special_tokens=not use_chat_template,
        truncation=True,
        max_length=max_length,
    )
    input_ids = encoded["input_ids"][0]
    print_block(
        model_label=model_label,
        stage="apply",
        raw_item=raw_item,
        prompt=prompt,
        target=raw_item.get("reference_response"),
        tokenizer=tokenizer,
        input_ids=input_ids,
        labels=None,
        prompt_length=len(input_ids),
    )


def inspect_model(model_label, tokenizer_path, dataset_name, args):
    if not tokenizer_path:
        print(f"[skip] {model_label}: tokenizer path not provided")
        return

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"tokenizer.name_or_path: {tokenizer.name_or_path}")
    print(f"chat_template startswith BOS literal: {str(tokenizer.chat_template).lstrip().startswith(tokenizer.bos_token or '')}")
    for stage in ("train", "apply", "final_loss"):
        if stage == "apply":
            inspect_apply(
                model_label,
                tokenizer,
                dataset_name,
                args.sample_index,
                args.max_length,
                args.system_prompt,
                args.use_chat_template,
            )
        else:
            inspect_train_or_final(
                model_label,
                tokenizer,
                dataset_name,
                args.sample_index,
                stage,
                args.max_length,
            )


def main() -> None:
    args = parse_args()
    inspect_model("qwen", args.qwen_tokenizer, args.qwen_dataset, args)
    inspect_model("llama", args.llama_tokenizer, args.llama_dataset, args)


if __name__ == "__main__":
    main()

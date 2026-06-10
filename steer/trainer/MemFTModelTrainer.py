from .BaseModelTrainer import ModelTrainer
import torch, random
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Dict, Tuple
from torch.utils.data import DataLoader, Subset
from .utils.model_utils import get_lr
import numpy as np
from transformers import get_scheduler
from .utils.data_utils import make_preference_data_module
import csv
import os


class MemFTModelTrainer(ModelTrainer):
    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'MemFTModelTrainer'

    def _is_curriculum_enabled(self):
        return bool(
            getattr(self.hparams, "use_memft", False)
            and getattr(self.hparams, "memft_method", "only_threshold") == "memft_curriculum"
            and getattr(self.hparams, "curriculum_enabled", False)
            and not getattr(self.hparams, "inference", False)
        )

    def _get_curriculum_ratio(self, epoch):
        ratios = list(getattr(self.hparams, "curriculum_ratios", [0.2, 0.4, 0.6, 0.8, 1.0]))
        boundaries = list(getattr(self.hparams, "curriculum_epoch_boundaries", [20, 40, 60, 80, 200]))
        if len(ratios) != len(boundaries):
            raise ValueError("curriculum_ratios and curriculum_epoch_boundaries must have the same length")
        for ratio, boundary in zip(ratios, boundaries):
            if epoch < boundary:
                return float(ratio)
        return float(ratios[-1])

    def _get_curriculum_order(self, total_samples):
        if getattr(self, "_curriculum_order", None) is not None and len(self._curriculum_order) == total_samples:
            return self._curriculum_order

        if getattr(self.hparams, "curriculum_shuffle_once", True):
            g = torch.Generator()
            g.manual_seed(self.hparams.seed)
            self._curriculum_order = torch.randperm(total_samples, generator=g).tolist()
        else:
            self._curriculum_order = list(range(total_samples))
        return self._curriculum_order

    def _get_curriculum_active_indices(self, epoch, total_samples):
        if getattr(self.hparams, "curriculum_type", "prefix_ratio") != "prefix_ratio":
            raise ValueError("memft_curriculum currently supports only curriculum_type='prefix_ratio'")
        batch_size = int(self.hparams.batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be positive for memft_curriculum")

        ratio = self._get_curriculum_ratio(epoch)
        drop_last = bool(getattr(self.hparams, "curriculum_drop_last", True))
        complete_total_samples = (total_samples // batch_size) * batch_size if drop_last else total_samples
        if complete_total_samples < batch_size:
            raise ValueError(
                f"memft_curriculum needs at least one full batch, got total_samples={total_samples}, batch_size={batch_size}"
            )

        active_before_rounding = int(total_samples * ratio)
        active_samples = (active_before_rounding // batch_size) * batch_size
        active_samples = max(batch_size, active_samples)
        active_samples = min(complete_total_samples, active_samples)
        active_batches = active_samples // batch_size

        order = self._get_curriculum_order(total_samples)
        indices = order[:active_samples]
        info = {
            "curriculum_ratio": ratio,
            "total_samples": total_samples,
            "batch_size": batch_size,
            "active_samples_before_rounding": active_before_rounding,
            "active_samples": active_samples,
            "active_batches": active_batches,
            "curriculum_drop_last": drop_last,
            "complete_total_samples": complete_total_samples,
        }
        return indices, info

    def _make_curriculum_dataloader(self, data_module, epoch):
        train_dataset = data_module["train_dataset"]
        indices, info = self._get_curriculum_active_indices(epoch, len(train_dataset))
        curriculum_dataset = Subset(train_dataset, indices)
        train_dataloader = DataLoader(
            curriculum_dataset,
            shuffle=False,
            batch_size=self.hparams.batch_size,
            collate_fn=data_module["data_collator"],
            drop_last=bool(getattr(self.hparams, "curriculum_drop_last", True)),
        )
        return train_dataloader, info

    def _get_per_token_losses(self, logits, labels):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        flat_shift_labels = shift_labels.view(-1)
        loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
        per_token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
        valid_mask = (flat_shift_labels != -100).float()
        return shift_logits, shift_labels, per_token_losses, valid_mask

    def _apply_position_weight(self, token_weights, pos_minibatch_inputs, hard_weight):
        batch_size_actual = pos_minibatch_inputs["input_ids"].shape[0]
        seq_len_actual = pos_minibatch_inputs["input_ids"].shape[1]
        shifted_seq_len = seq_len_actual - 1

        if shifted_seq_len <= 0 or token_weights.sum() <= 0:
            return token_weights

        hard_mask = torch.abs(token_weights - hard_weight) < 1e-6
        hard_mask_2d = hard_mask.view(batch_size_actual, shifted_seq_len)
        hard_ranks_2d = torch.cumsum(hard_mask_2d, dim=1).float() - 1.0
        num_hard_per_sample = hard_mask_2d.sum(dim=1)
        max_ranks_per_sample = (num_hard_per_sample - 1).clamp(min=1)
        max_ranks_expanded = max_ranks_per_sample.unsqueeze(1).expand(-1, shifted_seq_len)
        norm_ranks_2d = torch.clamp(hard_ranks_2d / max_ranks_expanded.float(), 0.0, 1.0)

        position_decay_lambda = getattr(self.hparams, "position_decay_lambda", 2.0)
        pos_weights_2d = torch.exp(-position_decay_lambda * norm_ranks_2d)
        pos_weights_flat = pos_weights_2d.contiguous().view(-1)
        token_weights[hard_mask] = token_weights[hard_mask] * pos_weights_flat[hard_mask]
        return token_weights

    def _compute_only_threshold_loss(self, per_token_losses, valid_mask, pos_steer_loss, pos_minibatch_inputs):
        prob_threshold = float(getattr(self.hparams, "memft_threshold", 0.5))
        loss_threshold = -np.log(prob_threshold)
        easy_token_weight = 0.0
        hard_token_weight = 1.0

        token_weights = torch.where(
            per_token_losses > loss_threshold,
            torch.full_like(per_token_losses, hard_token_weight),
            torch.full_like(per_token_losses, easy_token_weight),
        )
        token_weights = token_weights * valid_mask

        use_position_weight = bool(getattr(self.hparams, "use_position_weight", False))
        if use_position_weight:
            token_weights = self._apply_position_weight(
                token_weights, pos_minibatch_inputs, hard_token_weight
            )

        total_weight = token_weights.sum()
        if total_weight > 0:
            bp_loss = (per_token_losses * token_weights).sum() / total_weight
        else:
            bp_loss = pos_steer_loss

        supervised_tokens = int(valid_mask.sum().item())
        active_tokens = int((token_weights.detach() > 0).sum().item())
        masked_tokens = max(supervised_tokens - active_tokens, 0)
        return bp_loss, {
            "memft_method": "only_threshold",
            "memft_threshold": prob_threshold,
            "memft_masked_tokens": masked_tokens,
            "memft_active_tokens": active_tokens,
            "memft_mask_ratio": masked_tokens / max(supervised_tokens, 1),
            "use_position_weight": use_position_weight,
            "token_weights": token_weights.detach(),
            "loss_threshold": loss_threshold,
        }

    def _compute_only_threshold_zero_bp_loss(self, per_token_losses, valid_mask, pos_steer_loss, pos_minibatch_inputs):
        prob_threshold = float(getattr(self.hparams, "memft_threshold", 0.5))
        loss_threshold = -np.log(prob_threshold)
        easy_token_weight = 0.0
        hard_token_weight = 1.0

        token_weights = torch.where(
            per_token_losses > loss_threshold,
            torch.full_like(per_token_losses, hard_token_weight),
            torch.full_like(per_token_losses, easy_token_weight),
        )
        token_weights = token_weights * valid_mask

        use_position_weight = bool(getattr(self.hparams, "use_position_weight", False))
        if use_position_weight:
            token_weights = self._apply_position_weight(
                token_weights, pos_minibatch_inputs, hard_token_weight
            )

        total_weight = token_weights.sum()
        if total_weight > 0:
            bp_loss = (per_token_losses * token_weights).sum() / total_weight
            zero_bp = False
        else:
            bp_loss = torch.tensor(0.0, device=self.model.device, requires_grad=True)
            zero_bp = True

        supervised_tokens = int(valid_mask.sum().item())
        active_tokens = int((token_weights.detach() > 0).sum().item())
        masked_tokens = max(supervised_tokens - active_tokens, 0)
        return bp_loss, {
            "memft_method": "only_threshold_zero_bp",
            "memft_threshold": prob_threshold,
            "memft_masked_tokens": masked_tokens,
            "memft_active_tokens": active_tokens,
            "memft_mask_ratio": masked_tokens / max(supervised_tokens, 1),
            "use_position_weight": use_position_weight,
            "token_weights": token_weights.detach(),
            "loss_threshold": loss_threshold,
            "zero_bp": zero_bp,
        }

    def _compute_sliding_loss(self, shift_logits, shift_labels, per_token_losses, valid_mask, pos_steer_loss, labels):
        detached_losses = per_token_losses.detach()
        with torch.no_grad():
            predictions = shift_logits.argmax(dim=-1)
            is_wrong = (predictions != shift_labels) & (shift_labels != -100)

        batch_size = labels.shape[0]
        seq_len_shifted = labels.shape[1] - 1
        device = labels.device
        is_wrong_float = is_wrong.float()
        has_error = is_wrong_float.sum(dim=1) > 0
        current_anchors = torch.argmax(is_wrong_float, dim=1)

        prob_threshold = float(getattr(self.hparams, "memft_threshold", 0.5))
        loss_threshold = -np.log(prob_threshold)
        scale = float(getattr(self.hparams, "sliding_scale", 10.0))
        base_token_weights = torch.sigmoid(scale * (detached_losses - loss_threshold)) * valid_mask

        if not hasattr(self, "batch_patience") or self.batch_patience.shape[0] != batch_size:
            self.batch_patience = torch.zeros(batch_size, device=device)
            self.batch_expansion = torch.ones(batch_size, device=device)
            self.last_anchors = torch.full((batch_size,), -1, device=device)

        with torch.no_grad():
            is_stuck = (current_anchors == self.last_anchors) & has_error
            self.batch_patience = torch.where(
                is_stuck, self.batch_patience + 1, torch.zeros_like(self.batch_patience)
            )
            should_expand = self.batch_patience >= 3
            self.batch_expansion = torch.where(
                should_expand,
                torch.clamp(self.batch_expansion + 0.5, max=4.0),
                torch.ones_like(self.batch_expansion),
            )
            self.last_anchors = torch.where(
                has_error, current_anchors, torch.full_like(current_anchors, -1)
            )

        use_sliding_window = bool(getattr(self.hparams, "use_sliding_window", True))
        base_window_size = int(getattr(self.hparams, "memory_window_size", 50))
        sliding_tau = float(getattr(self.hparams, "sliding_tau", 20.0))
        base_floor = float(getattr(self.hparams, "sliding_base_floor", 0.01))

        if use_sliding_window:
            active_windows = (base_window_size * self.batch_expansion).unsqueeze(1)
            pos_range = torch.arange(seq_len_shifted, device=device).unsqueeze(0).expand(batch_size, -1)
            anchor_pos = current_anchors.unsqueeze(1)
            rel_pos = (pos_range - anchor_pos).float()
            decay_curve = torch.exp(-rel_pos.clamp(min=0) / sliding_tau)
            in_window_mask = (rel_pos < active_windows).float()
            final_mask = torch.where(
                in_window_mask > 0, decay_curve, torch.full_like(decay_curve, base_floor)
            )
            final_decay_mask = torch.where(has_error.unsqueeze(1), final_mask, torch.ones_like(final_mask))
            token_weights = (base_token_weights.view(batch_size, seq_len_shifted) * final_decay_mask).view(-1)
        else:
            active_windows = torch.full((batch_size, 1), base_window_size, device=device)
            token_weights = base_token_weights

        total_weight = token_weights.sum()
        if total_weight > 1e-5:
            bp_loss = (per_token_losses * token_weights).sum() / total_weight
        else:
            bp_loss = pos_steer_loss

        supervised_tokens = int(valid_mask.sum().item())
        active_tokens = int((token_weights.detach() > (base_floor + 0.01)).sum().item())
        masked_tokens = max(supervised_tokens - active_tokens, 0)
        return bp_loss, {
            "memft_method": "sliding",
            "memft_threshold": prob_threshold,
            "memft_masked_tokens": masked_tokens,
            "memft_active_tokens": active_tokens,
            "memft_mask_ratio": masked_tokens / max(supervised_tokens, 1),
            "use_position_weight": bool(getattr(self.hparams, "use_position_weight", False)),
            "token_weights": token_weights.detach(),
            "loss_threshold": loss_threshold,
            "avg_patience": self.batch_patience.mean().item(),
            "avg_expansion": self.batch_expansion.mean().item(),
            "anchors": current_anchors.detach().cpu().tolist(),
            "window_ends": (current_anchors + active_windows.squeeze().long()).detach().cpu().tolist(),
        }

    def _print_memft_debug(self, metrics, per_token_losses, valid_mask):
        if not getattr(self.hparams, "memft_debug", False):
            return
        if getattr(self, "_memft_debug_printed", False):
            return

        supervised_tokens = int(valid_mask.sum().item())
        masked_probs = []
        token_weights = metrics.get("token_weights")
        if token_weights is not None:
            masked_positions = ((token_weights <= 0) & (valid_mask > 0)).nonzero(as_tuple=False).flatten()
            if masked_positions.numel() > 0:
                sample_losses = per_token_losses.detach()[masked_positions[:5]]
                masked_probs = torch.exp(-sample_losses).detach().cpu().tolist()

        print(
            "[memft_debug] "
            f"method={metrics['memft_method']} "
            f"threshold={metrics['memft_threshold']} "
            f"use_position_weight={metrics['use_position_weight']} "
            f"supervised_tokens={supervised_tokens} "
            f"masked_tokens={metrics['memft_masked_tokens']} "
            f"mask_ratio={metrics['memft_mask_ratio']:.4f} "
            f"masked_probs={masked_probs}"
        )
        self._memft_debug_printed = True

    def make_preference_dataloader(self, examples, **kwargs):
        data_module = make_preference_data_module(self.model.tokenizer, examples, **kwargs)
        g = torch.Generator()
        g.manual_seed(self.hparams.seed)
        train_dataloader = DataLoader(
            data_module["train_dataset"], shuffle=True, # we shuffle for examples.
            batch_size=self.hparams.batch_size, 
            collate_fn=data_module["data_collator"],
            generator=g)
        return train_dataloader

    def train(self, examples, **kwargs):

        # prepare the data
        print(kwargs)
        curriculum_enabled = self._is_curriculum_enabled()
        if curriculum_enabled:
            data_module = make_preference_data_module(self.model.tokenizer, examples, **kwargs)
            train_dataloader = None
        else:
            data_module = None
            train_dataloader = self.make_preference_dataloader(
                examples, **kwargs)
    
        torch.cuda.empty_cache()

        # prepare the optimizer and learning rate scheduler
        optimizer = torch.optim.AdamW(
            self.model.steer_vector.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        print(optimizer.param_groups) 

        if curriculum_enabled:
            num_training_steps = 0
            for epoch in range(self.hparams.n_epochs):
                _, curriculum_info = self._get_curriculum_active_indices(
                    epoch, len(data_module["train_dataset"])
                )
                num_training_steps += curriculum_info["active_batches"] // self.hparams.gradient_accumulation_steps
        else:
            num_training_steps = self.hparams.n_epochs * (len(train_dataloader) // self.hparams.gradient_accumulation_steps)
        lr_scheduler = get_scheduler(
            "linear", 
            optimizer=optimizer,
            num_warmup_steps=0, 
            num_training_steps=num_training_steps
        )
        

        # training loop
        progress_bar, curr_step, logging_step = tqdm(range(num_training_steps), leave=True), 0, 0
        
        for epoch in range(self.hparams.n_epochs):
            curriculum_zero_bp_batches = 0
            curriculum_active_tokens = 0
            curriculum_masked_tokens = 0
            if curriculum_enabled:
                train_dataloader, curriculum_info = self._make_curriculum_dataloader(data_module, epoch)
                if getattr(self.hparams, "memft_debug", False):
                    print(
                        "[memft_curriculum_debug] "
                        f"epoch={epoch} "
                        f"ratio={curriculum_info['curriculum_ratio']} "
                        f"total_samples={curriculum_info['total_samples']} "
                        f"batch_size={curriculum_info['batch_size']} "
                        f"active_before_rounding={curriculum_info['active_samples_before_rounding']} "
                        f"active_samples={curriculum_info['active_samples']} "
                        f"active_batches={curriculum_info['active_batches']} "
                        f"drop_last={curriculum_info['curriculum_drop_last']} "
                        f"complete_total_samples={curriculum_info['complete_total_samples']} "
                        f"threshold={getattr(self.hparams, 'memft_threshold', 0.5)}"
                    )
            for step, batch in enumerate(train_dataloader):
                expanded_batch_size = self.hparams.batch_size * len(self.preference_pairs)
                minibatch_size = self.hparams.batch_size
                num_minibatches = (expanded_batch_size + minibatch_size - 1) // minibatch_size
                
                # prepare the batch data
                winning_inputs = {k: [] for k in ["input_ids", "attention_mask", "labels", "intervention_locations", "steering_factors"]}
                losing_inputs = {k: [] for k in ["input_ids", "attention_mask", "labels", "intervention_locations", "steering_factors"]}
                
                for i in range(self.hparams.batch_size):
                    for pair in self.preference_pairs:
                        # fill the winning and losing inputs

                        winning_inputs["input_ids"].append(batch[f"{pair}_winning_input_ids"][i])
                        winning_inputs["attention_mask"].append(batch[f"{pair}_winning_attention_mask"][i])
                        winning_inputs["labels"].append(batch[f"{pair}_winning_labels"][i])
                        winning_inputs["intervention_locations"].append(batch[f"{pair}_winning_intervention_locations"][i])
                        losing_inputs["input_ids"].append(batch[f"{pair}_losing_input_ids"][i])
                        losing_inputs["attention_mask"].append(batch[f"{pair}_losing_attention_mask"][i])
                        losing_inputs["labels"].append(batch[f"{pair}_losing_labels"][i])
                        losing_inputs["intervention_locations"].append(batch[f"{pair}_losing_intervention_locations"][i])
                        
                        # set the steering factors according to the type of the preference pair
                        if "_add" in pair: 
                            winning_inputs["steering_factors"].append(torch.tensor(random.choice(self.hparams.steering_factors)))
                            losing_inputs["steering_factors"].append(torch.tensor(random.choice(self.hparams.steering_factors)))
                        else: 
                            if self.hparams.substraction_type == "null_it_out": 
                                winning_inputs["steering_factors"].append(torch.tensor(0.0))
                                losing_inputs["steering_factors"].append(torch.tensor(0.0))
                            else: 
                                winning_inputs["steering_factors"].append(torch.tensor(-1.0 * random.choice(self.hparams.steering_factors)))
                                losing_inputs["steering_factors"].append(torch.tensor(-1.0 * random.choice(self.hparams.steering_factors)))
                
                # initialize the variables for accumulating the current batch metrics and loss
                loss_sum = 0
                
                # loop through the minibatches and compute the gradient

                for mb in range(num_minibatches):
                    start_idx = mb * minibatch_size
                    end_idx = min((mb + 1) * minibatch_size, expanded_batch_size)
                    
                    if start_idx >= expanded_batch_size:
                        break
                    
                    # minibatch_inputs = {
                    #     k: torch.stack(winning_inputs[k][start_idx:end_idx] + losing_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                    #     for k, _ in winning_inputs.items()
                    # }

                    if self.hparams.inference:
                        if self.hparams.sft_preference_type == "winning_only":
                            pos_minibatch_inputs = {
                                k: torch.stack(winning_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                                for k, _ in winning_inputs.items()
                            }
                        elif self.hparams.sft_preference_type == "losing_only":
                            pos_minibatch_inputs = {
                                k: torch.stack(losing_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                                for k, _ in losing_inputs.items()
                            }
                    else:
                        pos_minibatch_inputs = {
                            k: torch.stack(winning_inputs[k][start_idx:end_idx], dim=0).to(self.model.device) 
                            for k, _ in winning_inputs.items()
                        }

                    # prepare the intervention subspaces
                    subspaces = [{ "steering_factor": pos_minibatch_inputs["steering_factors"]}]
                    subspace_repeat = 1 if not isinstance(self.model.steer_vector, list) else len(self.model.steer_vector)
                    subspaces = subspaces * subspace_repeat
                    self.model.steer_vector.subspaces = subspaces
                    
                    # model forward propagation, severe bug here
                    self.model.steer_vector.intervention_locations = pos_minibatch_inputs["intervention_locations"]
                    pos_outputs_orig = self.model.model(
                        input_ids=pos_minibatch_inputs["input_ids"],
                        attention_mask=pos_minibatch_inputs["attention_mask"],
                        labels=pos_minibatch_inputs["labels"],
                        use_cache=False
                    )

                    # calculate the reference model output ref_outputs
                    if hasattr(self.model, "steer_vector"):
                        # remove the intervention
                        self.model.reset(self.hparams.alg_name)
                        
                        # forward propagation without intervention
                        pos_ref_outputs = self.model.model(
                            input_ids=pos_minibatch_inputs["input_ids"],
                            attention_mask=pos_minibatch_inputs["attention_mask"],
                            labels=pos_minibatch_inputs["labels"],
                            use_cache=False
                        )

                        for layer in self.layers:
                            self.model.set_intervention(layer, self.model.steer_vector, self.hparams.alg_name)
                    else:
                        pos_ref_outputs = pos_outputs_orig

                    pos_steer_loss = pos_outputs_orig.loss
                    pos_ref_loss = pos_ref_outputs.loss
                    bp_loss = pos_steer_loss
                    memft_metrics = {
                        "memft_method": "disabled",
                        "memft_threshold": float(getattr(self.hparams, "memft_threshold", 0.5)),
                        "memft_masked_tokens": 0,
                        "memft_active_tokens": pos_minibatch_inputs["labels"].ne(-100).sum().item(),
                        "memft_mask_ratio": 0.0,
                        "use_position_weight": bool(getattr(self.hparams, "use_position_weight", False)),
                        "zero_bp": False,
                    }

                    memft_enabled = bool(getattr(self.hparams, "use_memft", False)) and not self.hparams.inference
                    if memft_enabled:
                        logits = pos_outputs_orig.logits
                        labels = pos_minibatch_inputs["labels"]
                        shift_logits, shift_labels, per_token_losses, valid_mask = self._get_per_token_losses(
                            logits, labels
                        )
                        memft_method = getattr(self.hparams, "memft_method", "only_threshold")

                        if memft_method == "only_threshold":
                            bp_loss, memft_metrics = self._compute_only_threshold_loss(
                                per_token_losses, valid_mask, pos_steer_loss, pos_minibatch_inputs
                            )
                        elif memft_method == "only_threshold_zero_bp":
                            bp_loss, memft_metrics = self._compute_only_threshold_zero_bp_loss(
                                per_token_losses, valid_mask, pos_steer_loss, pos_minibatch_inputs
                            )
                        elif memft_method == "memft_curriculum":
                            if not getattr(self.hparams, "memft_zero_bp", True):
                                raise ValueError("memft_curriculum requires memft_zero_bp=true")
                            bp_loss, memft_metrics = self._compute_only_threshold_zero_bp_loss(
                                per_token_losses, valid_mask, pos_steer_loss, pos_minibatch_inputs
                            )
                            memft_metrics["memft_method"] = "memft_curriculum"
                        elif memft_method == "sliding":
                            bp_loss, memft_metrics = self._compute_sliding_loss(
                                shift_logits,
                                shift_labels,
                                per_token_losses,
                                valid_mask,
                                pos_steer_loss,
                                labels,
                            )
                        else:
                            raise ValueError(f"Unsupported memft_method: {memft_method}")

                        self._print_memft_debug(memft_metrics, per_token_losses, valid_mask)
                        if curriculum_enabled:
                            curriculum_zero_bp_batches += int(bool(memft_metrics.get("zero_bp", False)))
                            curriculum_active_tokens += int(memft_metrics.get("memft_active_tokens", 0))
                            curriculum_masked_tokens += int(memft_metrics.get("memft_masked_tokens", 0))

                    # Save steer_loss and ref_loss to CSV
                    if hasattr(self.hparams, "loss_output_dir") and self.hparams.loss_output_dir is not None:
                        self.loss_log_file = os.path.join(self.hparams.loss_output_dir if hasattr(self.hparams, "loss_output_dir") else ".", f"train_losses.csv")

                        # Write header if file does not exist
                        if not os.path.exists(self.loss_log_file):
                            header = [
                                "epoch", "step", "pos_steer_loss", "pos_ref_loss",
                                "pos_steering_factors", "token_nums", "memft_bp_loss",
                                "memft_method", "memft_threshold", "memft_masked_tokens",
                                "memft_active_tokens", "memft_mask_ratio", "use_position_weight"
                            ]
                            
                            with open(self.loss_log_file, "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(header)

                        # Build row dynamically based on weight values
                        row = [
                            epoch, 
                            step, 
                            pos_steer_loss.item(),
                            pos_ref_loss.item(),
                            pos_minibatch_inputs["steering_factors"], 
                            pos_minibatch_inputs["labels"].ne(-100).sum().item(),
                            bp_loss.item(),
                            memft_metrics["memft_method"],
                            memft_metrics["memft_threshold"],
                            memft_metrics["memft_masked_tokens"],
                            memft_metrics["memft_active_tokens"],
                            memft_metrics["memft_mask_ratio"],
                            memft_metrics["use_position_weight"],
                        ]

                        with open(self.loss_log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)

                    # Build print message dynamically
                    print_parts = [
                        f"steer_loss: {pos_steer_loss.item():.6f}",
                        f"bp_loss: {bp_loss.item():.6f}",
                        f"memft_method: {memft_metrics['memft_method']}",
                    ]
                    print(" ".join(print_parts))
                    
                    # Use bp_loss for backward
                    minibatch_loss = bp_loss
                    
                    # Normalize loss by total number of minibatches for this step
                    # (instead of dividing by gradient_accumulation_steps)
                    minibatch_loss = minibatch_loss / (num_minibatches * self.hparams.gradient_accumulation_steps)
                    
                    # Backward pass for this minibatch
                    if not self.hparams.inference:
                        minibatch_loss.backward()
                    else:
                        print("inference only, no backward!!!")
                    
                    # Track total loss for logging
                    loss_sum += bp_loss.detach() * (end_idx - start_idx)


                loss = loss_sum / expanded_batch_size
               
                # --- 4.8 optimizer step ---
                if (step + 1) % self.hparams.gradient_accumulation_steps == 0 or (step + 1) == len(train_dataloader):
                    torch.nn.utils.clip_grad_norm_(self.model.steer_vector.parameters(), 1.0)
                    curr_lr = get_lr(optimizer) 
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    

                    progress_bar.update(1)
                    progress_bar.set_description(
                        "lr %.6f || loss %.6f" % (
                            curr_lr, loss))
                    print(f"Epoch {epoch}, Step {step}")

                    curr_step += 1

            if curriculum_enabled and getattr(self.hparams, "memft_debug", False):
                print(
                    "[memft_curriculum_epoch_debug] "
                    f"epoch={epoch} "
                    f"zero_bp_batches={curriculum_zero_bp_batches} "
                    f"active_tokens={curriculum_active_tokens} "
                    f"masked_tokens={curriculum_masked_tokens}"
                )

        progress_bar.close()
        

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations

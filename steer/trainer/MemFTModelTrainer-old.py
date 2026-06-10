from .BaseModelTrainer import ModelTrainer
import torch, random
from tqdm.auto import tqdm
import torch.nn.functional as F
from typing import Dict, Tuple
from torch.utils.data import DataLoader
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

                    # --- START: Token-level Weighted Loss Calculation Modification ---
                    
                    # 1. Get logits and labels for manual calculation
                    logits = pos_outputs_orig.logits
                    labels = pos_minibatch_inputs["labels"]
                    
                    # Shift so that tokens < n predict n (standard LM loss calculation)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    
                    # Flatten the tokens
                    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
                    flat_shift_labels = shift_labels.view(-1)
                    
                    # Compute per-token CrossEntropy Loss (no reduction)
                    loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                    per_token_losses = loss_fct(flat_shift_logits, flat_shift_labels)
                    
                    # Create valid mask (ignore padding tokens labeled as -100)
                    valid_mask = (flat_shift_labels != -100).float()
                    
                    # # --- SANITY CHECK: Verify manual mean loss matches HF loss ---
                    # manual_loss_sum = (per_token_losses * valid_mask).sum()
                    # manual_token_count = valid_mask.sum()
                    
                    # if manual_token_count > 0:
                    #     computed_mean_loss = manual_loss_sum / manual_token_count
                    # else:
                    #     computed_mean_loss = torch.tensor(0.0, device=self.model.device)

                    # print(f"pos_outputs_orig.loss: {pos_outputs_orig.loss.item()}\ncomputed_mean_loss: {computed_mean_loss.item()}")
                    # # Calculate difference for debugging/logging
                    # sanity_diff = abs(computed_mean_loss.item() - pos_outputs_orig.loss.item())
                    # print(f"sanity_diff: {sanity_diff}")

                    
                    # --- Apply Probability Threshold Masking ---
                    # p=0.5 corresponds to loss = -ln(0.5) approx 0.6931
                    PROB_THRESHOLD = 0.5
                    LOSS_THRESHOLD = -np.log(PROB_THRESHOLD) 
                    EASY_TOKEN_WEIGHT = 0.0  # Set to 0.1 if you want small gradient for easy tokens
                    HARD_TOKEN_WEIGHT = 1.0
                    
                    # --- OPTIONAL: Position-based Weighting (Applied ONLY to active tokens) ---
                    USE_POSITION_WEIGHT = self.hparams.use_position_weight  # <--- Set to False to disable
                    POSITION_WEIGHT_FACTOR = 0.5 # <--- Controls decay strength. 0.5 means last token gets 0.5x weight of first.

                    # Create weights: 1.0 if loss > threshold (hard), else EASY_TOKEN_WEIGHT
                    token_weights = torch.where(
                        per_token_losses > LOSS_THRESHOLD, 
                        torch.full_like(per_token_losses, HARD_TOKEN_WEIGHT),
                        torch.full_like(per_token_losses, EASY_TOKEN_WEIGHT)
                    )
                    
                    # Ensure padded tokens have 0 weight
                    token_weights = token_weights * valid_mask
                                      
                    # Calculate Weighted Loss for Backpropagation
                    weighted_loss_numerator = (per_token_losses * token_weights).sum()
                    total_weight = token_weights.sum()
                    
                    if total_weight > 0:
                        # Normalize by sum of weights to maintain gradient scale stability
                        bp_loss = weighted_loss_numerator / total_weight
                    else:
                        bp_loss = pos_outputs_orig.loss

                    # Get original average loss for logging (detached to save memory/graph)
                    original_avg_loss = pos_outputs_orig.loss.detach()
                    pos_ref_loss_val = pos_ref_outputs.loss.detach()
                    
                    # --- END: Token-level Weighted Loss Calculation ---

                    # Save steer_loss and ref_loss to CSV
                    if hasattr(self.hparams, "loss_output_dir") and self.hparams.loss_output_dir is not None:
                        self.loss_log_file = os.path.join(self.hparams.loss_output_dir if hasattr(self.hparams, "loss_output_dir") else ".", f"train_losses.csv")

                        # Write header if file does not exist
                        if not os.path.exists(self.loss_log_file):
                            # Updated Header to include weighted loss and sanity check
                            header = ["epoch", "step", "weighted_bp_loss", "original_avg_loss", "pos_ref_loss", "pos_steering_factors", "token_nums", "LOSS_THRESHOLD", "EASY_TOKEN_WEIGHT", "HARD_TOKEN_WEIGHT", "USE_POSITION_WEIGHT", "POSITION_WEIGHT_FACTOR"]
                            
                            with open(self.loss_log_file, "w", newline="") as f:
                                writer = csv.writer(f)
                                writer.writerow(header)

                        # Build row dynamically based on weight values
                        row = [
                            epoch, 
                            step, 
                            f"{bp_loss.item():.6f}", 
                            f"{original_avg_loss.item():.6f}", 
                            f"{pos_ref_loss_val.item():.6f}", 
                            pos_minibatch_inputs["steering_factors"], 
                            pos_minibatch_inputs["labels"].ne(-100).sum().item(),
                            LOSS_THRESHOLD,
                            EASY_TOKEN_WEIGHT,
                            HARD_TOKEN_WEIGHT,
                            USE_POSITION_WEIGHT,
                            POSITION_WEIGHT_FACTOR
                        ]

                        with open(self.loss_log_file, "a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow(row)

                    # Build print message dynamically
                    print_parts = [
                        f"bp_loss: {bp_loss.item():.6f}", 
                        f"orig_loss: {original_avg_loss.item():.6f}",
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

        progress_bar.close()
        

    def pre_compute_mean_activations(self, dump_dir, **kwargs):
        self.max_activations = {}
        return self.max_activations


from .MergeModelTrainer import MergeModelTrainer
import torch
from tqdm.auto import tqdm
from ..models.interventions import PreferenceMergeLoraIntervention

class MergeLoraTrainer(MergeModelTrainer):

    # the base class for all preference models
    preference_pairs = ["orig_add"] # "orig_add", "orig_sub", "steered_add", "steered_sub"
    def __str__(self):
        return 'MergeLora'

    def make_model(self, **kwargs):
        """
        create a model with intervention
        """
        
        print("**Getting embed dim from the following model config**")
        
        intervention_type = kwargs.get("intervention_type", "addition") # addition
        if intervention_type == "addition":
            # create a preference vector intervention object
            steer_vector = PreferenceMergeLoraIntervention(
                input_dim=kwargs.get("input_dim", self.model.model.config.hidden_size),
                embed_dim=kwargs.get("embed_dim", self.model.model.config.hidden_size), 
                low_rank_dimension=kwargs.get("low_rank_dimension", 4),
                alpha=kwargs["model_params"].lora_alpha,
                torch_dtype=self.model.torch_dtype,
                dropout=kwargs.get("dropout", 0.0),
                intervention_components=kwargs.get("intervention_components", "mlp"),
                merge_num=kwargs.get("merge_num", 1),
                nonlinear=kwargs.get("nonlinear", None),
            )
        else:
            raise ValueError(f"Intervention type {intervention_type} not supported")

        self.intervention_type = intervention_type
        self.model.steer_vector = steer_vector.to(self.model.device, dtype=self.model.torch_dtype)
        self.model.steer_vector.train()
        
        self.preference_pairs = kwargs.get("preference_pairs", ["orig_add"])
        print("self.preference_pairs: ", self.preference_pairs)
        # set the model to eval mode and freeze the parameters
        self.model.model.eval()
        for param in self.model.model.parameters():
            param.requires_grad = False
        
        for layer in self.layers:
            intervention_copy = self.model.steer_vector  # all layers share the same intervention instance
            self.model.set_intervention(layer, intervention_copy, "mergelora")
    
 
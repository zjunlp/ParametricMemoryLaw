import yaml
from typing import List
from ...utils.hparams import HyperParams
from dataclasses import dataclass, field

@dataclass
class MemFTHyperParams(HyperParams):
    # === Basic Config ===
    alg_name: str = 'memft'
    layers: List[int] = field(default_factory=lambda: list(range(32)))
    save_vectors: bool = True
    steer_vector_output_dir: str = "../"
    
    # === Dataset Config ===
    exclude_bos: bool = True
    max_concepts: int = 500
    max_num_of_examples: int = None
    steer_train_dataset: str = "safeedit"
    preference_pairs: List[str] = field(default_factory=lambda: ["orig_add", "orig_sub"]) 
    output_length: int = None  # The length of the output sequence for the model to generate

    # === Training Config ===
    batch_size: int = 2  # the actual batch size also needs to multiply with |preference_pairs|
    dropout: float = 0.1
    gradient_accumulation_steps: int = 6
    lr: float = 0.08
    n_epochs: int = 12
    weight_decay: float = 0.00

    pos_loss_weight: float = 1.0
    neg_loss_weight: float = 0.0
    margin_penalty_weight: float = 0.0
    ref_loss_weight: float = 0.0
    margin_threshold: float = 0.5

    use_memft: bool = False
    memft_method: str = "only_threshold"
    memft_threshold: float = 0.5
    memft_zero_bp: bool = True
    memft_debug: bool = False
    curriculum_enabled: bool = False
    curriculum_type: str = "prefix_ratio"
    curriculum_ratios: List[float] = field(default_factory=lambda: [0.2, 0.4, 0.6, 0.8, 1.0])
    curriculum_epoch_boundaries: List[int] = field(default_factory=lambda: [20, 40, 60, 80, 200])
    curriculum_shuffle_once: bool = True
    curriculum_drop_last: bool = True
    use_sliding_window: bool = False
    memory_window_size: int = 50
    sliding_tau: float = 20.0
    sliding_base_floor: float = 0.01
    sliding_scale: float = 10.0
    position_decay_lambda: float = 2.0
    use_position_weight: bool = False
    
    # === Loss Config ===
    sft_preference_type: str = "winning_only"  # add_sub, add_null, sub_null
    loss_output_dir: str = None  # Directory to save the loss CSV file
    inference: bool = False  # If True, only perform inference to generate vectors without training
    all_labels: bool = False
    init_vector_path: str = None
    ablation_vector_path: str = None

    # === Intervention Config ===
    intervention_components: str = "mlp_mid"  # lora components to intervene, e.g., ["mlp", "attn", "block"]
    intervention_method: str = "vector"  # methods for dynamic weight generation, e.g., ["vector", "local_weight", "lora"]
    intervention_positions: str = "all"
    intervention_positions_dropout: float = 0.0
    intervention_type: str = "addition"  # clamping
    low_rank_dimension: int = 4
    
    # === Steering Config ===
    steering_factors: List[float] = field(default_factory=lambda: [2.0, 4.0, 6.0, 8.0, 10.0])  
    steering_prompt_type: str = "blend_in"
    substraction_type: str = "norm"  # normal or null_it_out

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)


        assert (config and config['alg_name'] == 'memft') or print(f'MemFTHyperParams can not load from {hparams_name_or_path}, ' f'alg_name is {config["alg_name"]} ')

        return cls(**config)

import optuna
import copy
import os
from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer


NUM_TRAINING_BATCHES_TUNING = 10_000  
NUM_TRIALS = 25
STORAGE_URL = "sqlite:///hyperopt-data-aug.db"
STUDY_NAME = "data-aug-distributed-optimization"
OPTUNA_OUTPUT_DIR = "data_aug_optuna_results"


BASE_CONFIG = {
    "model": {
        "n_input_features": 512,
        "n_units": 256,
        "rnn_dropout": 0.4,  # Will be tuned
        "rnn_trainable": True,
        "n_layers": 5,
        "patch_size": 14,    # Will be tuned
        "patch_stride": 4,   # Will be tuned
        "input_network": {
            "n_input_layers": 1,
            "input_layer_sizes": [256],
            "input_trainable": True,
            "input_layer_dropout": 0.2
        }
    },
    "mode": "train",
    "use_amp": True,
    "output_dir": "trained_models/optuna_placeholder", # Will be overwritten per trial
    "checkpoint_dir": "trained_models/optuna_placeholder/ckpt", # Will be overwritten per trial
    "init_from_checkpoint": False,
    "init_checkpoint_path": None,
    "save_best_checkpoint": False, # Disable strictly to save space during tuning, or set True if storage is ample
    "save_all_val_steps": False,
    "save_final_model": False,
    "save_val_metrics": True,
    "early_stopping": True,        # Enable early stopping for speed
    "early_stopping_val_steps": 10,
    
    "num_training_batches": NUM_TRAINING_BATCHES_TUNING, # Run for fewer batches during tuning 
    
    "lr_scheduler_type": "cosine",
    "lr_max": 0.005,
    # "lr_max": 0.0005, # Lowering learning rate...
    "lr_min": 0.0001,
    "lr_decay_steps": NUM_TRAINING_BATCHES_TUNING, # Synced with num_training_batches
    "lr_warmup_steps": 1000,
    "lr_max_day": 0.0005,
    "lr_min_day": 0.0001,
    "lr_decay_steps_day": NUM_TRAINING_BATCHES_TUNING, # Synced with num_training_batches
    "lr_warmup_steps_day": 1000,
    "beta0": 0.9,
    "beta1": 0.999,
    "epsilon": 0.1,
    "weight_decay": 0.001,
    "weight_decay_day": 0,
    "seed": 10,
    "grad_norm_clip_value": 10,
    "batches_per_train_log": 200,
    "batches_per_val_step": 2000,
    "batches_per_save": 0,
    "log_individual_day_val_PER": False,
    "log_val_skip_logs": False,
    "save_val_logits": False,
    "save_val_data": False,
    
    "dataset": {
        "data_transforms": {
            "white_noise_std": 1.0,
            "constant_offset_std": 0.2,
            "random_walk_std": 0.0,
            "random_walk_axis": -1,
            "static_gain_std": 0.0,
            "random_cut": 3,
            "smooth_kernel_size": 100,
            "smooth_data": True,
            "smooth_kernel_std": 2,
            "speed_changes": 2,
            "crop_length": 1000,
            "noise_mean": 1,
            "noise_std": 0.1,
            "conv_size": 10,
            "drift_pts": 10,
            "dropout_prob": 0.05,
            "pool_size": 10,
            "quant_levels": 5,
            "quant_method": 'uniform',
            "resize_size": 10
        },
        "neural_dim": 256,
        "batch_size": 32,
        "n_classes": 41, # CONSTANT
        "max_seq_elements": 500,
        "days_per_batch": 4,
        "seed": 1,
        "num_dataloader_workers": 4,
        "loader_shuffle": False,
        "must_include_days": None,
        "test_percentage": 0.1,
        "feature_subset": None,
        "dataset_dir": "/home/mkeller/data/brain-to-text/t15_copyTask_neuralData/hdf5_data_final/",
        "bad_trials_dict": None,
        "sessions": [
            "t15.2023.08.11",
            "t15.2023.08.13",
            "t15.2023.08.18",
            "t15.2023.08.20",
            "t15.2023.08.25",
            "t15.2023.08.27",
            "t15.2023.09.01",
            "t15.2023.09.03",
            "t15.2023.09.24",
            "t15.2023.09.29",
            "t15.2023.10.01",
            "t15.2023.10.06",
            "t15.2023.10.08",
            "t15.2023.10.13",
            "t15.2023.10.15",
            "t15.2023.10.20",
            "t15.2023.10.22",
            "t15.2023.11.03",
            "t15.2023.11.04",
            "t15.2023.11.17",
            "t15.2023.11.19",
            "t15.2023.11.26",
            "t15.2023.12.03",
            "t15.2023.12.08",
            "t15.2023.12.10",
            "t15.2023.12.17",
            "t15.2023.12.29",
            "t15.2024.02.25",
            "t15.2024.03.03",
            "t15.2024.03.08",
            "t15.2024.03.15",
            "t15.2024.03.17",
            "t15.2024.04.25",
            "t15.2024.04.28",
            "t15.2024.05.10",
            "t15.2024.06.14",
            "t15.2024.07.19",
            "t15.2024.07.21",
            "t15.2024.07.28",
            "t15.2025.01.10",
            "t15.2025.01.12",
            "t15.2025.03.14",
            "t15.2025.03.16",
            "t15.2025.03.30",
            "t15.2025.04.13",
       ],
        "dataset_probability_val": [
            0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
       ]
    }
}

def objective(trial):
    if len(trial.study.trials) >= NUM_TRIALS:
        trial.study.stop() # If running in parallel, stop after 20 trials
        return float('inf')

    # cfg = copy.deepcopy(BASE_CONFIG)
    cfg = OmegaConf.load('rnn_args.yaml')
    
    # Trial directories are created when training is initialized
    trial_dir = os.path.join(OPTUNA_OUTPUT_DIR, f"trial_{trial.number}")
    
    cfg['output_dir'] = trial_dir
    cfg['checkpoint_dir'] = os.path.join(trial_dir, "checkpoints")
    for batches_param in ['num_training_batches', 'lr_decay_steps', 'lr_decay_steps_day']:
        cfg[batches_param] = NUM_TRAINING_BATCHES_TUNING
    
    white_noise_std     = trial.suggest_float("white_noise_std", 0, 5)
    constant_offset_std = trial.suggest_float("constant_offset_std", 0, 1)
    random_walk_std     = trial.suggest_float("random_walk_std", 0, 1)
    random_walk_axis    = trial.suggest_categorical("random_walk_axis", [0, 1, 2])
    static_gain_std     = trial.suggest_float("static_gain_std", 0, 1)
    random_cut          = trial.suggest_int("random_cut", 0, 5)
    smooth_kernel_size  = trial.suggest_int("smooth_kernel_size", 0, 100)
    smooth_data         = trial.suggest_categorical("smooth_data", [True, False])
    smooth_kernel_std   = trial.suggest_float("smooth_kernel_std", 0, 5)
    speed_changes       = trial.suggest_int("speed_changes", 1, 10)
    noise_mean          = trial.suggest_float("noise_mean", 0, 5)
    noise_std           = trial.suggest_float("noise_std", 0, 2)
    conv_size           = trial.suggest_int("conv_size", 1, 10)
    drift_pts           = trial.suggest_int("drift_pts", 1, 10)
    pool_size           = trial.suggest_int("pool_size", 1, 10)
    quant_levels        = trial.suggest_int("quant_levels", 1, 5)
    quant_method        = trial.suggest_categorical("quant_method", ['uniform', 'quantile'])
    homogeneous_scaling_loc = trial.suggest_float("homogeneous_scaling_loc", 0, 2)
    homogeneous_scaling_std = trial.suggest_float("homogeneous_scaling_std", 0, 0.1)
    mag_warp_loc            = trial.suggest_float("mag_warp_loc", 0, 2)
    n_knots                 = trial.suggest_int("n_knots", 1, 50)

    cfg['dataset']['data_transforms']['white_noise_std']        = white_noise_std
    cfg['dataset']['data_transforms']['constant_offset_std']    = constant_offset_std
    cfg['dataset']['data_transforms']['random_walk_std']        = random_walk_std
    cfg['dataset']['data_transforms']['random_walk_axis']       = random_walk_axis
    cfg['dataset']['data_transforms']['static_gain_std']        = static_gain_std
    cfg['dataset']['data_transforms']['random_cut']             = random_cut
    cfg['dataset']['data_transforms']['smooth_kernel_size']     = smooth_kernel_size
    cfg['dataset']['data_transforms']['smooth_data']            = smooth_data
    cfg['dataset']['data_transforms']['smooth_kernel_std']      = smooth_kernel_std
    cfg['dataset']['data_transforms']['speed_changes']          = speed_changes
    cfg['dataset']['data_transforms']['noise_mean']             = noise_mean
    cfg['dataset']['data_transforms']['noise_std']              = noise_std
    cfg['dataset']['data_transforms']['conv_size']              = conv_size
    cfg['dataset']['data_transforms']['drift_pts']              = drift_pts
    cfg['dataset']['data_transforms']['pool_size']              = pool_size
    cfg['dataset']['data_transforms']['quant_levels']           = quant_levels
    cfg['dataset']['data_transforms']['quant_method']           = quant_method
    cfg['dataset']['data_transforms']['homogeneous_scaling_loc']= homogeneous_scaling_loc
    cfg['dataset']['data_transforms']['homogeneous_scaling_std']= homogeneous_scaling_std
    cfg['dataset']['data_transforms']['mag_warp_loc']           = mag_warp_loc       
    cfg['dataset']['data_transforms']['n_knots']                = n_knots              

    patch_size = 10 # trial.suggest_categorical("patch_size", [10, 14, 20])
    
    if patch_size > 0:
        # Stride must be <= patch_size
        patch_stride = trial.suggest_int("patch_stride", 1, patch_size)
    else:
        patch_stride = 0
        
    cfg['model']['patch_size'] = patch_size
    cfg['model']['patch_stride'] = patch_stride

    args = OmegaConf.create(cfg)

    trainer = BrainToTextDecoder_Trainer(args)
    try:
        metrics = trainer.train()
        return metrics['val_losses'][-1]
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

if __name__ == "__main__":
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,      # Connect to the SQLite DB
        load_if_exists=True,      # Join the study if it already exists
        direction="minimize"
    )

    study.optimize(objective, n_trials=NUM_TRIALS)  # try ~25–50 trials

    print("Best trial:")
    trial = study.best_trial
    print(f"  Accuracy: {trial.value:.2f}%")
    print("  Best hyperparameters:", trial.params)
    


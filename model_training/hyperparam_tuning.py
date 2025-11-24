import numpy as np
import os
import optuna
from omegaconf import OmegaConf
import copy
import gc
import torch
from rnn_trainer import BrainToTextDecoder_Trainer

NUM_TRAINING_BATCHES_TUNING = 20_000  
NUM_TRIALS = 50


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
            "smooth_kernel_std": 2
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

    cfg = copy.deepcopy(BASE_CONFIG)
    
    # Trial directories are created when training is initialized
    trial_dir = os.path.join("optuna_results", f"trial_{trial.number}")
    
    cfg['output_dir'] = trial_dir
    cfg['checkpoint_dir'] = os.path.join(trial_dir, "checkpoints")

    # 3. Hyperparameter Suggestions
    
    # --- Dropout ---
    rnn_dropout = trial.suggest_float("rnn_dropout", 0.1, 0.5, step=0.05)
    input_dropout = trial.suggest_float("input_dropout", 0.0, 0.5, step=0.05)
    
    cfg['model']['rnn_dropout'] = rnn_dropout
    cfg['model']['input_dropout'] = input_dropout

    patch_size = trial.suggest_categorical("patch_size", [10, 14, 20])
    
    if patch_size > 0:
        # Stride must be <= patch_size
        patch_stride = trial.suggest_int("patch_stride", 1, patch_size)
    else:
        patch_stride = 0
        
    cfg['model']['patch_size'] = patch_size
    cfg['model']['patch_stride'] = patch_stride

    # --- Heads ---
    # n_units is 512. n_heads must divide 512.
    # Options: 4 (128 dim/head), 8 (64 dim/head), 16 (32 dim/head)
    n_heads = trial.suggest_categorical("n_heads", [4, 8, 16])
    cfg['model']['n_heads'] = n_heads

    args = OmegaConf.create(cfg)
    
    try:
        trainer = BrainToTextDecoder_Trainer(
             args, 
             model_architecture="convformer", 
             compile=False 
        )
        
        train_stats = trainer.train()

        np.save(os.path.join(trial_dir, "metrics.npy"), train_stats)
        best_per_in_run = min(train_stats['val_PERs'])

        # Force memory clean up
        del trainer.model
        del trainer.optimizer
        del trainer
        
        gc.collect()
        
        torch.cuda.empty_cache()

        return best_per_in_run

    except Exception as e:
        print(f"Trial {trial.number} failed with error: {e}")
        
        # ALSO CLEAN UP HERE IN CASE OF FAILURE
        if 'trainer' in locals():
            del trainer
        gc.collect()
        torch.cuda.empty_cache()
        
        return float('inf')

if __name__ == "__main__":
    import optuna.visualization as vis
    import matplotlib.pyplot as plt

    os.makedirs("optuna_results", exist_ok=True)

    study = optuna.create_study(
        direction="minimize", 
        study_name="convformer_tuning",
        storage="sqlite:///optuna_results/fewer_trials_study.db",
        load_if_exists=True,
    )

    print("Starting optimization...")
    study.optimize(objective, n_trials=NUM_TRIALS, gc_after_trial=True)

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Best trial:")
    trial = study.best_trial

    print("    Value: ", trial.value)
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # --- VISUALIZATION SECTION ---
    print("Generating visualizations...")
    
    # 1. Optimization History (How metric improved over trials)
    try:
        fig = vis.plot_optimization_history(study)
        fig.write_html("optuna_results/optimization_history.html")
        print("Saved optimization_history.html")
    except Exception as e:
        print(f"Could not plot optimization history: {e}")

    # 2. Parallel Coordinate Plot 
    try:
        fig = vis.plot_parallel_coordinate(study)
        fig.write_html("optuna_results/parallel_coordinate.html")
        print("Saved parallel_coordinate.html")
    except Exception as e:
        print(f"Could not plot parallel coordinate: {e}")

    # 3. Hyperparameter Importance
    try:
        fig = vis.plot_param_importances(study)
        fig.write_html("optuna_results/param_importances.html")
        print("Saved param_importances.html")
    except Exception as e:
        print(f"Could not plot param importances: {e}")

    # 4. Slice Plot 
    try:
        fig = vis.plot_slice(study)
        fig.write_html("optuna_results/slice_plot.html")
        print("Saved slice_plot.html")
    except Exception as e:
        print(f"Could not plot slice: {e}")

import os
from omegaconf import OmegaConf
import optuna
from rnn_trainer import BrainToTextDecoder_Trainer
import torch

# --- CHANGE 1: Define a shared storage URL ---
# This creates a file 'hyperopt.db' in your current folder that all GPUs will share.
STORAGE_URL = "sqlite:///hyperopt.db"
STUDY_NAME = "rnn_distributed_optimization"

PATH_TO_BASE_ARGS = "bigru-trials/base_args_for_tuning.yaml"
NUM_TRIALS = 15

def objective(trial):
    # Check if we've hit the max trials (shared across all GPUs)
    if len(trial.study.trials) >= NUM_TRIALS:
        trial.study.stop()
        return float('inf')

    cfg = OmegaConf.load(PATH_TO_BASE_ARGS)
    
    trial_dir = os.path.join("rnn-optuna-results", f"trial_{trial.number}")
    cfg['output_dir'] = trial_dir
    cfg['checkpoint_dir'] = os.path.join(trial_dir, "checkpoints")
    
    # --- CRITICAL FIX from previous step ---
    # Lock input features to dataset dim
    n_input_features = cfg['dataset']['neural_dim'] 
    if 'input_network' in cfg['model']:
        cfg['model']['input_network']['input_layer_sizes'][0] = n_input_features

    # --- CHANGE 2: Force GPU 0 ---
    # Since we use CUDA_VISIBLE_DEVICES, every script will think it is on "cuda:0"
    # This prevents the script from trying to jump to a GPU it can't see.
    cfg['gpu_number'] = 0 
    
    # ... (Your existing hyperparam logic) ...
    n_units = trial.suggest_categorical("n_units", [128, 256, 512, 1024])
    n_layers = trial.suggest_int("n_layers", 1, 4)
    rnn_dropout = trial.suggest_float("rnn_dropout", 0.0, 0.5, step=0.1)
    patch_size = trial.suggest_int("patch_size", 10, 20, step=2)
    
    cfg['model']['n_input_features'] = n_input_features
    cfg['model']['n_units'] = n_units
    cfg['model']['n_layers'] = n_layers
    cfg['model']['rnn_dropout'] = rnn_dropout
    cfg['model']['patch_size'] = patch_size
    
    if patch_size > 0:
        patch_stride = trial.suggest_int("patch_stride", 1, patch_size)
        cfg['model']['patch_stride'] = patch_stride
    else:
        cfg['model']['patch_stride'] = 0
        
    args = OmegaConf.create(cfg)

    trainer = BrainToTextDecoder_Trainer(args, model_architecture="bigru")
    
    try:
        metrics = trainer.train()
        return metrics['val_losses'][-1]
    except Exception as e:
        print(f"Trial failed: {e}")
        return float('inf')

if __name__ == "__main__":
    # --- CHANGE 3: Create study with shared storage ---
    study = optuna.create_study(
        study_name=STUDY_NAME,
        storage=STORAGE_URL,      # Connect to the SQLite DB
        load_if_exists=True,      # Join the study if it already exists
        direction="minimize"
    )
    
    # You can set n_trials higher if you want to run essentially "infinite" loops 
    # until the condition in objective() stops them.
    study.optimize(objective, n_trials=NUM_TRIALS) 

    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value:.4f}")
    print("  Best hyperparameters:", trial.params)

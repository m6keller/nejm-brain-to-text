import os
from os import PathLike
import numpy as np

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
from utils import ModelArchitecture, MODEL_CHOICES

    
def main(model_architecture: ModelArchitecture = "rnn", config: PathLike = "rnn_args.yaml", compile: bool = False):
    args = OmegaConf.load(config)
    trainer = BrainToTextDecoder_Trainer(args, model_architecture=model_architecture, compile=compile)
    metrics = trainer.train()
    print("Training completed. Final metrics:", metrics)
    np.save(
        os.path.join(args.output_dir, "final_training_metrics.npy"), 
        metrics
    )



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Brain-to-Text Decoder Model")
    parser.add_argument("--model-architecture", type=str, choices=MODEL_CHOICES, default="rnn",
                        help="Model architecture to use: 'rnn', 'convformer', or 'mamba'")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file")
    parser.add_argument("--compile", action="store_true",
                        help="Whether to compile the model for optimization")
    args = parser.parse_args()

    main(model_architecture=args.model_architecture, config=args.config, compile=args.compile)
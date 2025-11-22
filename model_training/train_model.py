import json
import os
from os import PathLike

from omegaconf import OmegaConf
from rnn_trainer import BrainToTextDecoder_Trainer
from utils import ModelArchitecture


    
def main(model_architecture: ModelArchitecture = "rnn", config: PathLike = "rnn_args.yaml"):
    args = OmegaConf.load(config)
    trainer = BrainToTextDecoder_Trainer(args, model_architecture=model_architecture)
    metrics = trainer.train()
    print("Training completed. Final metrics:", metrics)
    json.dump(metrics, open(os.path.join(args.output_dir, "final_training_metrics.json"), "w"))



if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Brain-to-Text Decoder Model")
    parser.add_argument("--model-architecture", type=str, choices=["rnn", "convformer", "mamba"], default="rnn",
                        help="Model architecture to use: 'rnn', 'convformer', or 'mamba'")
    parser.add_argument("--config", type=str, required=True,
                        help="Path to the configuration YAML file")
    args = parser.parse_args()

    main(model_architecture=args.model_architecture, config=args.config)
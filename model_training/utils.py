import torch
from pathlib import Path
from typing import Literal

from rnn_model import GRUDecoder
from convformer_model import ConvFormerDecoder
from mamba_model import MambaDecoder
from relative_convformer_model import RelativeConvFormer


DATA_BASE_PATH = Path('/home/mkeller/data/brain-to-text/')

ModelArchitecture = Literal["rnn", "convformer", "mamba", "relative-convformer"]

def load_model(args, model_architecture: ModelArchitecture) -> torch.nn.Module:
    match model_architecture:
        case "rnn":
            return GRUDecoder(
                neural_dim = args['model']['n_input_features'],
                n_units = args['model']['n_units'],
                n_days = len(args['dataset']['sessions']),
                n_classes  = args['dataset']['n_classes'],
                rnn_dropout = args['model']['rnn_dropout'], 
                input_dropout = args['model']['input_network']['input_layer_dropout'], 
                n_layers = args['model']['n_layers'],
                patch_size = args['model']['patch_size'],
                patch_stride = args['model']['patch_stride'],
            )
        case "convformer":
            return ConvFormerDecoder(
                neural_dim= args['model']['n_input_features'],
                n_units = args['model']['n_units'],
                n_days = len(args['dataset']['sessions']),
                n_classes  = args['dataset']['n_classes'],
                rnn_dropout = args['model']['rnn_dropout'], 
                input_dropout = args['model']['input_network']['input_layer_dropout'], 
                n_layers = args['model']['n_layers'],
                patch_size = args['model']['patch_size'],
                patch_stride = args['model']['patch_stride'],
            )
        case "mamba":
            return MambaDecoder(
                neural_dim= args['model']['n_input_features'],
                n_units = args['model']['n_units'],
                n_days = len(args['dataset']['sessions']),
                n_classes  = args['dataset']['n_classes'],
                rnn_dropout = args['model']['rnn_dropout'], 
                input_dropout = args['model']['input_network']['input_layer_dropout'], 
                n_layers = args['model']['n_layers'],
                patch_size = args['model']['patch_size'],
                patch_stride = args['model']['patch_stride'],
            )
        case "relative-convformer":
            return RelativeConvFormer(
                input_dim= args['model']['n_input_features'],
                num_classes = args['dataset']['n_classes'],
                n_days = len(args['dataset']['sessions']),
                patch_size = args['model']['patch_size'],
            )
        case _:
            raise ValueError(f"Invalid model architecture: {model_architecture}")

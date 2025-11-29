import torch
from pathlib import Path
from typing import Literal

from rnn_model import GRUDecoder
from convformer_model import ConvFormerDecoder
from mamba_model import MambaDecoder
from relative_convformer_model import RelativeConvFormer
from bidirectional_gru_model import BidirectionalGRUDecoder


DATA_BASE_PATH = Path('/home/mkeller/data/brain-to-text/')

MODEL_CHOICES = ["rnn", "convformer", "mamba", "relative-convformer", "bigru"]

ModelArchitecture = Literal["rnn", "convformer", "mamba", "relative-convformer", "bigru"]

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
                n_layers= args['model']['n_layers'],
                patch_stride= args['model']['patch_stride'],
                input_dropout= args['model']['input_network']['input_layer_dropout'],
                n_attention_heads= args['model']['n_attention_heads'],
                intermediate_size= args['model']['intermediate_size'],
                hidden_size= args['model']['n_units'],
            )
        case "bigru":
            return BidirectionalGRUDecoder(
                neuraldim = args['model']['n_input_features'],
                nunits = args['model']['n_units'],
                ndays = len(args['dataset']['sessions']),
                nclasses  = args['dataset']['n_classes'],
                rnndropout = args['model']['rnn_dropout'],
                inputdropout = args['model']['input_network']['input_layer_dropout'],
                nlayers = args['model']['n_layers'],
                patchsize = args['model']['patch_size'],
                patchstride = args['model']['patch_stride'],
            )
        case _:
            raise ValueError(f"Invalid model architecture: {model_architecture}")

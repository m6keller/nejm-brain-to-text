import torch
from torch import nn

class BidirectionalGRUDecoder(nn.Module):
    '''
    Bidirectional GRU decoder adds backwards context.
    '''
    def __init__(self, neuraldim, nunits, ndays, nclasses,
                 rnndropout=0.0, inputdropout=0.0, nlayers=5,
                 patchsize=0, patchstride=0):
        super(BidirectionalGRUDecoder, self).__init__()
        self.neuraldim = neuraldim
        self.nunits = nunits
        self.nclasses = nclasses
        self.nlayers = nlayers
        self.ndays = ndays
        self.rnndropout = rnndropout
        self.inputdropout = inputdropout
        self.patchsize = patchsize
        self.patchstride = patchstride
        
        
        self.daylayeractivation = nn.Softsign()
        self.dayweights = nn.ParameterList(
            [nn.Parameter(torch.eye(neuraldim)) for _ in range(ndays)]
        )
        self.daybiases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, neuraldim)) for _ in range(ndays)]
        )
        self.daylayerdropout = nn.Dropout(inputdropout)

        self.inputsize = neuraldim
        if patchsize > 0:
            self.inputsize = patchsize * neuraldim

        self.gru = nn.GRU(
            input_size=self.inputsize,
            hidden_size=nunits,
            num_layers=nlayers,
            dropout=rnndropout,
            batch_first=True,
            bidirectional=True,
        )

        # Output size doubles due to bidirection
        self.out = nn.Linear(nunits * 2, nclasses)
        nn.init.xavier_uniform_(self.out.weight)

        self.h0 = nn.Parameter(torch.zeros(nlayers * 2, 1, nunits))

    def forward(self, x, day_idx, states=None, return_state=False):
        dayweights = torch.stack([self.dayweights[i] for i in day_idx], dim=0)
        daybiases = torch.cat([self.daybiases[i] for i in day_idx], dim=0).unsqueeze(1)
        x = torch.einsum('btd,bdf->btf', x, dayweights) + daybiases
        x = self.daylayeractivation(x)
        if self.inputdropout > 0:
            x = self.daylayerdropout(x)

        if self.patchsize > 0:
            batch_size, time_steps, feat_dim = x.size()
            x = x.unsqueeze(1)
            x = x.unfold(2, self.patchsize, self.patchstride)
            x = x.squeeze(1).permute(0, 1, 3, 2)
            x = x.reshape(batch_size, x.size(1), -1)

        if states is None:
            batch_size = x.shape[0]
            h0 = self.h0.expand(self.nlayers * 2, batch_size, self.nunits).contiguous()
            states = h0

        output, hidden_states = self.gru(x, states)
        logits = self.out(output)
        if return_state:
            return logits, hidden_states
        return logits


def freeze_layers(model, freeze_rnn=True, freeze_day_layers=False):
    '''
    Freeze specified parts of the model to enable parameter-efficient finetuning.
    - freeze_rnn: freezes GRU/LSTM layers
    - freeze_day_layers: freezes day-specific input layers (dayweights, daybiases)
    '''
    for name, param in model.named_parameters():
        if freeze_rnn and "gru" in name or "lstm" in name:
            param.requires_grad = False
        if freeze_day_layers and ("dayweights" in name or "daybiases" in name):
            param.requires_grad = False

def unfreeze_all(model):
    for _, param in model.named_parameters():
        param.requires_grad = True

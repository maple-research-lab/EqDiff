import torch.nn as nn
import torch

class Mapper(nn.Module):
    def __init__(self,
        input_dim: int,
        output_dim: int,
    ):
        super(Mapper, self).__init__()

        for i in range(5):
            setattr(self, f'mapping_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, 1024),
                                         nn.LayerNorm(1024),
                                         nn.LeakyReLU(),
                                         nn.Linear(1024, output_dim)))

            setattr(self, f'mapping_patch_{i}', nn.Sequential(nn.Linear(input_dim, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, 1024),
                                                        nn.LayerNorm(1024),
                                                        nn.LeakyReLU(),
                                                        nn.Linear(1024, output_dim)))

    def forward(self, embs):
        hidden_states = ()
        for i, emb in enumerate(embs):
            hidden_state = getattr(self, f'mapping_{i}')(emb[:, :1]) + getattr(self, f'mapping_patch_{i}')(emb[:, 1:]).mean(dim=1, keepdim=True)
            hidden_states += (hidden_state, )
        hidden_states = torch.cat(hidden_states, dim=1)
        return hidden_states
import torch
import torch.nn as nn
import torch.nn.functional as F


class Attn(nn.Module):
    def __init__(self, hidden_size, cuda=False):
        super(Attn, self).__init__()

        self.hidden_size = hidden_size
        self.do_cuda = cuda

        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.randn(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        batch_size = encoder_outputs.size(0)
        #print('[attn] seq len', seq_len)
        #print('[attn] encoder_outputs', encoder_outputs.size()) # B x S x N
        #print('[attn] hidden', hidden.size()) # B x S=1 x N

        # Create variable to store attention energies
        attn_energies = torch.zeros(batch_size, seq_len) # B x S

        if self.do_cuda:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(batch_size):
            # Calculate energy for each encoder output
            for i in range(seq_len):
                attn_energies[b, i] = self.score(hidden[b, :], encoder_outputs[b, i].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        #print('[attn] attn_energies', attn_energies.size())
        #print('[attn] energies: ', attn_energies)
        return F.softmax(attn_energies, dim=-1).unsqueeze(1)

    def score(self, hidden, encoder_output):

        energy = self.attn(torch.cat((hidden, encoder_output), 1))
        #print('energy: ', energy)
        energy = torch.bmm(self.v.unsqueeze(1), 
                           energy.unsqueeze(2))
        return energy

import torch.nn as nn
import torch

class AtNet_decoder(nn.Module):
    def __init__(self, in_ch, hid_ch, num_class):
        super(AtNet_decoder, self).__init__()
        self.mlp1 = Simple_MLP(in_ch, hid_ch)
        self.mlp2 = Simple_MLP(hid_ch, num_class)
        
    
    
    def forward(self, fused):
        output = self.mlp1(fused)
        output = self.mlp2(output)
        
        return output
    
    
class Simple_MLP(nn.Module):
    def __init__(self, in_ch, out_ch): 
        super(Simple_MLP, self).__init__()

        self.ln = nn.Linear(in_ch, out_ch)
        self.bn = nn.BatchNorm1d(out_ch)
        self.ru = nn.LeakyReLU(0.1)

    
    def forward(self, input):
        output = self.ln(input)
        output = self.bn(output)
        output = self.ru(output)

        return output
    
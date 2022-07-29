import torch
from torch.nn import Module, Dropout, Parameter, Linear
import math
from torch.autograd import Variable
import torch.nn.functional as F

def attention(q, k, d_k, mask=None, dropout=None):
    
    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(d_k)     # [bz,sl,h,topm,topm]
    
    if mask is not None:
        mask_3 = mask.unsqueeze(-1)
        mask_4 = mask.unsqueeze(-2)
        mask = (mask_3 & mask_4).unsqueeze(2)
        scores = scores.masked_fill(mask == False, -10000) # [bz,sl,h,topm,topm]
    
    scores = F.softmax(scores, dim=-1)
    
    if dropout is not None:
        scores = dropout(scores)
        
    return scores

class MultiHeadAttention(Module):
    def __init__(self, heads, dim, dropout = 0.1):
        super().__init__()
        
        self.dim = dim
        self.d_k = dim // heads
        self.h = heads
        
        self.q_linear = Linear(dim, dim)
        self.k_linear = Linear(dim, dim)
        
        self.dropout = Dropout(dropout)
    
    def forward(self, q, k, mask=None):
        
        bs,sq,_,_ = q.size()
        
        # perform linear operation and split into N heads
        k = self.k_linear(k).view(bs, sq, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, sq, -1, self.h, self.d_k)
        
        # transpose to get dimensions bs * sent_len * N * topm * dim
        k = k.transpose(2,3)
        q = q.transpose(2,3)

        # calculate attention using function we will define next
        scores = attention(q, k, self.d_k, mask, self.dropout)
        # concatenate heads and get the diagonal elements
        concat = torch.sum(scores,2)    # [bz,sl,topm,topm]
        output = torch.diagonal(concat, dim1=2, dim2=3)
    
        return output   # [bz,sl,topm]
    
class Norm(Module):
    def __init__(self, dim, eps = 1e-6):
        super().__init__()
        self.size = dim
        
        # create two learnable parameters to calibrate normalisation
        self.alpha = Parameter(torch.ones(self.size))
        self.bias = Parameter(torch.zeros(self.size))
        self.eps = eps
    
    def forward(self, x):
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) \
        / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class AttentionLayer(Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm = Norm(dim)
        self.attn = MultiHeadAttention(heads, dim, dropout=dropout)
        self.dropout = Dropout(dropout)
        
    def forward(self, x, mask):
        x2 = self.norm(x)
        atten_out = self.dropout(self.attn(x2,x2,mask))
        return atten_out

class PositionalEncoder(Module):
    def __init__(self, dim, topm, dropout = 0.1):
        super().__init__()
        self.dim = dim
        self.dropout = Dropout(dropout)
        # create constant 'pe' matrix with values dependant on 
        # pos and i
        pe = torch.zeros(topm, dim)
        for pos in range(topm):
            for i in range(0, dim, 2):
                pe[pos, i] = \
                math.sin(pos / (10000 ** ((2 * i)/dim)))
                pe[pos, i + 1] = \
                math.cos(pos / (10000 ** ((2 * (i + 1))/dim)))
        pe = (pe.unsqueeze(0)).unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # make embeddings relatively larger
        x = x * math.sqrt(self.dim)
        
        pe = Variable(self.pe, requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)
    
class Attention(Module):
    def __init__(self, dim, topm, heads, dropout):
        super().__init__()
        self.pe = PositionalEncoder(dim, topm, dropout=dropout)
        self.layers = AttentionLayer(dim, heads, dropout)
        self.norm = Norm(topm)
        
    def forward(self, src, mask):
        x = self.pe(src)
        atten_x = self.layers(x, mask)
        return self.norm(atten_x)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelArgs:
    dim : int= 4096
    n_layers : int= 32
    n_heads : int= 32
    n_kv_heads : Optional[int] = None
    vocab_size : int = -1  #set when load tokenizer
    multiple_of : int = 256
    ffn_dim_multiplier : Optional[float] = None
    norm_eps : float = 1e-5

    #kv cache
    max_batch_size : int = 32
    max_seq_len : int = 2048

    device : str = None

def precompute_theta_pos_frequencies(head_dim, seq_len, device, theta = 10000.0):
    assert head_dim %2 == 0, "Dimensions must be divisible by 2"
    #Build theta parameters
    #According to the formula theta_i = 1000 ^ (-2(i-1)/dim) for i = [1,2,... dim/2]
    # shape: (Head_dim /2)
    theta_numerator = torch.arange(0,head_dim,2).float()  #Like a for loop, (start, end, step)
    #shape: (Head_dim/2)
    theta = 1.0/ (theta ** (theta_numerator / head_dim)).to(device)
    #Constrcut positions (the "m" parameter)
    #shape: (Seq_len)
    m = torch.arange(seq_len, device = device)
    #Multiply each theta by each position using outer products
    #SHape: (seq_len) outer_product * (Head_dim/2) -> (seq_len, Head_dim/2)
    freqs = torch.outer(m,theta).float()  #takes first arg as rows, second as columns, then row[i] * col[j]
    #Compute complex numbers in the polar form c = R * exp(i*theta), where R = 1 as follows:
    #shape: (seq_len, Head_dim/2)
    freqs_complex = torch.polar(torch.ones_like(freqs), freqs)  #R = 1, theta = freqs
    return freqs_complex

def apply_rotary_pos_emb(x, freqs_complex, device: str):
    #(B, n_heads, H, head_dim) -> (B, seq_len, H, Head_dim/2)
    x_complex = torch.view_as_complex(x.float().reshape(*x.shape[:-1],-1,2))
    #(Seq_len, Head_dim/2) -> ((1, seq_len, 1, Head_dim/2))
    freqs_complex = freqs_complex.unsqueeze(0).unsqueeze(2)
    #(B, Seq_len, H, Head_dim/2) * (1, seq_len, 1, Head_dim/2) -> (B, seq_len, H, Head_dim/2)
    x_rotated = x_complex * freqs_complex
    #(B, seq_len, H, head_dim/2) -> (B, n_heads, H, head_dim/2, 2)
    x_out = torch.view_as_real(x_rotated)
    #(B, seq_len, H, head_dim/2, 2) -> (B, n_heads, H, head_dim)
    x_out = x_out.reshape(*x.shape)
    return x_out.type_as(x).to(device)

def repeat_kv(x, n_rep):
    batch_size, seq_len, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    else:
        #(B, seq_len, n_kv_heads,1, head_dim)
        return (x[:,:,:,None,:].expand(batch_size, seq_len, n_kv_heads, n_rep, head_dim)
                     .reshape(batch_size, seq_len, n_kv_heads * n_rep, head_dim))

class RMSNorm(nn.Module):

    def __init__(self, dim , eps = 1e-6):
        super().__init__()
        self.eps = eps
        #Gamma param
        self.weight = nn.Parameter(torch.ones(dim))
    
    def _norm(self, x: torch.Tensor):
        #(B, seq_len, Dim)
        #rsqrt = 1/sqrt(x)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim = True) + self.eps)
    
    def forward(self, x):
        #(Dim * (B, seq_len, Dim)) = (B, seq_len, Dim)
        return self._norm(x.float()).type_as(x) * self.weight
    
class EncoderBlock(nn.Module):
     def __init__(self, args: ModelArgs):
         super().__init__()

         self.args = args
         self.dim = args.dim
         self.head_dim = args.dim // args.n_heads

         self.attention = SelfAttention(args)
         self.feed_forward = FeedForward(args)

         #Normalization Before the self attention
         self.attention_norm = RMSNorm(args.dim, eps = args.norm_eps)
         #Normalization before forward block
         self.ffn_norm = RMSNorm(args.dim, eps = args.norm_eps)

     def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
        # (B, seq_len, Dim) + (B, seq_len, Dim) -> (B, seq_len, Dim)
        h = x + self.attention.forward(self.attention_norm(x), start_pos, freqs_complex)
        out = h + self.feed_forward.forward(self.ffn_norm(h))
        return out
    

class SelfAttention(nn.Module):
     def __init__(self, args):
         super().__init__()

         #Indicated number of heads for keys and values
         self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
         #Indicates number of heads for queries
         self.n_heads_q = args.n_heads
         #Indicates how many times the heads of keys and values shld be repeated to match the head  of the queires
         self.n_rep = self.n_heads_q // self.n_kv_heads
         #Indicates dimension of each head 
         self.head_dim = args.dim // args.n_heads

         self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias = False)
         self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
         self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias = False)
         self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias = False)

         self.cache_k = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
         self.cache_v = torch.zeros((args.max_batch_size, args.max_seq_len, self.n_kv_heads, self.head_dim))
    
     def forward(self, x: torch.Tensor, start_pos: int, freqs_complex: torch.Tensor):
    
        batch_size, seq_len, _ = x.shape #(B, 1, Dim)
        #(B,1,Dim) -> (B,1,H_Q * Head_dim)
        xq = self.wq(x) 
        #(B, 1,Dim) -> (B, 1, H_KV * head_dim)
        xk = self.wk(x)
        xv = self.wv(x)

        #(B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.view(batch_size, seq_len, self.n_heads_q, self.head_dim)
        xk = xk.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
        #(B, 1, H_KV, head_dim) -> (B, H_KV, 1, head_dim)
        xv = xv.view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

        #Does not change shape
        xq = apply_rotary_pos_emb(xq, freqs_complex, device = x.device)
        xk = apply_rotary_pos_emb(xk, freqs_complex, device = x.device)

        #Replace entry in cache for this token
        self.cache_k[:batch_size, start_pos:start_pos + seq_len] = xk
        self.cache_v[:batch_size, start_pos:start_pos + seq_len] = xv


        #retrieve all keys and values from cache up to current position
        keys = self.cache_k[:batch_size, 0: start_pos + seq_len]  #(B, Seq_length_kv, H_KV, head_dim)
        values = self.cache_v[:batch_size, 0: start_pos + seq_len]  #(B, Seq_length_kv, H_KV, head_dim)

        #Repeat heads of k and v to reach the number of heads of query
        keys = repeat_kv(keys, self.n_rep)
        values = repeat_kv(values, self.n_rep)

        #(B, 1, H_Q, head_dim) -> (B, H_Q, 1, head_dim)
        xq = xq.transpose(1,2) 
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)

        #(B, H_Q, 1, head_dim) @ (B, H_Q, head_dim, Seq_length_kv) -> (B, H_Q, 1, Seq_length_kv)
        scores = torch.matmul(xq, keys.transpose(2,3))/math.sqrt(self.head_dim)
        scores = F.softmax(scores.float(), dim= -1).type_as(xq)

        #(B, H_Q, 1, Seq_length_kv) @ (B, H_Q, Seq_length_kv, head_dim) -> (B, H_Q, 1, head_dim)  
        output = torch.matmul(scores, values)
        output = (output.transpose(1,2).contiguous().view(batch_size, seq_len, -1))
        return self.wo(output) #(B, 1 Dim) -> (B, 1, Dim)

class FeedForward(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        hidden_dim = args.dim * 4
        hidden_dim = int(2* hidden_dim/3)
        if args.ffn_dim_multiplier is not None:
            hidden_dim = int(args.ffn_dim_multiplier * args.dim)
        #round the hidden_dim to the nearest multiple of args.multiple_of
        hidden_dim = args.multiple_of * ((hidden_dim + args.multiple_of -1)// args.multiple_of)

        self.w1 = nn.Linear(args.dim, hidden_dim, bias = False)
        self.w2 = nn.Linear(hidden_dim, args.dim,  bias =False)
        self.w3 = nn.Linear(args.dim, hidden_dim, bias = False)

    def forward(self, x):
        swish = F.silu(self.w1(x))
        x_V = self.w3(x)
        x = swish * x_V
        x = self.w2(x)
        return x

class Transformer(nn.Module):
    
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size != -1, "Vocab size must be set"

        self.args = args
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.tok_embeddings = nn.Embedding(self.vocab_size, args.dim)

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(EncoderBlock(args))
        
        self.norm = RMSNorm(args.dim, eps = args.norm_eps)  #so u nvr divide by zero
        self.output = nn.Linear(args.dim, self.vocab_size, bias = False)

        self.freqs_complex = precompute_theta_pos_frequencies(self.args.dim // self.args.n_heads, self.args.max_seq_len *2, device = self.args.device)

    def forward(self, tokens: torch.Tensor, start_pos):
        # (Batch B, seq_length)
        batch_size, seq_len = tokens.shape
        assert seq_len == 1, "Only one token at a time can be processed"

        #(B, sqeq_length) -> (B, seq_length, Dim)
        h = self.tok_embeddings(tokens)

        #retrieve pairs (m, theta) corresponding to te posiitons [start pos, start_pos + seq_length]
        freqs_complex = self.freqs_complex[start_pos:start_pos + seq_len]

        #consequitvely apply all encoder layers
        for layer in self.layers:
            h = layer(h, start_pos, freqs_complex)
        h = self.norm(h)
        output = self.output(h).float()
        return output


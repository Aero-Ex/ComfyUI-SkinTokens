import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

class SimVQ(nn.Module):
    """
    Improved version over VectorQuantizer, can be used as a drop-in replacement. Mostly
    avoids costly matrix multiplications and allows for post-hoc remapping of indices.
    """
    # NOTE: due to a bug the beta term was applied to the wrong term. for
    # backwards compatibility we use the buggy version by default, but you can
    # specify legacy=False to fix it.
    def __init__(self, n_e, e_dim, beta=0.25, remap=None, unknown_index="random",
                 same_index_shape=False, legacy=True):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.legacy = legacy

        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        self.codebook_size = self.n_e
        nn.init.normal_(self.embedding.weight, mean=0, std=self.e_dim**-0.5)
        for p in self.embedding.parameters():
            p.requires_grad = False
        
        self.embedding_proj = nn.Linear(self.e_dim, self.e_dim)
    
        self.remap = remap
        if self.remap is not None:
            self.register_buffer("used", torch.tensor(np.load(self.remap)))
            self.re_embed = self.used.shape[0]
            self.unknown_index = unknown_index # "random" or "extra" or integer
            if self.unknown_index == "extra":
                self.unknown_index = self.re_embed
                self.re_embed = self.re_embed+1
            print(f"Remapping {self.n_e} indices to {self.re_embed} indices. "
                  f"Using {self.unknown_index} for unknown indices.")
        else:
            self.re_embed = n_e

        self.same_index_shape = same_index_shape

    def remap_to_used(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        match = (inds[:,:,None]==used[None,None,...]).long()
        new = match.argmax(-1)
        unknown = match.sum(2)<1
        if self.unknown_index == "random":
            new[unknown]=torch.randint(0,self.re_embed,size=new[unknown].shape).to(device=new.device)
        else:
            new[unknown] = self.unknown_index
        return new.reshape(ishape)

    def unmap_to_all(self, inds):
        ishape = inds.shape
        assert len(ishape)>1
        inds = inds.reshape(ishape[0],-1)
        used = self.used.to(inds)
        if self.re_embed > self.used.shape[0]: # extra token
            inds[inds>=self.used.shape[0]] = 0 # simply set to zero
        back=torch.gather(used[None,:][inds.shape[0]*[0],:], 1, inds)
        return back.reshape(ishape)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        assert z.shape[-1] == self.e_dim
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        quant_codebook = self.embedding_proj(self.embedding.weight)

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            quantization_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            quantization_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.same_index_shape:
            min_encoding_indices = min_encoding_indices.reshape(
                z_q.shape[0], z_q.shape[2], z_q.shape[3])
            
        return z_q, min_encoding_indices, quantization_loss

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        if self.remap is not None:
            indices = indices.reshape(shape[0],-1) # add batch axis
            indices = self.unmap_to_all(indices)
            indices = indices.reshape(-1) # flatten again

        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

    def indices_to_codes(self, indices):
        return self.get_codebook_entry(indices, None)

def entropy(prob):
    return (-prob * log(prob)).sum(dim=-1)

class SimVQ1D(SimVQ):

    def __init__(self, n_e, e_dim, dim, beta=0.25, remap=None, unknown_index="random", same_index_shape=True, legacy=True):
        super().__init__(n_e, e_dim, beta, remap, unknown_index, same_index_shape, legacy)

        self.project_in = nn.Linear(dim, e_dim)
        self.project_out = nn.Linear(e_dim, dim)

    def forward(self, z):
        # reshape z -> (batch, height, width, channel) and flatten
       #assert z.shape[-1] == self.e_dim
        z = self.project_in(z)
        
        z_flattened = z.view(-1, self.e_dim)
        # distances from z to embeddings e_j (z - e)^2 = z^2 + e^2 - 2 e * z
        
        quant_codebook = self.embedding_proj(self.embedding.weight)

        # # Use IBQ
        # logits = torch.matmul(z_flattened, quant_codebook.t())
        # Ind_soft = torch.softmax(logits, dim=1)
        # indices = torch.argmax(Ind_soft, dim=1)
        # Ind_hard = F.one_hot(indices, num_classes=Ind_soft.shape[1])
        # Ind = Ind_hard - Ind_soft.detach() + Ind_soft
        # z_q = torch.matmul(Ind, quant_codebook).view(z.shape)

        # if not self.legacy:
        #     quantization_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
        #            torch.mean((z_q - z.detach()) ** 2)
        # else:
        #     quantization_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
        #            torch.mean((z_q - z.detach()) ** 2)

        # return z_q, indices, quantization_loss

        d = torch.sum(z_flattened ** 2, dim=1, keepdim=True) + \
            torch.sum(quant_codebook**2, dim=1) - 2 * \
            torch.einsum('bd,dn->bn', z_flattened, rearrange(quant_codebook, 'n d -> d n'))

        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = F.embedding(min_encoding_indices, quant_codebook).view(z.shape)

        # compute loss for embedding
        if not self.legacy:
            quantization_loss = self.beta * torch.mean((z_q.detach()-z)**2) + \
                   torch.mean((z_q - z.detach()) ** 2)
        else:
            quantization_loss = torch.mean((z_q.detach()-z)**2) + self.beta * \
                   torch.mean((z_q - z.detach()) ** 2)

        # preserve gradients
        z_q = z + (z_q - z).detach()

        if self.remap is not None:
            min_encoding_indices = min_encoding_indices.reshape(z.shape[0],-1) # add batch axis
            min_encoding_indices = self.remap_to_used(min_encoding_indices)
            min_encoding_indices = min_encoding_indices.reshape(-1,1) # flatten

        if self.same_index_shape:
            min_encoding_indices = min_encoding_indices.view(z.shape[0], z.shape[1])
        z_q = self.project_out(z_q.view(z.shape))

        return z_q, min_encoding_indices, quantization_loss


import numpy as np
from helpers import LayerNorm, GELU, attention_weights, causal_mask, unit_vector

class MultiHeadAttention:
    def __init__(self,weights,prefix,heads):
        self.qkv_weight=weights[prefix+'.in_proj_weight']
        self.qkv_bias=weights[prefix+'.in_proj_bias']
        self.out_weight=weights[prefix+'.out_proj.weight']
        self.out_bias=weights[prefix+'.out_proj.bias']
        self.heads=heads
        self.width=self.out_bias.shape[0]
        self.head_dim=self.width//heads

    def split_heads(self,x):
        B,N,D=x.shape
        x=x.reshape(B,N,self.heads,self.head_dim)
        return x.transpose(0,2,1,3)

    def __call__(self,x,causal=False):
        qkv=x@self.qkv_weight.T+self.qkv_bias
        q=self.split_heads(qkv[:,:,:self.width])
        k=self.split_heads(qkv[:,:,self.width:2*self.width])
        v=self.split_heads(qkv[:,:,2*self.width:])
        scores=q@k.transpose(0,1,3,2)/np.sqrt(np.float32(self.head_dim))
        if causal:
            scores=causal_mask(scores)
        coefficients=attention_weights(scores)
        context=coefficients@v
        B,H,N,Dh=context.shape
        merged=context.transpose(0,2,1,3).reshape(B,N,H*Dh)
        return merged@self.out_weight.T+self.out_bias

class TransformerBlock:
    def __init__(self,weights,prefix,heads):
        self.norm1=LayerNorm(weights[prefix+'.ln_1.weight'],weights[prefix+'.ln_1.bias'])
        self.norm2=LayerNorm(weights[prefix+'.ln_2.weight'],weights[prefix+'.ln_2.bias'])
        self.attention=MultiHeadAttention(weights,prefix+'.attn',heads)
        self.fc1_weight=weights[prefix+'.mlp.c_fc.weight']
        self.fc1_bias=weights[prefix+'.mlp.c_fc.bias']
        self.fc2_weight=weights[prefix+'.mlp.c_proj.weight']
        self.fc2_bias=weights[prefix+'.mlp.c_proj.bias']
        self.gelu=GELU()

    def __call__(self,x,causal=False):
        update=self.attention(self.norm1(x),causal=causal)
        residual=x+update
        u=self.norm2(residual)
        hidden=u@self.fc1_weight.T+self.fc1_bias
        activated=self.gelu(hidden)
        mlp_update=activated@self.fc2_weight.T+self.fc2_bias
        return residual+mlp_update

class ImageEncoder:
    def __init__(self,weights):
        self.patch_weight=weights['visual.conv1.weight'].reshape(768,3072)
        self.cls=weights['visual.class_embedding'].reshape(1,1,768)
        self.position=weights['visual.positional_embedding']
        self.pre_norm=LayerNorm(weights['visual.ln_pre.weight'],weights['visual.ln_pre.bias'])
        self.blocks=[TransformerBlock(weights,'visual.transformer.resblocks.'+str(i),12)for i in range(12)]
        self.post_norm=LayerNorm(weights['visual.ln_post.weight'],weights['visual.ln_post.bias'])
        self.projection=weights['visual.proj']

    def patchify(self,images):
        B=images.shape[0]
        grid=224//32
        x=images.reshape(B,3,grid,32,grid,32)
        x=x.transpose(0,2,4,1,3,5)
        return x.reshape(B,grid*grid,3*32*32)

    def embed(self,images):
        patches=self.patchify(images)
        patch_tokens=patches@self.patch_weight.T
        cls=np.tile(self.cls,(images.shape[0],1,1))
        tokens=np.concatenate([cls,patch_tokens],axis=1)
        return tokens+self.position

    def encode_tokens(self,images):
        x=self.pre_norm(self.embed(images))
        for block in self.blocks:
            x=block(x)
        return x

    def __call__(self,images):
        tokens=self.encode_tokens(images)
        cls=tokens[:,0,:]
        output=self.post_norm(cls)@self.projection
        return unit_vector(output)
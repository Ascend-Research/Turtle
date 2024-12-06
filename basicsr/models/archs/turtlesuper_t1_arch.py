import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
from einops import rearrange
import math
from importlib import import_module


def make_model(opt):
    # Extracting model configuration directly from the opt dictionary without defaults
    model_config = {
        'inp_channels': opt['n_colors'],
        'out_channels': opt['n_colors'],
        'dim': opt['dim'],
        'Enc_blocks': opt['Enc_blocks'],
        'Middle_blocks': opt['Middle_blocks'],
        'Dec_blocks': opt['Dec_blocks'],
        'num_refinement_blocks': opt.get('num_refinement_blocks', 1),
        'ffn_expansion_factor': opt.get('ffn_expansion_factor', 1),
        'bias': opt.get('bias', False),
        'LayerNorm_type': opt.get('LayerNorm_type', 'WithBias'),
        'num_heads_blks': opt.get('num_heads_blks', [1,2,4,8]),
        'encoder1_attn_type1': opt['encoder1_attn_type1'],
        'encoder1_attn_type2': opt['encoder1_attn_type2'],
        'encoder2_attn_type1': opt['encoder2_attn_type1'],
        'encoder2_attn_type2': opt['encoder2_attn_type2'],
        'encoder3_attn_type1': opt['encoder3_attn_type1'],
        'encoder3_attn_type2': opt['encoder3_attn_type2'],
        'decoder1_attn_type1': opt['decoder1_attn_type1'],
        'decoder1_attn_type2': opt['decoder1_attn_type2'],
        'decoder2_attn_type1': opt['decoder2_attn_type1'],
        'decoder2_attn_type2': opt['decoder2_attn_type2'],
        'decoder3_attn_type1': opt['decoder3_attn_type1'],
        'decoder3_attn_type2': opt['decoder3_attn_type2'],
        'encoder1_ffw_type': opt['encoder1_ffw_type'],
        'encoder2_ffw_type': opt['encoder2_ffw_type'],
        'encoder3_ffw_type': opt['encoder3_ffw_type'],
        'decoder1_ffw_type': opt['decoder1_ffw_type'],
        'decoder2_ffw_type': opt['decoder2_ffw_type'],
        'decoder3_ffw_type': opt['decoder3_ffw_type'],
        'latent_attn_type1': opt['latent_attn_type1'],
        'latent_attn_type2': opt['latent_attn_type2'],
        'latent_attn_type3': opt['latent_attn_type3'],
        'latent_ffw_type': opt['latent_ffw_type'],
        'refinement_attn_type1': opt['refinement_attn_type1'],
        'refinement_attn_type2': opt['refinement_attn_type2'],
        'refinement_ffw_type': opt['refinement_ffw_type'],
        'use_both_input': opt['use_both_input'],
        'num_frames_tocache': opt.get('num_frames_tocache', 1),
        'num_heads': opt.get('num_heads', [1, 1, 1, 1])
    }
    return TurtleSuper_t1(**model_config)


def create_video_model(opt):
    module = import_module('basicsr.models.archs.turtle_super_t1_arch')
    model = module.make_model(opt)
    return model

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

    
def clipped_softmax(tensor, dim=-1):
    # Create a mask for zero elements
    zero_mask = tensor == 0
    
    # Apply the mask to ignore zero elements in the softmax computation
    # Set zero elements to `-inf` so that they become 0 after softmax
    masked_tensor = tensor.masked_fill(zero_mask, float('-inf'))
    
    # Compute softmax on the modified tensor
    softmaxed = F.softmax(masked_tensor, dim=dim)
    
    # Zero out `-inf` elements (which are now 0 due to softmax) if any original zeros existed
    softmaxed = softmaxed.masked_fill(zero_mask, 0)
    
    non_zero_softmaxed_sum = softmaxed.sum(dim=dim, keepdim=True)
    normalized_softmaxed = softmaxed / non_zero_softmaxed_sum
    
    return normalized_softmaxed
    
##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat//2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat*2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


##########################################################################
# Gated Feed-Forward Network
class GatedFeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(GatedFeedForward, self).__init__()

        hidden_features = int(dim*ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features*2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features*2, hidden_features*2, 
                                kernel_size=3, stride=1, 
                                padding=1, groups=hidden_features*2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

# Feed_Forward Network
class FeedForward(nn.Module):
    def __init__(self, c, FFN_Expand=2, drop_out_rate=0.):
        super(FeedForward, self).__init__()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, 
                               out_channels=ffn_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, 
                               out_channels=c, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = self.conv4(inp)
        x = F.gelu(x)
        x = self.conv5(x)
        x = self.dropout2(x)

        return x * self.gamma



##########################################################################
## History based Attentions


class FrameHistoryRouter(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache=1):
        """
        Initializes the FrameHistoryRouter module.

        Args:
            dim (int): The input dimension.
            num_heads (int): Number of attention heads.
            bias (bool): Whether to use bias in convolution layers.
            num_frames_tocache (int): Number of frames to cache for attention computation.
        """
        super(FrameHistoryRouter, self).__init__()
        self.dim = dim
        self.bias = bias

        self.num_heads= num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

        self.num_frames_tocache = num_frames_tocache
        
        
    def forward(self, x, k_cached=None, v_cached=None):
        """
        Forward pass of the FrameHistoryRouter.
        Given teh history states, it aggregates critical features for the restoration of the input frame

        Args:
            x (Tensor): Input tensor of shape (batch, channels, height, width).
            k_cached (Tensor, optional): Cached key tensor from previous frames.
            v_cached (Tensor, optional): Cached value tensor from previous frames.

        Returns:
            Tuple: Output tensor, and updated cached key and value tensors.
        """
        b, c, h, w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        # Concatenate cached key and value tensors if provided
        # Keys and values are concatenated with historical (cached) frames
        # This allows the attention mechanism to consider both the current and past frames.
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=2)
            v = torch.cat([v_cached, v], dim=2)
        
        # Calculating Attention scores
        # Query is from the current frame, while key and value are from both current and cached frames
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        num_cache_to_keep = int(self.num_frames_tocache*c/self.num_heads)
        return out, k[:, :, -num_cache_to_keep:, :], v[:, :, -num_cache_to_keep:, :]
    
class StateAlignBlock(nn.Module):
    def __init__(self, dim, num_heads, bias, num_frames_tocache, Scale_patchsize=1, plot_attn=False):
        """
        Initializes the StateAlignBlock module.
        
        Args:
            dim (int): The input dimension.
            num_heads (int): Number of attention heads.
            bias (bool): Whether to use bias in convolution layers.
            num_frames_tocache (int): Number of frames to cache for attention computation.
            Scale_patchsize (int): Scale patch size for windowing.
            plot_attn (bool): Whether to plot attention (used for visualization/debugging).
        """
        super(StateAlignBlock, self).__init__()
        self.num_heads= 1
        self.temperature = nn.Parameter(torch.ones(1, 1, 1))

        self.qk = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.qk_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)

        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.v_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim*1, bias=bias)

        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.num_frames_tocache = num_frames_tocache 
        self.window_size = 2 * Scale_patchsize


        self.k2 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.k2_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=self.window_size, stride=self.window_size, padding=1, groups=dim*2, bias=bias)
        self.q2 = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.q2_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=self.window_size, stride=self.window_size, padding=1, groups=dim*2, bias=bias)


    def zero_out_non_top_k(self, attn_matrix, k):
        """
        Zero out all but the top-k values in the attention matrix.
        
        Args:
            attn_matrix (Tensor): The attention matrix.
            k (int): Number of top elements to keep.
            
        Returns:
            Tensor: The modified attention matrix with only top-k values.
        """

        # Step 1: Get the top-k values and their indices for the last dimension
        a, n, b, c, c = attn_matrix.shape 
        _, topk_indices = torch.topk(attn_matrix, k=k, dim=-1)

        # Step 2: Create a mask of zeros
        mask = torch.zeros_like(attn_matrix)

        # Use these indices with scatter_ to update the mask. This time correctly broadcasting
        mask.scatter_(dim=-1, index=topk_indices, value=1)

        return attn_matrix * mask

    
    def positionalencoding2d(self, d_model, height, width):
        """
        Generates a 2D positional encoding for attention mechanism.
        
        Args:
            d_model (int): The dimension of the model.
            height (int): Height of the position encoding grid.
            width (int): Width of the position encoding grid.
            
        Returns:
            Tensor: A 2D positional encoding matrix of shape (d_model, height, width).
        """
        if d_model % 4 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dimension (got dim={:d})".format(d_model))
        pe = torch.zeros(d_model, height, width)
        # Each dimension use half of d_model
        d_model = int(d_model / 2)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                            -(math.log(10000.0) / d_model))
        pos_w = torch.arange(0., width).unsqueeze(1)
        pos_h = torch.arange(0., height).unsqueeze(1)
        pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
        pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
        pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

        return pe
        
    def create_local_attention_mask(self, h, w, n):
        """
        Creates a local attention mask So that every patch can only attend to neighboring patches.

        Args:
            h (int): Height of the attention mask.
            w (int): Width of the attention mask.
            n (int): Local attention range.

        Returns:
            Tensor: A binary mask tensor that determines the local attention scope.
        """
        y_coords, x_coords = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        coords = torch.stack([y_coords, x_coords], dim=-1).view(-1, 2)  # Shape: (hw, 2)
        distances = torch.cdist(coords.float(), coords.float(), p=1)  # Using L1 distance
        mask = distances <= n
        return mask
    
    # def forward(self, x, k_cached=None, v_cached=None):
    #     """
    #     Forward pass of the StateAlignBlock, which aligns the History states, Cached K,V of previous frames,
    #     with the current frame.

    #     Args:
    #         x (Tensor): Input tensor of shape (batch, channels, height, width).
    #         k_cached (Tensor, optional): Cached key tensor.
    #         v_cached (Tensor, optional): Cached value tensor.

    #     Returns:
    #         Tuple: Output tensor and cached key, value tensors.
    #     """
    #     b, c, h, w = x.shape
    #     head_dim = c//self.num_heads
        
    #     pos = self.positionalencoding2d(c, h, w)
    #     x_qk = x + pos.to(x.device)
        

    #     qk = self.qk_dwconv(self.qk(x_qk))
    #     q, k = qk.chunk(2, dim=1)   
    #     v = self.v_dwconv(self.v(x))

    #     # Rearrange inputs into windows and split into multiple heads in one step
    #     q = rearrange(q, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) d p1 p2', 
    #                   h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)
    #     k = rearrange(k, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) d p1 p2', 
    #                   h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)
    #     v = rearrange(v, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) (p1 p2 d)', 
    #                   h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)


    #     n = q.shape[3]
    #     q = rearrange(q, 'b 1 h_head n d p1 p2 -> (b h_head n) d p1 p2')
    #     q = self.q2_dwconv(self.q2(q))
    #     q = rearrange(q, '(b h_head n) d 1 1 -> b 1 h_head n d', b=b, h_head=self.num_heads, n=n)

    #     k = rearrange(k, 'b 1 h_head n d p1 p2 -> (b h_head n) d p1 p2')
    #     k = self.k2_dwconv(self.k2(k))
    #     k = rearrange(k, '(b h_head n) d 1 1 -> b 1 h_head n d', b=b, h_head=self.num_heads, n=n)


    #     H, W = q.shape[2], q.shape[3]
        
    #     q = torch.nn.functional.normalize(q, dim=-1)
    #     k = torch.nn.functional.normalize(k, dim=-1)
    #     # Concatenate cached k and v if they exist
    #     if k_cached is not None and v_cached is not None:
    #         k = torch.cat([k_cached, k], dim=1)
    #         v = torch.cat([v_cached, v], dim=1)

    #     curr_num_frames = k.shape[1]
    #     attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature

    #     # Create local attention mask
    #     mask = self.create_local_attention_mask(H, W, 4)
    #     mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    #     mask = mask.to(attn.device)
    #     attn1 = mask * attn

    #     # Zero out non-top-k attention scores
    #     attn2 = self.zero_out_non_top_k(attn, 5)

    #     # Combine the two attention mechanisms and apply softmax
    #     attn = (attn1 + attn2)/2
    #     attn = clipped_softmax(attn)

    #     del q
    #     out = torch.matmul(attn, v)
    #     del attn

    #     out = rearrange(out, 'b curr_num_frames h_head (h w) (p1 p2 d) -> (b curr_num_frames) (h_head d) (p1 h) (p2 w)',
    #             h_head=self.num_heads, p1=self.window_size, p2=self.window_size,
    #             h=h//self.window_size, w=w//self.window_size, d=head_dim)

    #     out = self.project_out(out)
    #     out = rearrange(out, '(b curr_num_frames) c h w -> b curr_num_frames c h w',
    #                     b=b, curr_num_frames=curr_num_frames)

    #     return out, k[:, -self.num_frames_tocache:, :, :, :], v[:, -self.num_frames_tocache:, :, :, :]

    def forward(self, x, k_cached=None, v_cached=None):
        b, c, h, w = x.shape
        head_dim = c//self.num_heads
        
        # pos = self.positionalencoding2d(c, h, w)
        # x_qk = x + pos.to(x.device)
        
        qk = self.qk_dwconv(self.qk(x))
        q, k = qk.chunk(2, dim=1)   
        v = self.v_dwconv(self.v(x))

        k = self.k2_dwconv(self.k2(k))
        q = self.q2_dwconv(self.q2(q))
        H, W = q.shape[2], q.shape[3]

        # pos = self.positionalencoding2d(c*2, q.shape[2], q.shape[3])
        # q = q + pos.to(x.device)
        # k = k + pos.to(x.device)


        # Rearrange inputs into windows and split into multiple heads in one step
        q = rearrange(q, 'b (h_head d) h w -> b 1 h_head (h w) d', 
                      h_head=self.num_heads, d=head_dim*2)
        k = rearrange(k, 'b (h_head d) h w -> b 1 h_head (h w) d', 
                      h_head=self.num_heads, d=head_dim*2)
        v = rearrange(v, 'b (h_head d) (p1 h) (p2 w) -> b 1 h_head (h w) (p1 p2 d)', 
                      h_head=self.num_heads, p1=self.window_size, p2=self.window_size, d=head_dim)

        
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        # Concatenate cached k and v if they exist
        if k_cached is not None and v_cached is not None:
            k = torch.cat([k_cached, k], dim=1)
            v = torch.cat([v_cached, v], dim=1)

        curr_num_frames = k.shape[1]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.temperature
        

        attn1 = self.zero_out_non_top_k(attn, 5)

        mask = self.create_local_attention_mask(H, W, 4)
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = mask.to(attn.device)
        attn2 = mask * attn

        attn = attn1 + attn2
        attn = clipped_softmax(attn)

        del q
        out = torch.matmul(attn, v)
        del attn

        out = rearrange(out, 'b curr_num_frames h_head (h w) (p1 p2 d) -> (b curr_num_frames) (h_head d) (p1 h) (p2 w)',
                h_head=self.num_heads, p1=self.window_size, p2=self.window_size,
                h=h//self.window_size, w=w//self.window_size, d=head_dim)

        out = self.project_out(out)
        out = rearrange(out, '(b curr_num_frames) c h w -> b curr_num_frames c h w',
                        b=b, curr_num_frames=curr_num_frames)

        return out, k[:, -self.num_frames_tocache:, :, :, :], v[:, -self.num_frames_tocache:, :, :, :]
    
class CausalHistoryModel(nn.Module):
    def __init__(self, dim, num_heads, bias, scale_patchsize, num_frames_tocache=1):
        super(CausalHistoryModel, self).__init__()

        #SAB
        self.spatial_aligner = StateAlignBlock(dim, num_heads, bias, num_frames_tocache, Scale_patchsize=scale_patchsize)

        #FHR
        self.ChanAttn = FrameHistoryRouter(dim, num_heads, bias)


        self.kv = nn.Conv2d(dim, dim*2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim*2, dim*2, kernel_size=3, stride=1, padding=1, groups=dim*2, bias=bias)
        self.num_heads = num_heads
        
    def forward(self, x, k_cached=None, v_cached=None):
        """
        Forward pass of the CausalHistoryModel.
        MotionCompensatedHistory = concat(SAB(History, Input), Input)
        Output = FHR(MotionCompensatedHistory, Input) + Input

        Args:
            x (Tensor): Input tensor of shape (batch, channels, height, width).
            k_cached (Tensor, optional): Cached key tensor from previous frames.
            v_cached (Tensor, optional): Cached value tensor from previous frames.

        Returns:
            Tuple: Output tensor from channel attention, and updated cached key and value tensors.
        """

        # Perform spatial alignment and get the updated cached key-value tensors
        x_spatial, k_tocache, v_tocache = self.spatial_aligner(x, k_cached, v_cached)

        cached_num_frames = x_spatial.shape[1]
        x_spatial =  rearrange(x_spatial, 'b cached_num_frames c h w -> (b cached_num_frames) c h w')
        
        # Compute key and value embeddings of aligned history
        kv = self.kv_dwconv(self.kv(x_spatial))
        k, v = kv.chunk(2, dim=1)   
        
        k = rearrange(k, '(b cached_num_frames) (head c) h w -> b head (cached_num_frames c) (h w)',
                       head=self.num_heads, cached_num_frames=cached_num_frames)
        v = rearrange(v, '(b cached_num_frames) (head c) h w -> b head (cached_num_frames c) (h w)',
                       head=self.num_heads, cached_num_frames=cached_num_frames)

        k = torch.nn.functional.normalize(k, dim=-1)

        #pass the input frame(x) and aligned history(k,v) to the Frame History router
        X_channel, _, _ = self.ChanAttn(x, k, v)

        return X_channel, k_tocache, v_tocache

##########################################################################
## Normal Attentions
class ChannelAttention(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(ChannelAttention, self).__init__()
        self.dim = dim
        self.bias = bias
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.qkv = nn.Conv2d(dim, dim*3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim*3, dim*3, kernel_size=3, stride=1, padding=1, groups=dim*3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        


    def forward(self, x, k_cached=None, v_cached=None):
        b,c,h,w = x.shape

        qkv = self.qkv_dwconv(self.qkv(x))
        q,k,v = qkv.chunk(3, dim=1)   
        
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)


        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out, None, None

class ReducedAttn(nn.Module):
    def __init__(self, c, DW_Expand=2.0, drop_out_rate=0.):
        super().__init__()
        dw_channel = int(c * DW_Expand)
        self.conv1 = nn.Conv2d(in_channels=c, 
                               out_channels=dw_channel, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        
        self.conv2 = nn.Conv2d(in_channels=dw_channel, 
                               out_channels=dw_channel, 
                               kernel_size=3, 
                               padding=1, 
                               stride=1, 
                               groups=dw_channel,
                               bias=True)
        
        self.conv3 = nn.Conv2d(in_channels=dw_channel, 
                               out_channels=c, 
                               kernel_size=1, 
                               padding=0, 
                               stride=1, 
                               groups=1, 
                               bias=True)
        

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp, k_cached=None, v_cached=None):
        x = self.conv1(inp)
        x = self.conv2(x)
        x = F.gelu(x)
        x = self.conv3(x)
        x = self.dropout1(x)
        return x * self.beta, None, None


##########################################################################
class TurtleAttnBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_heads=1, Scale_patchsize=1, attention_type='channel', FFW_type="GFFW", num_frames_tocache=1, plot_attn=False):
        super(TurtleAttnBlock, self).__init__()
        """
        This class gives you the freedom to design custom transfomer blocks with teh abilty to 
        specify the attention type and Feed forward fucntion.

        Args:
            dim (int): Dimension of the input feature space.
            ffn_expansion_factor (int): Expansion factor for the Feed Forward Network (FFN).
            bias (bool): Whether to use bias in the layers.
            LayerNorm_type (str): Type of Layer Normalization to use.
            num_blocks (int): Number of TurtleAttnBlocks to be used in the LevelBlock.
            attn_type1 (str): Attention type for the initial blocks. Default is "Channel".
            attn_type2 (str): Attention type for the last block. Default is "CHM".
            FFW_type (str): Type of Feed Forward Network. Default is "GFFW".
            num_frames_tocache (int): Number of frames to cache for the attention mechanism. Default is 1.
            num_heads (int): Number of attention heads in each TurtleAttnBlock. Default is 1.
            Scale_patchsize (int): patch size for the CHM's spatial alignemnt. Default is 1.

        Supported Attention types:
            - `FHR`: Frame History Router for utilizing past information without spatial alignment.
            - `CHM`: Causal History Model for utilizing past information.
            - `Channel`: Channel Attention for feature refinement.
            - `ReducedAttn`: Uses convolutions and gating, replacing Channel Attention to reduce computational complexity.
            - `NoAttn`: Only applies feed-forward layers without any attention mechanism.

        Supported FeedForward Types:
            - `FFW`: Feed Forward.
            - `GFFW`: Gated Feed Forward.

        """
        self.norm1 = LayerNorm(dim, LayerNorm_type)

        if attention_type == "Channel":
            self.attn = ChannelAttention(dim, num_heads, bias)
        elif attention_type == "ReducedAttn":
            self.attn = ReducedAttn(dim)
        elif attention_type == "FHR":  # Caches num_frames_tocache
            self.attn = FrameHistoryRouter(dim, num_heads, bias, num_frames_tocache)
        elif attention_type == "CHM": # Best march14
            self.attn = CausalHistoryModel(dim, num_heads, bias, Scale_patchsize, num_frames_tocache)
        elif attention_type == "NoAttn":
            self.attn = None
        else:
            print(attention_type, " Not defined")
            exit()
            
        self.norm2 = LayerNorm(dim, LayerNorm_type)

        if FFW_type == "GFFW":
            self.ffn = GatedFeedForward(dim, ffn_expansion_factor, bias)
        elif FFW_type == "FFW":
            self.ffn = FeedForward(dim)
        else:
            print(FFW_type, " Not defined")
            exit()

    def forward(self, x, k_cached=None, v_cached=None):
        if self.attn is None:
            return x + self.ffn(self.norm2(x)), None, None
        else:
            attn_out, k_tocahe, v_tocahe = self.attn(self.norm1(x), k_cached, v_cached)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
            return x, k_tocahe, v_tocahe

class LevelBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                  attn_type1="Channel", attn_type2="CHM", FFW_type="GFFW", num_frames_tocache=1, num_heads=1, Scale_patchsize=1):
        super(LevelBlock, self).__init__()
        """
        Initializes multiple `TurtleAttnBlock` layers where the last blocks use `CHM` or 'FHR' 
        attention to handle historical dependencies, while the initial layers use a different
        attention type like `Channel` or `ReducedAttn`.

        Args:
            dim (int): Dimension of the input feature space.
            ffn_expansion_factor (int): Expansion factor for the Feed Forward Network (FFN).
            bias (bool): Whether to use bias in the layers.
            LayerNorm_type (str): Type of Layer Normalization to use.
            num_blocks (int): Number of TurtleAttnBlocks to be used in the LevelBlock.
            attn_type1 (str): Attention type for the initial blocks. Default is "Channel".
            attn_type2 (str): Attention type for the last block. Default is "CHM".
            FFW_type (str): Type of Feed Forward Network. Default is "GFFW".
            num_frames_tocache (int): Number of frames to cache for the attention mechanism. Default is 1.
            num_heads (int): Number of attention heads in each TurtleAttnBlock. Default is 1.
            Scale_patchsize (int): patch size for the CHM's spatial alignemnt. Default is 1.

        Attention type options used:
        - `FHR`: Frame History Router for utilizing past information without spatial alignment.
        - `CHM`: Causal History Model for utilizing past information.
        - `Channel`: Channel Attention for feature refinement.
        - `ReducedAttn`: Uses convolutions and gating, replacing Channel Attention to reduce computational complexity.
        - `NoAttn`: Only applies feed-forward layers without any attention mechanism.
        """
        self.num_blocks = num_blocks
        Block_list = []
            
        for _ in range(num_blocks - 1):
            Block_list.append(TurtleAttnBlock(dim=dim, num_heads=num_heads, 
                             ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                             LayerNorm_type=LayerNorm_type, attention_type=attn_type1, FFW_type=FFW_type, num_frames_tocache=num_frames_tocache, Scale_patchsize=Scale_patchsize))
            
        Block_list.append(TurtleAttnBlock(dim=dim, num_heads=num_heads, 
                        ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                        LayerNorm_type=LayerNorm_type, attention_type=attn_type2, FFW_type=FFW_type, num_frames_tocache=num_frames_tocache, Scale_patchsize=Scale_patchsize) )
            
        self.transformer_blocks = nn.ModuleList(Block_list)

    def forward(self, x, k_cached=None, v_cached=None):
        for i in range(self.num_blocks - 1):
            x, _, _ = self.transformer_blocks[i](x)

        # Pass k_cached and v_cached to the last block
        out1, k_tocahe, v_tocahe = self.transformer_blocks[-1](x, k_cached, v_cached)
        if k_tocahe != None:
            return out1, k_tocahe, v_tocahe
        else:
            return out1, None, None
      
class LatentCacheBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, LayerNorm_type, num_blocks,
                  attn_type1="FHR", attn_type2="Channel", attn_type3="FHR", FFW_type="GFFW", num_frames_tocache=1, num_heads=1):
        super(LatentCacheBlock, self).__init__()
        """
        Initializes multiple `TurtleAttnBlock` layers where the first and last blocks use `CHM` or 'FHR' 
        attention to handle historical dependencies, while the intermediate layers use a different
        attention type like `Channel` or `ReducedAttn`.

        Args:
            dim (int): Dimension of the input feature space.
            ffn_expansion_factor (int): Expansion factor for the Feed Forward Network (FFN).
            bias (bool): Whether to use bias in the layers.
            LayerNorm_type (str): Type of Layer Normalization to use.
            num_blocks (int): Number of TurtleAttnBlocks to be used in the LevelBlock.
            attn_type1 (str): Attention type for the latent middle blocks. Default is "Channel".
            attn_type2 (str): Attention type for the first block. Default is "CHM".
            attn_type2 (str): Attention type for the last block. Default is "CHM".
            FFW_type (str): Type of Feed Forward Network. Default is "GFFW".
            num_frames_tocache (int): Number of frames to cache for the attention mechanism. Default is 1.
            num_heads (int): Number of attention heads in each TurtleAttnBlock. Default is 1.
            Scale_patchsize (int): patch size for the CHM's spatial alignemnt. Default is 1.

        Attention type options used:
        - `FHR`: Frame History Router for utilizing past information without spatial alignment.
        - `CHM`: Causal History Model for utilizing past information.
        - `Channel`: Channel Attention for feature refinement.
        - `ReducedAttn`: Uses convolutions and gating, replacing Channel Attention to reduce computational complexity.
        - `NoAttn`: Only applies feed-forward layers without any attention mechanism.
        """
        self.num_blocks = num_blocks
        Block_list = []
        if self.num_blocks < 2:
            print("LatentCacheBlock should have more than 2 layers")
            exit()

        Block_list.append(TurtleAttnBlock(dim=dim, num_heads=num_heads, 
                        ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                        LayerNorm_type=LayerNorm_type, attention_type=attn_type1, FFW_type=FFW_type, num_frames_tocache= num_frames_tocache, plot_attn=True) )
        
        if self.num_blocks > 2:
            for _ in range(self.num_blocks - 2):
                Block_list.append(TurtleAttnBlock(dim=dim, num_heads=num_heads, 
                                ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                                LayerNorm_type=LayerNorm_type, attention_type=attn_type2, FFW_type=FFW_type, num_frames_tocache= num_frames_tocache) )
            
        Block_list.append(TurtleAttnBlock(dim=dim, num_heads=num_heads, 
                        ffn_expansion_factor=ffn_expansion_factor, bias=bias, 
                        LayerNorm_type=LayerNorm_type, attention_type=attn_type3, FFW_type=FFW_type, num_frames_tocache= num_frames_tocache, plot_attn=True) )
            
        self.transformer_blocks = nn.ModuleList(Block_list)

    def forward(self, x, k1_cached=None, v1_cached=None, k2_cached=None, v2_cached=None):
        out, k1_tocahe, v1_tocahe = self.transformer_blocks[0](x, k1_cached, v1_cached)

        if self.num_blocks > 2:
            for i in range(1, self.num_blocks - 1):
                out, _, _ = self.transformer_blocks[i](out)

        out, k2_tocahe, v2_tocahe = self.transformer_blocks[-1](out, k2_cached, v2_cached)

        return out, k1_tocahe, v1_tocahe, k2_tocahe, v2_tocahe


##########################################################################
class TurtleSuper_t1(nn.Module):
    def __init__(self,
        inp_channels,
        out_channels,
        dim,
        Enc_blocks,
        Middle_blocks,
        Dec_blocks,
        num_heads, 
        num_refinement_blocks,
        ffn_expansion_factor,
        bias,
        LayerNorm_type,
        num_heads_blks,

        # Encoder attention types
        encoder1_attn_type1, encoder1_attn_type2,
        encoder2_attn_type1, encoder2_attn_type2,
        encoder3_attn_type1, encoder3_attn_type2,

        # Decoder attention types
        decoder1_attn_type1, decoder1_attn_type2,
        decoder2_attn_type1, decoder2_attn_type2,
        decoder3_attn_type1, decoder3_attn_type2,

        # FFW types for each encoder and decoder level
        encoder1_ffw_type, encoder2_ffw_type, encoder3_ffw_type,
        decoder1_ffw_type, decoder2_ffw_type, decoder3_ffw_type,

        # Latent
        latent_attn_type1, latent_attn_type2, latent_attn_type3, latent_ffw_type,

        # Refinement
        refinement_attn_type1, refinement_attn_type2, refinement_ffw_type,


        use_both_input,
        num_frames_tocache):
        super(TurtleSuper_t1, self).__init__()
        if use_both_input:
            inp_channels *= 2
        self.use_both_input = use_both_input
        self.num_heads = num_heads
        # 4x upsampling
        self.upsample_4x = nn.Upsample(scale_factor=4,
                                       mode="bilinear")
        
        self.input_projection = nn.Conv2d(inp_channels, 
                                     dim, kernel_size=3, 
                                     stride=1, padding=1, 
                                     bias=bias) 
        
        # Encoder Levels
        self.encoder_level1 = LevelBlock(dim=dim, bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                          LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[0],
                                          attn_type1=encoder1_attn_type1, attn_type2=encoder1_attn_type2,
                                          FFW_type=encoder1_ffw_type, num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[0])
        
        self.down1_2 = Downsample(dim)  # From Level 1 to Level 2
        self.encoder_level2 = LevelBlock(dim=int(dim*2**1), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                          LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[1],
                                          attn_type1=encoder2_attn_type1, attn_type2=encoder2_attn_type2,
                                          FFW_type=encoder2_ffw_type, num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[1])
        
        self.down2_3 = Downsample(int(dim*2**1))  # From Level 2 to Level 3
        self.encoder_level3 = LevelBlock(dim=int(dim*2**2), bias=bias, ffn_expansion_factor=ffn_expansion_factor,
                                          LayerNorm_type=LayerNorm_type, num_blocks=Enc_blocks[2],
                                          attn_type1=encoder3_attn_type1, attn_type2=encoder3_attn_type2, 
                                          FFW_type=encoder3_ffw_type, num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[2])

        # Middle block
        self.down3_4 = Downsample(int(dim*2**2)) ## From Level 3 to Level 4
        self.latent = LatentCacheBlock(dim=int(dim*2**3), ffn_expansion_factor=ffn_expansion_factor,
                                       bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Middle_blocks,
                                       attn_type1=latent_attn_type1, attn_type2=latent_attn_type2, 
                                       attn_type3=latent_attn_type3, FFW_type=latent_ffw_type, 
                                       num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[3])

        # Decoder Levels
        self.up4_3 = Upsample(int(dim*2**3))  # From Level 4 to Level 3
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = LevelBlock(dim=int(dim*2**2), ffn_expansion_factor=ffn_expansion_factor,
                                         bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[0],
                                         attn_type1=decoder1_attn_type1, attn_type2=decoder1_attn_type2, 
                                         FFW_type=decoder1_ffw_type, num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[2], Scale_patchsize=2)
        
        self.up3_2 = Upsample(int(dim*2**2)) ## From Level 3 to Level 2
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = LevelBlock(dim=int(dim*2**1), ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[1],
                                           attn_type1=decoder2_attn_type1, attn_type2=decoder2_attn_type2, 
                                           FFW_type=decoder2_ffw_type, num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[1], Scale_patchsize=4)
        
        
        self.up2_1 = Upsample(int(dim*2**1))  ## From Level 2 to Level 1 
        self.reduce_chan_level1 = nn.Conv2d(int(dim*2**1), int(dim*1), kernel_size=1, bias=bias)
        self.decoder_level1 = LevelBlock(dim=int(dim*1), ffn_expansion_factor=ffn_expansion_factor,
                                           bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=Dec_blocks[2],
                                           attn_type1=decoder3_attn_type1, attn_type2=decoder3_attn_type2, 
                                           FFW_type=decoder3_ffw_type, num_frames_tocache=2, num_heads=self.num_heads[0], Scale_patchsize=8)
        
        # Refinement Block
        self.refinement = LevelBlock(dim=int(dim*1), ffn_expansion_factor=ffn_expansion_factor,
                                     bias=bias, LayerNorm_type=LayerNorm_type, num_blocks=num_refinement_blocks,
                                     attn_type1=refinement_attn_type1, attn_type2=refinement_attn_type2, 
                                     FFW_type=refinement_ffw_type, num_frames_tocache=num_frames_tocache, num_heads=self.num_heads[0])
        
        self.ending = nn.Conv2d(in_channels=int(dim*1),
                                 out_channels=out_channels,
                                 kernel_size=3,
                                 padding=1,
                                 stride=1,
                                 groups=1,
                                 bias=True)
        
        self.padder_size = (2**3)*4

    def forward(self, inp_img_, k_cached=None, v_cached=None):
        B, _, C, H, W = inp_img_.shape
        H, W = H*4, W*4 # scale factor
        if k_cached == None:
            k_cached = [None] * 8
            v_cached = [None] * 8

        k_to_cache = []
        v_to_cache = []

        if self.use_both_input:
            previous, current = inp_img_[:, 0, :, :, :], inp_img_[:, 1, :, :, :]
            inp_img = torch.cat([previous, 
                                current], dim=1)
            # do upsampling since it is superresolution task.
            inp_img = self.upsample_4x(inp_img)
            inp_img = self.check_image_size(inp_img)
        else:
            inp_img = inp_img_[:, 1, :, :, :]
            # do upsampling since it is superresolution task.
            inp_img = self.upsample_4x(inp_img)
            inp_img = self.check_image_size(inp_img)
            current = inp_img

        inp_enc_level1 = self.input_projection(inp_img.float())
        
        out_enc_level1, k1_tocahe, v1_tocahe = self.encoder_level1(inp_enc_level1, 
                                                                   k_cached[0], 
                                                                   v_cached[0])
        k_to_cache.append(k1_tocahe)
        v_to_cache.append(v1_tocahe)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2, k2_tocahe, v2_tocahe = self.encoder_level2(inp_enc_level2, 
                                                                   k_cached[1], 
                                                                   v_cached[1])
        k_to_cache.append(k2_tocahe)
        v_to_cache.append(v2_tocahe)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3, k3_tocahe, v3_tocahe = self.encoder_level3(inp_enc_level3, 
                                                                   k_cached[2], 
                                                                   v_cached[2]) 
        k_to_cache.append(k3_tocahe)
        v_to_cache.append(v3_tocahe)

        inp_enc_level4 = self.down3_4(out_enc_level3)        

        latent, k4_tocahe, v4_tocahe, k5_tocahe, v5_tocahe = self.latent(inp_enc_level4, 
                                                                         k_cached[3], v_cached[3],
                                                                         k_cached[4], v_cached[4]) 
        k_to_cache.append(k4_tocahe)
        k_to_cache.append(k5_tocahe)

        v_to_cache.append(v4_tocahe)
        v_to_cache.append(v5_tocahe)


        inp_dec_level3 = self.up4_3(latent)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3, k6_tocahe, v6_tocahe = self.decoder_level3(inp_dec_level3, 
                                                                   k_cached[5], 
                                                                   v_cached[5]) 

        k_to_cache.append(k6_tocahe)
        v_to_cache.append(v6_tocahe)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2, k7_tocahe, v7_tocahe = self.decoder_level2(inp_dec_level2, 
                                                                   k_cached[6], 
                                                                   v_cached[6]) 
        k_to_cache.append(k7_tocahe)
        v_to_cache.append(v7_tocahe)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        out_dec_level1, k8_tocahe, v8_tocahe = self.decoder_level1(inp_dec_level1, 
                                                                   k_cached[7], 
                                                                   v_cached[7])

        k_to_cache.append(k8_tocahe)
        v_to_cache.append(v8_tocahe)

        out_dec_level1, _, _ = self.refinement(out_dec_level1)
        
        ending = self.ending(out_dec_level1) + current

        return (ending[:, :, :H, :W],
                k_to_cache,
                v_to_cache)

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

import time
def measure_inference_speed(model, data, max_iter=200, log_interval=50):
    model.eval()

    # the first several iterations may be very slow so skip them
    num_warmup = 5
    pure_inf_time = 0
    fps = 0

    # benchmark with 2000 image and take the average
    for i in range(max_iter):

        torch.cuda.synchronize()
        start_time = time.perf_counter()

        with torch.no_grad():
            model(*data)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start_time

        if i >= num_warmup:
            pure_inf_time += elapsed
            if (i + 1) % log_interval == 0:
                fps = (i + 1 - num_warmup) / pure_inf_time
                print(
                    f'Done image [{i + 1:<3}/ {max_iter}], '
                    f'fps: {fps:.1f} img / s, '
                    f'times per image: {1000 / fps:.1f} ms / img',
                    flush=True)

        if (i + 1) == max_iter:
            fps = (i + 1 - num_warmup) / pure_inf_time
            print(
                f'Overall fps: {fps:.1f} img / s, '
                f'times per image: {1000 / fps:.1f} ms / img',
                flush=True)
            break
    return fps

if __name__ == '__main__':
    from ptflops import get_model_complexity_info
    from basicsr.utils.options import parse

    opt = parse(
                "path_to_options/options/Turtle_desnow.yml",
                 is_train=True)

    model = create_video_model(opt)

    inp_shape = (2, 3, 256, 256)
    macs, params = get_model_complexity_info(model, inp_shape, 
                                             verbose=False, 
                                             print_per_layer_stat=False)
    print(f"MACs: {macs}")
    print(f"Params: {params}")

    # device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    data = torch.randn((1, *inp_shape))
    print(device)
    measure_inference_speed(model.to(device), (data.to(device),), max_iter=500, log_interval=50)

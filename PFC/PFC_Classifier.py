import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from icecream import ic


class Downsample_Fusion_map(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """
        Parameters
        ----------
        in_channels : int
            number of input features.
        out_channels : int
            number of output features.
        kernel_size : int
            size of convolution kernel.
        stride : int

        Returns
        -------
        Tensor
            Downsampled fusion feature map.

        """

        super(Downsample_Fusion_map, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        return x


class ConcatFusionMaps(nn.Module):
    def __init__(self, in_channels, out_channels):
        """

        Parameters
        ----------
        in_channels : int
            Number of input channels for the first downsampling layer.
        out_channels : list of int
            List containing the number of output channels for each downsampling layer.

        Returns
        ----------
            None.
        """
        super(ConcatFusionMaps, self).__init__()
        self.downsample_convs = nn.ModuleList()

        number_layers = len(out_channels)
        # we do not apply downsampling on the last fused feature map
        for i in range(number_layers-1):
            stirde_value = 2**(number_layers - 1 - i)
            in_ch = in_channels if i == 0 else out_channels[i-1]
            kernel_szie = 3 if i < number_layers-2 else 1
            self.downsample_convs.append(Downsample_Fusion_map(
                in_ch, in_ch, kernel_szie, stirde_value))

    def forward(self, fused_feature_map):
        downsampled_feature_maps = []
        for i in range(len(fused_feature_map) - 1):
            downsampled_feature_maps.append(
                (self.downsample_convs[i])(fused_feature_map[i]))

        # -- concatenate downsampled fused feature maps
        # add the last fused map to the downsampled ones
        downsampled_feature_maps.append(fused_feature_map[-1])

        downsampled_feature_maps = torch.cat(downsampled_feature_maps, dim=1)
        return downsampled_feature_maps


# -- Since the sptial size of the input patch, unlike the swin, is not big,
# there is no need for patch embedding
class embeding(nn.Module):
    def __init__(self, in_channels=3, embed_features=16):
        """
         Parameters
         ----------
         in_channels : int
             Number of input channels for the first downsampling layer.
         embed_features : int
             number of embedding features.

         Returns
         ----------
             embedded features.

        """
        super(embeding, self).__init__()

        self.embed_features = embed_features

        # -- a convolutional operator to increase number of initial features
        self.increasing_features = nn.Conv2d(in_channels=in_channels,
                                             out_channels=embed_features,
                                             kernel_size=1,
                                             stride=1,
                                             padding=0)

        self.norm = nn.BatchNorm2d(embed_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.norm(self.increasing_features(x))


#-- Reduction
class reduction (nn.Module):
    def __init__(self, in_channels, out_channel):
        """
        This calss is used to downsample feature maps
         Parameters
         ----------
         in_channels : int
             Number of input channels for the first downsampling layer.
         embed_features : int
             number of output features.

         Returns
         ----------
             downsampled feature maps.

        """

        super(reduction, self).__init__()

        # -- in_channels is the number of channels after concatenation

        self.reduction = nn.Conv2d(in_channels=in_channels,
                                   out_channels=out_channel,
                                   kernel_size=3,
                                   stride=2,
                                   padding=1)

        self.norm = nn.BatchNorm2d(out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : Tensor [B, C, H, W]
            Output of the fusion

        Returns
        -------
        Tensor [B, C', H/2, W/2]
            Downsampled and being ready to feed to next layer.

        """
        output = self.norm(self.reduction(x))

        return output


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.4):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """

    Parameters
    ----------
    x : Tensor [B, C, H, W]
        Output of the fusion
    window_size: int

    Returns
    -------
    windows: Tensor [num_windows*B, window_size, window_size, C]

    """
    if len(window_size) > 1:  # windowsize has been tuples into an array [windowsize, windowsize]
        window_size = window_size[0]

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
   Parameters
    ----------
    windows : torch.Tensor
        Tensor of shape (num_windows*B, window_size, window_size, C).
    window_size : int
        Window size.
    H : int
        Height of the image.
    W : int
        Width of the image.

    Returns
    -------
    x : torch.Tensor
        Tensor of shape (B, H, W, C).
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PFCAttention(nn.Module):
    """ 
   Parameters
    ----------
    dim : int
        Number of input channels.
    window_size : tuple[int]
        The height and width of the window.
    num_heads : int
        Number of attention heads.
    feature_map_size : list
        Spatial size of the input feature map.
    qkv_bias : bool, optional
        If True, add a learnable bias to query, key, value. Default is True.
    qk_scale : float or None, optional
        Override default qk scale of head_dim ** -0.5 if set. Default is None.
    attn_drop : float, optional
        Dropout ratio of attention weight. Default is 0.0.
    proj_drop : float, optional
        Dropout ratio of output. Default is 0.0.

    Returns
     ----------
     PFC_Attn

    """

    def __init__(self, dim, window_size, num_heads,
                 feature_map_size, qkv_bias=True,
                 qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        #self.feature_map_size = feature_map_size

        self.norm_kv = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3*dim, bias=qkv_bias)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.dim)

        # -- used by both local and global attention
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # -- Local Attention
        # -- define a parameter table of relative position bias for local attention
        # -- To understand how relative position is calculated, read the comments on swin_transformer.py
        self.local_relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        local_coords_h = torch.arange(self.window_size[0])
        local_coords_w = torch.arange(self.window_size[1])
        local_coords = torch.stack(torch.meshgrid(
            [local_coords_h, local_coords_w]))

        local_coords_flatten = torch.flatten(local_coords, 1)

        local_relative_coords = local_coords_flatten[:,
                                                     :, None] - local_coords_flatten[:, None, :]
        local_relative_coords = local_relative_coords.permute(
            1, 2, 0).contiguous()
        # shift to start from 0
        local_relative_coords[:, :, 0] += self.window_size[0] - 1
        local_relative_coords[:, :, 1] += self.window_size[1] - 1
        local_relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        local_relative_position_index = local_relative_coords.sum(-1)
        self.register_buffer("local_relative_position_index",
                             local_relative_position_index)
        trunc_normal_(self.local_relative_position_bias_table, std=.02)

        # -- Project the concatenated headers
        self.local_proj = nn.Linear(dim, dim)

        # -- Global Attention
        # before calling WindowAttention, we must be sure that the spatial
        # size ofthe input feature map is divisible by window_size.
        self.total_windows = (
            feature_map_size[0]//window_size[0],  feature_map_size[1]//window_size[1])

        # -- from focal paper
        self.relative_position_bias_table_to_windows = nn.Parameter(
            torch.zeros((window_size[0] + self.total_windows[0] - 1) *
                        (window_size[1] + self.total_windows[1] - 1), num_heads))  #
        trunc_normal_(self.relative_position_bias_table_to_windows, std=.02)

        coords_h_k = torch.arange(self.total_windows[0])
        coords_w_k = torch.arange(self.total_windows[1])
        coords_k = torch.stack(torch.meshgrid(
            [coords_h_k, coords_w_k]))  # 2, Wh_k, Ww_k
        coords_flatten_k = torch.flatten(coords_k, 1)  # 2, Wh_k*Ww_k

        # -- This line computes the pair-wise differences between the row and
        # column indices of each position in the local and global windows
        # 2, Wh_q*Ww_q, Wh_k*Ww_k
        relative_coords = local_coords_flatten[:,
                                               :, None] - coords_flatten_k[:, None, :]
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh_q*Ww_q, Wh_k*Ww_k, 2

        # -- shift to start from zero
        relative_coords[:, :, 0] += self.total_windows[0] - 1  # relative rows
        relative_coords[:, :, 1] += self.total_windows[1] - 1  # relative cols

        # -- scales the row indices of the relative position tensor so that they can be used as
        #   indices into the relative positional embedding tensor.
        relative_coords[:, :,
                        0] *= (self.window_size[1] + self.total_windows[1]) - 1

        # -- computes the relative position index for each token
        # Wh_q*Ww_q, Wh_k*Ww_k
        global_relative_position_index = relative_coords.sum(-1)

        self.register_buffer("global_relative_position_index",
                             global_relative_position_index)

        # -- fully coonected layer for reducing size of K and V
        self.pool_layer_k = nn.Linear(
            self.window_size[0] * self.window_size[1], 1)
        self.pool_layer_k.weight.data.fill_(
            1. / self.window_size[0] * self.window_size[1])
        self.pool_layer_k.bias.data.fill_(0)

        self.pool_layer_v = nn.Linear(
            self.window_size[0] * self.window_size[1], 1)
        self.pool_layer_v.weight.data.fill_(
            1. / (self.window_size[0] * self.window_size[1]))
        self.pool_layer_v.bias.data.fill_(0)

        # -- Project the concatenated headers
        self.global_proj = nn.Linear(dim, dim)

        # -- Mapping global and local attention for combining together
        self.pfcmapping = nn.Linear(2 * dim, dim, bias=qkv_bias)

    def forward(self, x):
        """
        Parameters
        ----------
            x: Tensfor
                input features with shape of (B, H, W, C)

        """
        B, H, W, C = x.shape

        # -- partition the input feature maps into non-overlapping windows
        # (nW*B, windowsize, windowsize, C)
        x_windows = window_partition(x, self.window_size)

        # -- merge pixels inside of a window
        # (nW*B, window_size*window_size, C)
        x_windows = x_windows.view(-1,
                                   self.window_size[0] * self.window_size[1], C)

        B_, N, C = x_windows.shape

        # -- Calculate local attention which is attention iside of each window

        qkv = self.qkv(x)  # (B_, N, 3*C)
        # -- Note: C must be divisible by the number of heads
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        # (3, B_, nHeads, N, C/nHeads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        # -- Seperate Query, Key, and Value matrices
        # (B_, nHeads, N, C/nHeads)
        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]

        # -- local attention
        # (B_, nHeads, N, N)
        local_attn = (q_local * self.scale) @ (k_local.transpose(-2, -1))

        # -- local relative position bias
        local_relative_position_bias = self.local_relative_position_bias_table[self.local_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # N, N, nHeads

        local_relative_position_bias = local_relative_position_bias.permute(
            2, 0, 1).contiguous()  # nHeads, N, N

        # -- Add the local relative position bias to the local attention
        local_attn = local_attn + \
            local_relative_position_bias.unsqueeze(0)  # (B_, nHeads, N, N)

        # -- Apply softmax on the local attention
        local_attn = self.softmax(local_attn)  # (B_, nHeads, N, N)

        # -- Apply dropout
        local_attn = self.attn_drop(local_attn)

        # -- Multiply by Value matrix
        local_attn = (local_attn @ v_local).transpose(1,
                                                      2).reshape(B_, N, C).contiguous()

        # -- Multihead weight
        local_attn = self.local_proj(local_attn)

        # -- Apply dropout on the projected local attention
        local_attn = self.proj_drop(local_attn)

        # -- Calculate global attention which is attention iside of each window

        # -- global Keys
        nWs = B_//B  # number of windows in a partitioned feature map

        # -- Seperate partitioned windows from batch
        k_local = k_local.reshape(
            B, nWs, self.num_heads, N, C // self.num_heads).contiguous()

        # -- Transpose the pixels inside of a local window to the last dimension to apply a linear projection
        # (B, nWs, nHeads, C/nHeads, N)  N is the pixels inside of a local windows
        k_local = k_local.transpose(-2, -1).contiguous()
        # -- Apply linear projection on each local window
        K_global = self.pool_layer_k(k_local)  # (B, nWs, nHeads, C/nHeads, 1)
        # -- swip partitioned windows to the last dimension to have a correct repetition
        # (B, 1, nHeads, nWs, C/nHeads)
        K_global = K_global.permute(0, 4, 2, 1, 3).contiguous()

        # -- repeat the global windows to the number of partitioned windows. becuase global attention is calculated
        #   between local q and each global windows
        # (B, nWs, nHeads, nWs, C/nHeadss)
        K_global = K_global.repeat(1, nWs, 1, 1, 1)

        # -- Finally, combined the number of partitioned windows with batches to have similar shape as local Q
        # (B_, nHeads, nWs, C/nHeadss)
        K_global = K_global.reshape(-1, self.num_heads,
                                    nWs,  C // self.num_heads).contiguous()

        # -- global Value
        # -- Seperate partitioned windows from batch
        v_local = v_local.reshape(
            B, nWs, self.num_heads, N, C // self.num_heads).contiguous()

        # -- Transpose the pixels inside of a local window to the last dimension to apply a linear projection
        v_local = v_local.transpose(-2, -1).contiguous()

        # -- Apply linear projection on each local window
        V_global = self.pool_layer_k(v_local)

        # -- swip partitioned windows to the last dimension to have a correct repetition
        V_global = V_global.permute(0, 4, 2, 1, 3).contiguous()

        # -- repeat the global windows
        V_global = V_global.repeat(1, nWs, 1, 1, 1)

        # -- Finally, combined the number of partitioned windows with batches to have similar shape as local Q
        V_global = V_global.reshape(-1, self.num_heads,
                                    nWs,  C // self.num_heads).contiguous()

        # -- global attention
        # (B_, nHeads, N, total windows * total windows)
        global_attn = (q_local * self.scale) @ (K_global.transpose(-2, -1))

        # -- relative positional bias between K-global windows and local q windows
        global_relative_position_bias = self.relative_position_bias_table_to_windows[self.global_relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.total_windows[0] * self.total_windows[1], -1)  # N, total windows * total windows, nHeads

        global_relative_position_bias = global_relative_position_bias.permute(
            2, 0, 1).contiguous()  # nHeads, N, total windows * total windows

        # -- Add relative positional bias to global_attn
        # (B_, nHeads, N, total windows * total windows)
        global_attn = global_attn + global_relative_position_bias.unsqueeze(0)

        # -- Apply softmax on the local attention
        # (B_, nHeads, N, total windows * total windows)
        global_attn = self.softmax(global_attn)

        # -- Apply dropout
        global_attn = self.attn_drop(global_attn)

        # -- Multiply by Value matrix
        global_attn = (global_attn @ V_global).transpose(1,
                                                         2).reshape(B_, N, C).contiguous()  # (B_, N,C)

        # -- Multihead weight
        global_attn = self.global_proj(global_attn)

        # -- Apply dropout on the projected global attention
        global_attn = self.proj_drop(global_attn)

        # -- PFC Attention
        PFC_Attn = torch.cat((local_attn, global_attn),
                             dim=-1)  # (B_, N, 2 *C)

        # -- Apply Mapping
        PFC_Attn = self.pfcmapping(PFC_Attn)  # (B_, N, C)

        return PFC_Attn


class PFClBlock (nn.Module):
    """
    Parameters
    ----------
    in_channels : int
        Number of input channels.
    feature_map_size : tuple[int]
        Resolution of the input feature map.
    num_heads : int
        Number of attention heads.
    window_size : int
        Window size.
    mlp_ratio : float
        Ratio of mlp hidden dim to embedding dim.
    qkv_bias : bool, optional
        If True, add a learnable bias to query, key, value. Default is True.
    qk_scale : float or None, optional
        Override default qk scale of head_dim ** -0.5 if set. Default is None.
    drop_path : float, optional
        Stochastic depth rate. Default is 0.0.
    mlp_drop : float, optional
        Dropout rate. Default is 0.0.
    attn_drop : float, optional
        Attention dropout rate. Default is 0.0.
    proj_drop : float, optional
        Dropout ratio of output. Default is 0.0.
    act_layer : nn.Module, optional
        Activation layer. Default is nn.GELU.
    norm_layer : nn.Module, optional
        Normalization layer. Default is nn.LayerNorm.

    Returns
    ----------
    x: Tensor
        output of the PFC layer

    """

    def __init__(self, in_channels, feature_map_size, num_heads,
                 window_size=4, mlp_ratio=4.,
                 qkv_bias=True, qk_scale=None, drop_path=0.,
                 mlp_drop=0.2, attn_drop=0.2, proj_drop=0.2,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):

        super().__init__()
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.feature_map_size = feature_map_size
        # -- Make sure that the window size of partitioning is not larger than the
        if min(self.feature_map_size) <= self.window_size:
            # -- if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.feature_map_size)

        # -- First normaliztion
        self.norm1 = norm_layer(in_channels)

        # -- PFC Attention
        self.PFC_attn = PFCAttention(in_channels,
                                     window_size=to_2tuple(self.window_size),
                                     num_heads=num_heads,
                                     feature_map_size=feature_map_size,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     attn_drop=attn_drop,
                                     proj_drop=proj_drop)

        # -- drop path to reduce overfitting  posibility

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()

        # -- Second layer normalization
        self.norm2 = norm_layer(in_channels)

        #-- MLP
        mlp_hidden_dim = int(in_channels * mlp_ratio)
        self.mlp = Mlp(in_features=in_channels,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=mlp_drop)

    def forward(self, x):
        """


        Parameters
        ----------
        x : Tensor
            feature map with the shape of (B, C, H, W).

        Returns
        -------
        PFC attention (B, C, H, W)

        """
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        # -- reshape x into (B, H*W, C)
        x = x.reshape(B, -1, C)
        _, L, _ = x.shape
        assert L == self.feature_map_size[0] * self.feature_map_size[1]

        # -- shortcut for connection
        shortcut1 = x

        # -- first layer normalization
        x = self.norm1(x)

        # -- reshape again into (B, H, W, c)
        x = x.view(B, H, W, C)

        # -- Calculate PFC attention
        PFC_attn = self.PFC_attn(x)  # (B_, N, C)
        # -- merge windows
        # (B_, windowsize, windowsize, C)
        PFC_attn = PFC_attn.view(-1, self.window_size, self.window_size, C)

        # -- Window reverse
        x = window_reverse(PFC_attn, self.window_size, H, W)  # B H' W' C

        # -- reshape x
        x = x.view(B, H * W, C)

        #-- Connection
        x = shortcut1 + self.drop_path(x)
        shortcut2 = x
        # -- Apply layer normalization
        x = self.norm2(x)

        #-- MLP
        x = self.mlp(x)

        # -- drop path
        x = self.drop_path(x)

        # -- second connection
        x = shortcut2 + x  # (B, H*W, C)

        # -- reshape the output into tensor shape such that can be fused with the output of ResCNN
        x = x.view(B, H, W, C)
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)

        # -- return the output of the PFC layer
        return x


# -- Basic Layer
class BasicLayer(nn.Module):
    def __init__(self,
                 depth: int,
                 in_channel: int,
                 out_channel: int,
                 feature_map_size: list,
                 num_heads: int,  # it must be divisable to in_channles
                 window_size: int,
                 mlp_ratio: int,
                 downsampling: bool = True,
                 qkv_bias: bool = True,
                 qk_scale: float = None,
                 drop_path: float = 0.,
                 mlp_drop: float = 0.2,
                 attn_drop: float = 0.2,
                 proj_drop: float = 0.2,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm
                 ):
        """

    Parameters
    ----------
    depth : int
        Depth of the network.
    in_channel : int
        Number of input channels.
    out_channel : int
        Number of output channels.
    feature_map_size : list
        Spatial size of the input feature map.
    num_heads : int
        Number of attention heads, must be divisible by `in_channel`.
    window_size : int
        Window size.
    mlp_ratio : int
        Ratio of mlp hidden dim to embedding dim.
    downsampling : bool, optional
        If True, apply downsampling. Default is True.
    qkv_bias : bool, optional
        If True, add a learnable bias to query, key, value. Default is True.
    qk_scale : float or None, optional
        Override default qk scale of head_dim ** -0.5 if set. Default is None.
    drop_path : float, optional
        Stochastic depth rate. Default is 0.0.
    mlp_drop : float, optional
        Dropout rate for MLP. Default is 0.2.
    attn_drop : float, optional
        Attention dropout rate. Default is 0.2.
    proj_drop : float, optional
        Dropout ratio of output. Default is 0.2.
    act_layer : nn.Module, optional
        Activation layer. Default is nn.GELU.
    norm_layer : nn.Module, optional
        Normalization layer. Default is nn.LayerNorm.

    Returns
    -------
    pfc_output : Tensor
    downsampled_feature_map : Tensor

    """
        super(BasicLayer, self).__init__()

        assert in_channel % num_heads == 0, (
            "\n error in PFC class"
            "\nThe num_heads must be divisable with in_channles.\n"
            "Check number of features of the input feature map and number of heads "
        )

        self.downsampling = downsampling
        self.out_channel = out_channel
        # -- PFC Blocks depth times
        self.PFC_Blocks = nn.ModuleList([
            PFClBlock(in_channels=in_channel,
                      feature_map_size=feature_map_size,
                      num_heads=num_heads,
                      window_size=window_size,
                      mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias,
                      qk_scale=qk_scale,
                      drop_path=drop_path[i] if isinstance(
                          drop_path, list) else drop_path,
                      mlp_drop=mlp_drop,
                      attn_drop=attn_drop,
                      proj_drop=proj_drop,
                      act_layer=nn.GELU,
                      norm_layer=nn.LayerNorm)
            for i in range(depth)
        ])

        # -- Check whether downsampling must be applied or not
        if self.downsampling:
            self.reduction = reduction(in_channel, out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """

        Parameters
        ----------
        x : torch.Tensor
            Input feature map with the shape of [B, C, H, W].

        Returns
        -------
        y: torch.Tensor.
            Fused feature maps
            with downsampling : [B, new_C, H/2, W/2]
            without downsampling : [B, new_C, H, W]

        """

        # make a copy of the tensor using the clone method
        pfc_output = x.clone()
        # -- Apply PFC block l times
        for pfc_blk in self.PFC_Blocks:
            pfc_output = pfc_blk(pfc_output)

        # -- Downsampling ?

        if self.downsampling:
            downsampled_feature_map = self.reduction(pfc_output)
        else:
            downsampled_feature_map = pfc_output

        # -- Return the output of layer
        return pfc_output, downsampled_feature_map


# -- PFC Classifier
class pfc_classifier(nn.Module):
    def __init__(self,
                 img_size: int = 32,
                 patch_size: int = 1,
                 in_channels: int = 3,
                 embed_dim: int = 8,
                 apply_emd: bool = False,
                 out_channels: list = [16, 32, 64, 64],
                 pyramid_Fusion: bool = True,
                 num_classes: int = 4,
                 depths: list = [2, 2, 2, 2],
                 num_heads: list = [1, 4, 4, 8],
                 windowsize: int = 4,
                 mlp_ratio: int = 4.,
                 qkv_bias: bool = True,
                 qk_scale: int = None,
                 drop_rate: float = 0.,
                 attn_drop_rate: float = 0.,
                 drop_path_rate: float = 0.1,
                 norm_layer=nn.BatchNorm2d,
                 ape: bool = False,  # absolute position embedding
                 patch_norm=True,  # applyign layer normaliztion after embedding
                 **kwargs  # ignore the rest arguments
                 ):
        """
    Parameters
    ----------
    img_size : int, optional
        Size of the input image. Default is 32.
    patch_size : int, optional
        Size of the patch. Default is 1.
    in_channels : int, optional
        Number of input channels. Default is 3.
    embed_dim : int, optional
        Embedding dimension. Default is 8.
    apply_emd : bool, optional
        If True, apply embedding. Default is False.
    out_channels : list, optional
        List of output channels for each layer. Default is [16, 32, 64, 64].
    pyramid_Fusion : bool, optional
        If True, apply pyramid fusion. Default is True.
    num_classes : int, optional
        Number of output classes. Default is 4.
    depths : list, optional
        List of depths for each stage. Default is [2, 2, 2, 2].
    num_heads : list, optional
        List of number of attention heads for each stage. Default is [1, 4, 4, 8].
    windowsize : int, optional
        Size of the attention window. Default is 4.
    mlp_ratio : float, optional
        Ratio of mlp hidden dim to embedding dim. Default is 4.0.
    qkv_bias : bool, optional
        If True, add a learnable bias to query, key, value. Default is True.
    qk_scale : int or None, optional
        Override default qk scale of head_dim ** -0.5 if set. Default is None.
    drop_rate : float, optional
        Dropout rate. Default is 0.0.
    attn_drop_rate : float, optional
        Dropout rate for attention layers. Default is 0.0.
    drop_path_rate : float, optional
        Drop path rate for stochastic depth. Default is 0.1.
    norm_layer : nn.Module, optional
        Normalization layer. Default is nn.BatchNorm2d.
    ape : bool, optional
        If True, use absolute position embedding. Default is False.
    patch_norm : bool, optional
        If True, apply layer normalization after embedding. Default is True.
    **kwargs : dict
        Additional keyword arguments.

    Returns
    ----------
    y : Tensor
        classified patch


        """

        super(pfc_classifier, self).__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = out_channels[-1]
        self.mlp_ratio = mlp_ratio
        self.apply_embeding = apply_emd
        self.pyramid_Fusion = pyramid_Fusion

        #print(f'img_size in pfc_classifier: {img_size.shape}')
        num_patches = img_size * img_size  # becuase we auume that p=1
        patches_resolution = [img_size, img_size]
        self.patches_resolution = patches_resolution

        # -- applying patch embedding
        if self.apply_embeding:
            self.patch_embed = embeding(
                in_channels=in_channels, embed_features=embed_dim)
            self.in_channels = embed_dim

        else:
            self.in_channels = in_channels

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # print(f'self.num_layers:{self.num_layers}')
            #print(f'num_heads[i_layer]: {num_heads[i_layer]}     i_layer:{i_layer}         num_heads:{num_heads}')
            if i_layer == 0:
                input_featur_map_channels = int(self.in_channels)
            else:
                input_featur_map_channels = int(out_channels[i_layer-1])

            layer = BasicLayer(depth=depths[i_layer],
                               in_channel=input_featur_map_channels,
                               out_channel=int(out_channels[i_layer]),
                               feature_map_size=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               num_heads=num_heads[i_layer],
                               window_size=windowsize,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsampling=True if (
                                   i_layer < self.num_layers - 1) else False
                               )
            self.layers.append(layer)

        # -- Batch normalize the last output feature map : [B, out_channels[-1], H//2**range (self.num_layers), W//2**range (self.num_layers)]
        self.norm_last_fused_map = norm_layer(out_channels[-2])

        self.norm_pyramid_fused_map = norm_layer(
            self.in_channels + sum(out_channels[:-1]))

        # -- cancatenate pfc outputs
        self.cat_pfc_outputs = ConcatFusionMaps(self.in_channels, out_channels)

        # -- pfc Average
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        #-- classification
        self.head = nn.Linear(self.in_channels + sum(out_channels[:-1]), num_classes) if self.pyramid_Fusion \
            else nn.Linear(out_channels[-2], num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        """

        Parameters
        ----------
        x : Tensor
            input feature map with the shape of [B, C, H, W].

        Returns
        -------
        x : a list of Tensors
            outputs of the pfc block at each layer.

        """

        if self.apply_embeding:
            x = self.patch_embed(x)  #
        else:
            pass

        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)

        pfc_outputs = []

        for idx, layer in enumerate(self.layers, 1):
            pfc_output, x = layer(x)  # x is the downsampled feature map
            pfc_outputs.append(pfc_output)

        # the last list in the pfc_outputs can be used when we do not want to use pyramid fusion
        return pfc_outputs

    # --
    def forward_pfc_outputs(self, pfc_outputs):
        if self.pyramid_Fusion:
            x = self.cat_pfc_outputs(pfc_outputs)
            x = self.norm_pyramid_fused_map(x)
        else:
            # last pfc feature map is used to classify a sample
            x = pfc_outputs[-1]
            x = self.norm_last_fused_map(x)
        return x

    def forward(self, x):
        # -- feed the input feature map into the layers
        pfc_outputs = self.forward_features(x)
        # -- use fused gloacl outputs or the last outptu of the pfc
        y = self.forward_pfc_outputs(pfc_outputs)

        y = self.avgpool(y)  # B C 1
        y = torch.flatten(y, 1)  # B C

        # -- Classification head
        y = self.head(y)
        return y

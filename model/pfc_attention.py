import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Downsample_Fusion_map(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        """


        Parameters
        ----------
        in_channels : TYPE
            DESCRIPTION.
        out_channels : TYPE
            DESCRIPTION.
        kernel_size : TYPE
            DESCRIPTION.
        stride : TYPE
            DESCRIPTION.

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
    """
    This class confuses different feature levels
    """

    def __init__(self, feature_maps_channels):
        super(ConcatFusionMaps, self).__init__()
        self.downsample_convs = nn.ModuleList()

        number_layers = len(feature_maps_channels)
        # we do not apply downsampling on the last fused feature map
        for i in range(number_layers - 1):
            stirde_value = kernel_size = 2 ** (number_layers - 1 - i)
            in_ch = feature_maps_channels[i]
            self.downsample_convs.append(
                Downsample_Fusion_map(in_ch, in_ch, kernel_size, stirde_value)
            )

    def forward(self, fused_feature_map):
        downsampled_feature_maps = []
        for i in range(len(fused_feature_map) - 1):
            downsampled_feature_maps.append(
                (self.downsample_convs[i])(fused_feature_map[i])
            )

        # add the last fused map to the downsampled ones
        downsampled_feature_maps.append(fused_feature_map[-1])
        # -- concatenate downsampled fused feature maps
        downsampled_feature_maps = torch.cat(downsampled_feature_maps, dim=1)
        return downsampled_feature_maps


class embeding(nn.Module):
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 4,
        in_channels: int = 3,
        embed_features: int = 96,
    ):
        super(embeding, self).__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        ]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.embed_features = embed_features

        # -- a convolutional operator to increase number of initial features
        self.increasing_features = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_features,
            kernel_size=patch_size,
            stride=patch_size,
            padding=0,
        )

        self.norm = nn.BatchNorm2d(embed_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.norm(self.increasing_features(x))


# -- Reduction


class reduction(nn.Module):
    def __init__(self, dim):
        super(reduction, self).__init__()

        # -- in_channels is the number of channels after concatenation

        self.reduction = nn.Conv2d(
            in_channels=dim, out_channels=2 * dim, kernel_size=3, stride=2, padding=1
        )

        self.norm = nn.BatchNorm2d(2 * dim)

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
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.4,
    ):
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
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    if (
        len(window_size) > 1
    ):  # windowsize has been tuples into an array [windowsize, windowsize]
        window_size = window_size[0]

    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous(
        ).view(-1, window_size, window_size, C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """

    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B, H // window_size, W // window_size, window_size, window_size, -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PFCAttention(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        input_resolution (list) : spatial size of the input feature map
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        input_resolution,
        qkv_bias=True,
        attn_drop=0.0,
        proj_drop=0.0,
    ):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim**-0.5


        
        #self.scale = head_dim**-0.5

        self.norm_kv = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(self.dim)

        # -- used by both fine- and coarse-grained attention
        self.softmax = nn.Softmax(dim=-1)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

        # -- Fine-grained Attention
        # -- define a parameter table of relative position bias for Fine-grained attention
        # -- To understand how relative position is calculated, read swin_transformer
        self.local_relative_position_bias_table = nn.Parameter(
            # 2*Wh-1 * 2*Ww-1, nH
            torch.zeros((2 * window_size[0] - 1)
                        * (2 * window_size[1] - 1), num_heads)
        )

        # get pair-wise relative position index for each token inside the window
        local_coords_h = torch.arange(self.window_size[0])
        local_coords_w = torch.arange(self.window_size[1])
        local_coords = torch.stack(torch.meshgrid(
            [local_coords_h, local_coords_w]))

        local_coords_flatten = torch.flatten(local_coords, 1)

        local_relative_coords = (
            local_coords_flatten[:, :, None] - local_coords_flatten[:, None, :]
        )
        local_relative_coords = local_relative_coords.permute(
            1, 2, 0).contiguous()
        # shift to start from 0
        local_relative_coords[:, :, 0] += self.window_size[0] - 1
        local_relative_coords[:, :, 1] += self.window_size[1] - 1
        local_relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        local_relative_position_index = local_relative_coords.sum(-1)
        self.register_buffer(
            "local_relative_position_index", local_relative_position_index
        )
        trunc_normal_(self.local_relative_position_bias_table, std=0.02)

        # -- Project the concatenated headers
        self.local_proj = nn.Linear(dim, dim)

        # -- Coarse-grained Attention
        # -- before calling WindowAttention, we must be sure that the spatial size of
        #   the input feature map is divisible by window_size.
        self.total_windows = (
            input_resolution[0] // window_size[0],
            input_resolution[1] // window_size[1],
        )

        # -- from focal paper
        self.relative_position_bias_table_to_windows = nn.Parameter(
            torch.zeros(
                (window_size[0] + self.total_windows[0] - 1) *
                #
                (window_size[1] + self.total_windows[1] - 1),
                num_heads,
            )
        )
        trunc_normal_(self.relative_position_bias_table_to_windows, std=0.02)

        # -- define a parameter table of relative position bias for coarse-grained attention
        coords_h_k = torch.arange(self.total_windows[0])
        coords_w_k = torch.arange(self.total_windows[1])
        coords_k = torch.stack(
            torch.meshgrid([coords_h_k, coords_w_k])
        )
        coords_flatten_k = torch.flatten(coords_k, 1)

        relative_coords = (
            local_coords_flatten[:, :, None] - coords_flatten_k[:, None, :]
        )
        relative_coords = relative_coords.permute(
            1, 2, 0
        ).contiguous()

        # -- shift to start from zero
        relative_coords[:, :, 0] += self.total_windows[0] - 1
        relative_coords[:, :, 1] += self.total_windows[1] - 1

        # -- scales the row indices of the relative position tensor so that they can be used as
        #   indices into the relative positional embedding tensor.
        relative_coords[:, :,
                        0] *= (self.window_size[1] + self.total_windows[1]) - 1

        # -- computes the relative position index for each token
        global_relative_position_index = relative_coords.sum(-1)

        self.register_buffer(
            "global_relative_position_index", global_relative_position_index
        )

        # -- fully coonected layer for reducing size of K and V
        self.pool_layer_k = nn.Linear(
            self.window_size[0] * self.window_size[1], 1)
        self.pool_layer_k.weight.data.fill_(
            1.0 / self.window_size[0] * self.window_size[1]
        )
        self.pool_layer_k.bias.data.fill_(0)

        self.pool_layer_v = nn.Linear(
            self.window_size[0] * self.window_size[1], 1)
        self.pool_layer_v.weight.data.fill_(
            1.0 / (self.window_size[0] * self.window_size[1])
        )
        self.pool_layer_v.bias.data.fill_(0)

        # -- project the concatenated headers
        self.global_proj = nn.Linear(dim, dim)

        # -- mapping global and local attention for combining together
        self.pfcmapping = nn.Linear(2 * dim, dim, bias=qkv_bias)

    def forward(self, x):
        """
        Args:
            x: input features

        """
        B, H, W, C = x.shape

        # -- partition the input feature maps into non-overlapping windows
        x_windows = window_partition(x, self.window_size)

        # -- merge pixels inside of a window
        x_windows = x_windows.view(-1,
                                   self.window_size[0] * self.window_size[1], C)

        B_, N, C = x_windows.shape

        # -- calculate fine-grained attention which is attention inside of each window

        qkv = self.qkv(x)
        # -- Note: C must be divisible by the number of heads
        qkv = qkv.reshape(B_, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4).contiguous()

        # -- seperate Query, Key, and Value matrices
        q_local, k_local, v_local = qkv[0], qkv[1], qkv[2]
        # -- fine-grained attention
        fine_attn = (q_local * self.scale) @ (k_local.transpose(-2, -1))

        # -- local relative position bias
        local_relative_position_bias = self.local_relative_position_bias_table[
            self.local_relative_position_index.view(-1)
        ].view(
            # N, N, nHeads
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )

        local_relative_position_bias = local_relative_position_bias.permute(
            2, 0, 1
        ).contiguous()

        # -- add the local relative position bias to the local attention
        fine_attn = fine_attn + local_relative_position_bias.unsqueeze(
            0
        )

        # -- apply softmax on the fine-grained attention
        fine_attn = self.softmax(fine_attn)

        # -- apply dropout
        fine_attn = self.attn_drop(fine_attn)

        # -- multiply by Value matrix
        fine_attn = (
            (fine_attn @ v_local).transpose(1, 2).reshape(B_, N, C).contiguous()
        )

        # -- multihead weight
        fine_attn = self.local_proj(fine_attn)

        # -- apply dropout on the projected local attention
        fine_attn = self.proj_drop(fine_attn)

        # -- calculate coarse-grained attention which is attention between windows

        # -- global Keys
        nWs = B_ // B  # number of windows in a partitioned feature map

        # -- seperate partitioned windows from batch
        k_local = k_local.reshape(
            B, nWs, self.num_heads, N, C // self.num_heads
        ).contiguous()

        # -- transpose the pixels inside of a local window to the last dimension to apply a linear projection
        k_local = k_local.transpose(-2, -1).contiguous()
        # -- apply linear projection on each local window
        K_global = self.pool_layer_k(k_local)
        # -- swip partitioned windows to the last dimension to have a correct repetition
        K_global = K_global.permute(0, 4, 2, 1, 3).contiguous()

        # -- repeat the global windows to the number of partitioned windows. becuase global attention is calculated
        #   between local q and each global windows
        K_global = K_global.repeat(1, nWs, 1, 1, 1)

        # -- finally, combined the number of partitioned windows with batches to have similar shape as local Q
        K_global = K_global.reshape(
            -1, self.num_heads, nWs, C // self.num_heads
        ).contiguous()

        # -- global Value
        # -- seperate partitioned windows from batch
        v_local = v_local.reshape(
            B, nWs, self.num_heads, N, C // self.num_heads
        ).contiguous()

        # -- transpose the pixels inside of a local window to the last dimension to apply a linear projection
        v_local = v_local.transpose(-2, -1).contiguous()

        # -- apply linear projection on each local window
        V_global = self.pool_layer_k(v_local)

        # -- swip partitioned windows to the last dimension to have a correct repetition
        V_global = V_global.permute(0, 4, 2, 1, 3).contiguous()

        # -- repeat the global windows
        V_global = V_global.repeat(1, nWs, 1, 1, 1)

        # -- finally, combined the number of partitioned windows with batches to have similar shape as local Q
        V_global = V_global.reshape(
            -1, self.num_heads, nWs, C // self.num_heads
        ).contiguous()

        # -- coarse-grained attention
        coarse_attn = (q_local * self.scale) @ (K_global.transpose(-2, -1))

        # -- relative positional bias between K-global windows and local q windows
        global_relative_position_bias = self.relative_position_bias_table_to_windows[
            self.global_relative_position_index.view(-1)
        ].view(

            self.window_size[0] * self.window_size[1],
            self.total_windows[0] * self.total_windows[1],
            -1,
        )

        global_relative_position_bias = global_relative_position_bias.permute(
            2, 0, 1
        ).contiguous()

        # -- add relative positional bias to coarse_attn
        coarse_attn = coarse_attn + global_relative_position_bias.unsqueeze(0)

        # -- apply softmax on the local attention
        coarse_attn = self.softmax(coarse_attn)

        # -- apply dropout
        coarse_attn = self.attn_drop(coarse_attn)

        # -- multiply by Value matrix
        coarse_attn = (
            (coarse_attn @ V_global)
            .transpose(
                1,
                2,
            )
            .reshape(B_, N, C)
            .contiguous()
        )

        # -- multihead weight
        coarse_attn = self.global_proj(coarse_attn)

        # -- apply dropout on the projected global attention
        coarse_attn = self.proj_drop(coarse_attn)

        # -- PFC Attention
        PFC_Attn = torch.cat((fine_attn, coarse_attn),
                             dim=-1)  # (B_, N, 2 *C)

        # -- apply Mapping
        PFC_Attn = self.pfcmapping(PFC_Attn)  # (B_, N, C)

        return PFC_Attn


class PFClBlock(nn.Module):
    """
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]):  resulotion of input feature map.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        mlp_drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        proj_drop (floar, optinal): Dropout ratio of output. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm

    """

    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        drop_path=0.0,
        mlp_drop=0.2,
        attn_drop=0.2,
        proj_drop=0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.input_resolution = input_resolution
        # -- Make sure that the window size of partitioning is not larger than the input resolution
        if min(self.input_resolution) <= self.window_size:
            # -- if window size is larger than input resolution, we don't partition windows
            self.window_size = min(self.input_resolution)

        # -- first normaliztion
        self.norm1 = norm_layer(dim)

        # -- PFC Attention
        self.PFC_attn = PFCAttention(
            dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            input_resolution=input_resolution,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )

        # -- drop path to reduce overfitting  posibility

        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()

        # -- second layer normalization
        self.norm2 = norm_layer(dim)

        # -- MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=mlp_drop,
        )

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
        assert L == self.input_resolution[0] * self.input_resolution[1]

        # -- shortcut for connection
        shortcut1 = x

        # -- first layer normalization
        x = self.norm1(x)

        # -- reshape again into (B, H, W, c)
        x = x.view(B, H, W, C)

        # -- calculate PFC attention
        PFC_attn = self.PFC_attn(x)  # (B_, N, C)
        # -- merge windows
        # (B_, windowsize, windowsize, C)
        PFC_attn = PFC_attn.view(-1, self.window_size, self.window_size, C)

        # -- window reverse
        x = window_reverse(PFC_attn, self.window_size, H, W)  # B H' W' C

        # -- reshape x
        x = x.view(B, H * W, C)

        # -- connection
        x = shortcut1 + self.drop_path(x)
        shortcut2 = x
        # -- apply layer normalization
        x = self.norm2(x)

        # -- MLP
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
    def __init__(
        self,
        dim: int,
        input_resolution: list,
        depth: int,
        num_heads: int,  # it must be divisable to in_channles
        window_size: int,
        mlp_ratio: int,
        downsampling: bool = True,
        qkv_bias: bool = True,
        drop_path: float = 0.0,
        mlp_drop: float = 0.2,
        attn_drop: float = 0.2,
        proj_drop: float = 0.2,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super(BasicLayer, self).__init__()

        assert dim % num_heads == 0, (
            "\n error in PFC class"
            "\nThe num_heads must be divisable with in_channles.\n"
            "Check number of features of the input feature map and number of heads "
        )

        self.downsampling = downsampling
        # -- PFC Blocks depth times
        self.PFC_Blocks = nn.ModuleList(
            [
                PFClBlock(
                    dim=dim,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop_path=(
                        drop_path[i] if isinstance(
                            drop_path, list) else drop_path
                    ),
                    mlp_drop=mlp_drop,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    act_layer=nn.GELU,
                    norm_layer=nn.LayerNorm,
                )
                for i in range(depth)
            ]
        )

        # -- check whether downsampling must be applied or not
        if self.downsampling:
            self.reduction = reduction(dim)

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
        # -- apply PFC block l times
        for pfc_blk in self.PFC_Blocks:
            pfc_output = pfc_blk(pfc_output)

        # -- downsampling ?

        if self.downsampling:
            downsampled_feature_map = self.reduction(pfc_output)
        else:
            downsampled_feature_map = pfc_output

        # -- return the output of layer
        return pfc_output, downsampled_feature_map


# -- PFC Classifier
class pfc_classifier(nn.Module):
    def __init__(
        self,
        config
    ):
        super(pfc_classifier, self).__init__()
        norm_layer = nn.BatchNorm2d
        self.num_classes = config["MODEL"]["num_classes"]
        self.num_layers = len(config["MODEL"]["depths"])
        self.ape = config["MODEL"]["ape"]
        self.patch_norm = config["MODEL"]["patch_norm"]
        self.pyramid_Fusion = config["MODEL"]["pyramid_Fusion"]
        output_channels = [config["MODEL"]["embed_dim"]
                           * 2**i for i in range(self.num_layers)]

        # apply embedding
        self.patch_embed = embeding(img_size=config["MODEL"]["img_size"],
                                    patch_size=config["MODEL"]["patch_size"],
                                    in_channels=config["MODEL"]["in_channels"],
                                    embed_features=config["MODEL"]["embed_dim"])
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, config["MODEL"]["embed_dim"])
            )
            trunc_normal_(self.absolute_pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=config["MODEL"]["drop_rate"])

        # stochastic depth
        dpr = [
            x.item()
            for x in torch.linspace(
                0,
                config["MODEL"]["drop_path_rate"],
                # stochastic depth decay rule
                sum(config["MODEL"]["depths"]),
            )
        ]
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):

            layer = BasicLayer(
                dim=int(config["MODEL"]["embed_dim"] * 2**i_layer),
                input_resolution=(
                    patches_resolution[0] // (2**i_layer),
                    patches_resolution[1] // (2**i_layer),
                ),
                depth=config["MODEL"]["depths"][i_layer],
                num_heads=config["MODEL"]["num_heads"][i_layer],
                window_size=config["MODEL"]["window_size"],
                mlp_ratio=config["MODEL"]["mlp_ratio"],
                qkv_bias=config["MODEL"]["qkv_bias"],
                attn_drop=config["MODEL"]["attn_drop_rate"],
                drop_path=dpr[sum(config["MODEL"]["depths"][:i_layer]): sum(
                    config["MODEL"]["depths"][: i_layer + 1])],
                norm_layer=norm_layer,
                downsampling=True if (
                    i_layer < self.num_layers - 1) else False,
            )
            self.layers.append(layer)

        # -- Batch normalize the last output feature map :
        self.norm_last_fused_map = norm_layer(output_channels[-1])

        self.norm_pyramid_fused_map = norm_layer(sum(output_channels))

        # -- concatenate pfc outputs
        self.cat_pfc_outputs = ConcatFusionMaps(output_channels)

        # -- pfc Average
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

        # -- classification
        self.head = (
            nn.Linear(sum(output_channels), config["MODEL"]["num_classes"])
            if self.pyramid_Fusion
            else nn.Linear(output_channels[-1], config["MODEL"]["num_classes"])
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
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
        x = self.patch_embed(x)  #

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
        if self.pyramid_Fusion:  # PFC Method
            x = self.cat_pfc_outputs(pfc_outputs)
            x = self.norm_pyramid_fused_map(x)
        else:  # FC Method
            # last pfc feature map is used to classify a sample
            x = pfc_outputs[-1]
            x = self.norm_last_fused_map(x)
        return x

    def forward(self, x):
        # -- feed the input ffeature map into the layers
        pfc_outputs = self.forward_features(x)
        # -- use fused gloacl outputs or the last outptu of the pfc
        y = self.forward_pfc_outputs(pfc_outputs)

        y = self.avgpool(y)  # B C 1
        y = torch.flatten(y, 1)  # B C

        # -- Classification head
        y = self.head(y)
        return y

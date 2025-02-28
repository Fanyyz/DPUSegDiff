import torch.nn as nn
import torch
from einops import rearrange
import math
from src.utils.helper_funcs import exists
from timm.models.vision_transformer import PatchEmbed
from src.blocks.BasicConvolutin import ResnetBlock,local_Attention,init_ConvBlock,Final_ConvBlock
from src.blocks.BasicTransformer import WaveTransformer,WaveCrossTransformer,SpatialTransformer,SpatialCrossTransformer
from src.blocks.Graph import GraphFusion
from src.blocks.Bottleneck import fuseBlock
__all__ = ["V10"]

class concat(nn.Module):
    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

class UP(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UP, self).__init__()
        self.up_sampling = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=2,stride=2,bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return self.norm(self.up_sampling(x))

class UP_4(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(UP_4, self).__init__()
        self.up_sampling = nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4,stride=4,bias=False)
        self.norm = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return self.norm(self.up_sampling(x))

class TimestepEmbedder(nn.Module):
    """
    嵌入时间步标量为向量表示。
    这是一个神经网络模块，将时间步（标量 t）嵌入到高维向量中。
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        """
        初始化函数，定义嵌入层和多层感知机（MLP）。

        :param hidden_size: 隐藏层大小，即最终嵌入的维度。
        :param frequency_embedding_size: 用于正弦嵌入的频率嵌入大小，默认为 256。
        """
        super().__init__()
        # 定义一个 MLP，将时间嵌入映射到隐藏维度空间
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),  # 线性层将频率嵌入转换为隐藏大小
            nn.SiLU(),  # 激活函数 SiLU（Swish 激活）
            nn.Linear(hidden_size, hidden_size, bias=True),  # 另一个线性层，保持隐藏大小不变
        )
        self.frequency_embedding_size = frequency_embedding_size  # 保存频率嵌入的维度

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        创建基于正弦和余弦函数的时间步嵌入。

        :param t: 一个 1 维张量，包含 N 个时间步索引，表示每个 batch 中的时间步。
                  这些值可以是小数。
        :param dim: 输出嵌入的维度大小。
        :param max_period: 控制嵌入中的最小频率（默认值为 10000）。
        :return: 返回一个形状为 (N, D) 的张量，包含每个 batch 样本的时间步嵌入。
        """
        # 计算嵌入的维度一半，因为我们要使用正弦和余弦对称地嵌入
        half = dim // 2
        # 生成指数衰减的频率
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)  # 将频率移动到与时间步相同的设备
        # 计算正弦和余弦嵌入
        args = t[:, None].float() * freqs[None]  # 每个时间步乘以频率，生成每个维度的嵌入
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # 连接 cos 和 sin 结果作为最终嵌入
        # 如果维度是奇数，需要补上一个维度
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # 补上 0
        return embedding

    def forward(self, t):
        """
        前向传播，生成时间步的嵌入并通过 MLP 处理。

        :param t: 输入的时间步张量。
        :return: 经过 MLP 处理后的时间步嵌入张量。
        """
        # 生成时间步的频率嵌入
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        # 通过 MLP 进一步处理嵌入
        t_emb = self.mlp(t_freq)
        return t_emb  # 返回最终的时间步嵌入

class Basic_ShallowEncode_Block(nn.Module):
    def __init__(self, time_emb_dim,
                 hidden_size,
                 resolution,
                 mlp_bias,
                 mlp_drop,
                 resnet_block_groups=8,
                 add_transformer=False,
                 add_resnet=False):  # 新增参数控制是否增加额外Transformer
        super().__init__()

        # 原始层的定义
        self.tb_1 = WaveTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, resolution=resolution,
                                    mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.tb_2 = WaveCrossTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, resolution=resolution,
                                         mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.rb_1 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,
                                groups=resnet_block_groups)
        self.rb_2 = local_Attention(channel=hidden_size, time_emb_dim=time_emb_dim)

        # 增加额外的 Transformer 层
        self.add_transformer = add_transformer
        if self.add_transformer:
            self.tb_extra = WaveTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, resolution=resolution,
                                            mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.add_resnet = add_resnet
        if self.add_resnet:
            self.rb_extra = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,groups=resnet_block_groups)
    def forward(self, x, g, t):
        x, _ = self.tb_1(x, t)
        g = self.rb_1(g, t)

        x, x_skip = self.tb_2(x, g, t)

        if self.add_transformer:
            x, x_skip = self.tb_extra(x, t)

        g = self.rb_2(g, x_skip, t)

        if self.add_resnet:
            g = self.rb_extra(g, t)

        return x_skip, g

class Basic_DepthEncode_Block(nn.Module):
    def __init__(self, time_emb_dim,
                 hidden_size,
                 resolution,
                 head_dim,
                 sr_ratio,
                 attn_drop,
                 drop,
                 resnet_block_groups=8,
                 add_transformer=False,
                 add_resnet=False):
        super().__init__()
        self.tb_1 = SpatialTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)
        self.tb_2 = SpatialCrossTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)

        self.rb_1 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,groups=resnet_block_groups)
        self.rb_2 = local_Attention(channel=hidden_size, time_emb_dim=time_emb_dim)

        self.add_transformer = add_transformer
        if self.add_transformer:
            self.tb_extra1 = SpatialTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)
            self.tb_extra2 = SpatialTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,
                                               resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)

        self.add_resnet = add_resnet
        if self.add_resnet:
            self.rb_extra1 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,
                                        groups=resnet_block_groups)
            self.rb_extra2 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,
                                        groups=resnet_block_groups)
    def forward(self, x, g, t):
        x, _ = self.tb_1(x, t)
        g = self.rb_1(g, t)

        x, x_skip = self.tb_2(x, g, t)
        if self.add_transformer:
            x, x_skip = self.tb_extra1(x, t)
            x, x_skip = self.tb_extra2(x, t)
        g = self.rb_2(g, x_skip, t)
        if self.add_resnet:
            g = self.rb_extra1(g, t)
            g = self.rb_extra2(g, t)

        return x_skip, g

class Encoder(nn.Module):
    def __init__(self,
                 time_emb_dim,
                 hidden_size,
                 head_dim,
                 resolution,
                 sr_ratio,
                 attn_drop,
                 drop,
                 mlp_bias,
                 mlp_drop):
        super().__init__()
        self.initConvBlock = init_ConvBlock(time_emb_dim=time_emb_dim)
        self.patch_embed1_x = PatchEmbed(img_size=resolution[0], patch_size=4, in_chans=hidden_size[0], embed_dim=hidden_size[1])
        self.patch_embed2_x = PatchEmbed(img_size=resolution[1], patch_size=2, in_chans=hidden_size[1], embed_dim=hidden_size[2])
        self.patch_embed3_x = PatchEmbed(img_size=resolution[2], patch_size=2, in_chans=hidden_size[2], embed_dim=hidden_size[3])
        self.patch_embed4_x = PatchEmbed(img_size=resolution[3], patch_size=2, in_chans=hidden_size[3], embed_dim=hidden_size[4])

        # self.em1 = Basic_ShallowEncode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[1], resolution=resolution[1],mlp_bias=mlp_bias, mlp_drop=mlp_drop,add_transformer=False,add_resnet=True)
        # self.em2 = Basic_ShallowEncode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[2], resolution=resolution[2],mlp_bias=mlp_bias, mlp_drop=mlp_drop,add_transformer=True,add_resnet=False)
        self.em1 = Basic_DepthEncode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[1], head_dim=head_dim,resolution=resolution[1], sr_ratio=sr_ratio[0], attn_drop=attn_drop, drop=drop,add_transformer=False,add_resnet=True)
        self.em2 = Basic_DepthEncode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[2], head_dim=head_dim,resolution=resolution[2], sr_ratio=sr_ratio[3], attn_drop=attn_drop, drop=drop,add_transformer=True,add_resnet=False)

        self.em3 = Basic_DepthEncode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[3], head_dim=head_dim,resolution=resolution[3], sr_ratio=sr_ratio[3], attn_drop=attn_drop, drop=drop,add_transformer=True,add_resnet=False)
        self.em4 = Basic_DepthEncode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[4], head_dim=head_dim,resolution=resolution[4], sr_ratio=sr_ratio[3], attn_drop=attn_drop, drop=drop,add_transformer=False,add_resnet=False)

        self.down1 = nn.Conv2d(in_channels=hidden_size[0], out_channels=hidden_size[1], stride=4, kernel_size=4)
        self.down2 = nn.Conv2d(in_channels=hidden_size[1], out_channels=hidden_size[2], stride=2, kernel_size=2)
        self.down3 = nn.Conv2d(in_channels=hidden_size[2], out_channels=hidden_size[3], stride=2, kernel_size=2)
        self.down4 = nn.Conv2d(in_channels=hidden_size[3], out_channels=hidden_size[4], stride=2, kernel_size=2)

    def forward(self,x,g,t):
        x_skips = []
        g_skips = []
        x,g = self.initConvBlock(x,g,t)
        x_skips.append(x)
        g_skips.append(g)

        x = self.patch_embed1_x(x)
        g = self.down1(g)
        x,g = self.em1(x,g,t)
        x_skips.append(x)
        g_skips.append(g)

        x = self.patch_embed2_x(x)
        g = self.down2(g)
        x, g = self.em2(x, g, t)
        x_skips.append(x)
        g_skips.append(g)

        x = self.patch_embed3_x(x)
        g = self.down3(g)
        x, g = self.em3(x, g, t)
        x_skips.append(x)
        g_skips.append(g)

        x = self.patch_embed4_x(x)
        g = self.down4(g)
        x, g = self.em4(x, g, t)
        x_skips.append(x)
        g_skips.append(g)

        return x,g,x_skips,g_skips

class Basic_DepthDecode_Block(nn.Module):
    def __init__(self, time_emb_dim,
                 hidden_size,
                 resolution,
                 head_dim,
                 sr_ratio,
                 attn_drop,
                 drop,
                 resnet_block_groups=8,
                 add_transformer=False,
                 add_resnet=False):
        super().__init__()
        self.tb_1 = SpatialTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)
        self.tb_2 = SpatialCrossTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)

        self.rb_1 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,groups=resnet_block_groups)
        self.rb_2 = local_Attention(channel=hidden_size, time_emb_dim=time_emb_dim)
        self.add_transformer = add_transformer
        if self.add_transformer:
            self.tb_extra1 = SpatialTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)
            self.tb_extra2 = SpatialTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,
                                               resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop)

        self.add_resnet = add_resnet
        if self.add_resnet:
            self.rb_extra1 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,
                                        groups=resnet_block_groups)
            self.rb_extra2 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,
                                        groups=resnet_block_groups)
    def forward(self, x, g, t):
        x = rearrange(x,"b c h w -> b (h w) c" )
        x, _ = self.tb_1(x, t)
        g = self.rb_1(g, t)

        x, x_skip = self.tb_2(x, g, t)
        if self.add_transformer:
            x, x_skip = self.tb_extra1(x, t)
            x, x_skip = self.tb_extra2(x, t)
        g = self.rb_2(g, x_skip, t)
        if self.add_resnet:
            g = self.rb_extra1(g, t)
            g = self.rb_extra2(g, t)

        return x_skip, g

class Basic_ShallowDecode_Block(nn.Module):
    def __init__(self,time_emb_dim,
                 hidden_size,
                 resolution,
                 mlp_bias,
                 mlp_drop,
                 resnet_block_groups=8,
                 add_transformer=False,
                 add_resnet=False):
        super().__init__()
        self.tb_1 = WaveTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, resolution=resolution,mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.tb_2 = WaveCrossTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, resolution=resolution, mlp_bias=mlp_bias, mlp_drop=mlp_drop)

        self.rb_1 = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,groups=resnet_block_groups)
        self.rb_2 = local_Attention(channel=hidden_size, time_emb_dim=time_emb_dim)
        self.add_transformer = add_transformer
        if self.add_transformer:
            self.tb_extra = WaveTransformer(time_emb_dim=time_emb_dim, hidden_size=hidden_size, resolution=resolution,
                                            mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.add_resnet = add_resnet
        if self.add_resnet:
            self.rb_extra = ResnetBlock(dim=hidden_size, dim_out=hidden_size, time_emb_dim=time_emb_dim,groups=resnet_block_groups)
    def forward(self,x,g,t):
        x = rearrange(x, "b c h w -> b (h w) c")
        x,_ = self.tb_1(x,t)
        g = self.rb_1(g,t)

        x,x_skip = self.tb_2(x,g,t)
        if self.add_transformer:
            x, x_skip = self.tb_extra(x, t)
        g = self.rb_2(g,x_skip,t)
        if self.add_resnet:
            g = self.rb_extra(g, t)

        return x_skip,g

class Decoder(nn.Module):
    def __init__(self,
                 time_emb_dim,
                 hidden_size,
                 head_dim,
                 resolution,
                 sr_ratio,
                 attn_drop,
                 drop,
                 mlp_bias,
                 mlp_drop):
        super().__init__()
        self.concat4x = concat(ch_in=hidden_size[5],ch_out=hidden_size[4])
        self.concat3x = concat(ch_in=hidden_size[4], ch_out=hidden_size[3])
        self.concat2x = concat(ch_in=hidden_size[3], ch_out=hidden_size[2])
        self.concat1x = concat(ch_in=hidden_size[2], ch_out=hidden_size[1])
        self.concat0x = concat(ch_in=hidden_size[1], ch_out=hidden_size[0])

        self.concat4g = concat(ch_in=hidden_size[5], ch_out=hidden_size[4])
        self.concat3g = concat(ch_in=hidden_size[4], ch_out=hidden_size[3])
        self.concat2g = concat(ch_in=hidden_size[3], ch_out=hidden_size[2])
        self.concat1g = concat(ch_in=hidden_size[2], ch_out=hidden_size[1])
        self.concat0g = concat(ch_in=hidden_size[1], ch_out=hidden_size[0])

        self.up4x = UP(in_channels=hidden_size[4],out_channels=hidden_size[3])
        self.up3x = UP(in_channels=hidden_size[3], out_channels=hidden_size[2])
        self.up2x = UP(in_channels=hidden_size[2], out_channels=hidden_size[1])
        self.up1x = UP_4(in_channels=hidden_size[1], out_channels=hidden_size[0],)

        self.up4g = UP(in_channels=hidden_size[4], out_channels=hidden_size[3])
        self.up3g = UP(in_channels=hidden_size[3], out_channels=hidden_size[2])
        self.up2g = UP(in_channels=hidden_size[2], out_channels=hidden_size[1])
        self.up1g = UP_4(in_channels=hidden_size[1], out_channels=hidden_size[0])

        self.dm4 = Basic_DepthDecode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[4], head_dim=head_dim,resolution=resolution[4], sr_ratio=sr_ratio[3], attn_drop=attn_drop, drop=drop,add_transformer=False)
        self.dm3 = Basic_DepthDecode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[3], head_dim=head_dim,resolution=resolution[3], sr_ratio=sr_ratio[3], attn_drop=attn_drop, drop=drop,add_transformer=True)

        # self.dm2 = Basic_ShallowDecode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[2], resolution=resolution[2],mlp_bias=mlp_bias, mlp_drop=mlp_drop,add_transformer=True,add_resnet=False)
        # self.dm1 = Basic_ShallowDecode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[1], resolution=resolution[1],mlp_bias=mlp_bias, mlp_drop=mlp_drop,add_transformer=False,add_resnet=True)
        self.dm2 = Basic_DepthDecode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[2], head_dim=head_dim,resolution=resolution[2], sr_ratio=sr_ratio[3], attn_drop=attn_drop, drop=drop,add_transformer=True,add_resnet=False)
        self.dm1 = Basic_DepthDecode_Block(time_emb_dim=time_emb_dim, hidden_size=hidden_size[1], head_dim=head_dim,resolution=resolution[1], sr_ratio=sr_ratio[0], attn_drop=attn_drop, drop=drop,add_transformer=False,add_resnet=True)
        self.final_layer = Final_ConvBlock(time_emb_dim=time_emb_dim)
    def forward(self,x,g,x_skip,g_skip,t):
        g = torch.cat((g,g_skip[4]),dim=1)
        x = torch.cat((x,x_skip[4]),dim=1)
        x = self.concat4x(x)
        g = self.concat4g(g)
        x,g = self.dm4(x,g,t)
        x = self.up4x(x)
        g = self.up4g(g)

        g = torch.cat((g, g_skip[3]), dim=1)
        x = torch.cat((x, x_skip[3]), dim=1)
        x = self.concat3x(x)
        g = self.concat3g(g)
        x, g = self.dm3(x, g, t)
        x = self.up3x(x)
        g = self.up3g(g)

        g = torch.cat((g, g_skip[2]), dim=1)
        x = torch.cat((x, x_skip[2]), dim=1)
        x = self.concat2x(x)
        g = self.concat2g(g)
        x, g = self.dm2(x, g, t)
        x = self.up2x(x)
        g = self.up2g(g)

        g = torch.cat((g, g_skip[1]), dim=1)
        x = torch.cat((x, x_skip[1]), dim=1)
        x = self.concat1x(x)
        g = self.concat1g(g)
        x, g = self.dm1(x, g, t)

        x = self.up1x(x)
        g = self.up1g(g)
        g = torch.cat((g, g_skip[0]), dim=1)
        x = torch.cat((x, x_skip[0]), dim=1)
        x = self.concat0x(x)
        g = self.concat0g(g)

        pred_noise = self.final_layer(x,g,t)

        return pred_noise

class DPUSegDiff(nn.Module):
    def __init__(self,
                 hidden_size=[32,64, 128, 256, 512,1024],
                 head_dim=32,
                 time_emb_dim=512,
                 resolution=[256,64, 32, 16, 8, 4],
                 sr_ratio=[2, 2, 1, 1],
                 attn_drop=0,
                 drop=0,
                 mlp_bias=True,
                 mlp_drop=[0, 0],
                 in_channels_x=1,
                 in_channels_g=3,
                 ):
        super().__init__()
        self.time_emb = TimestepEmbedder(hidden_size=time_emb_dim)
        self.encoder = Encoder(time_emb_dim=time_emb_dim,hidden_size=hidden_size,head_dim=head_dim,resolution=resolution,sr_ratio=sr_ratio,attn_drop=attn_drop,drop=drop,mlp_bias=mlp_bias,mlp_drop=mlp_drop)
        # self.bottleneck = GraphFusion(in_channel=hidden_size[4])
        self.decoder = Decoder(time_emb_dim=time_emb_dim, hidden_size=hidden_size, head_dim=head_dim,resolution=resolution, sr_ratio=sr_ratio, attn_drop=attn_drop, drop=drop,mlp_bias=mlp_bias, mlp_drop=mlp_drop)
        self.bottleneck = fuseBlock(time_emb_dim=time_emb_dim, hidden_size=hidden_size[4], resolution=resolution[4])


    def forward(self,x,g,t):
        t = self.time_emb(t)
        x,g,x_skips,g_skips = self.encoder(x,g,t)
        x,g = self.bottleneck(x,g,t)
        # x, g = self.bottleneck(x,g)
        x = self.decoder(x,g,x_skips,g_skips,t)
        return x

# model = V9()
#
# # 计算参数量
# total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"模型的参数总量: {total_params}")
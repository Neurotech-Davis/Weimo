import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torchvision.transforms import Compose, Resize, ToTensor
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce
import math

n_channels = 10
n_classes = 2
sampling_rate = 50
trial_duration = 4.5
n_timepoints = int(sampling_rate * trial_duration)

'''
1. Classical Baseline: CSP + LDA
Architecture: Common Spatial Patterns (CSP) for spatial filtering on µ/β frequency bands, followed by Linear Discriminant Analysis (LDA) for classification. No deep learning.
This remains a widespread approach: CSP extracts task-discriminative spatial features, which are then passed to LDA for classification. Frontiers It serves as the floor baseline against which all deep learning models are compared. Typical 4-class accuracy on BCI IV-2a hovers around 65–70%.
'''
# no deep learning, no need to build it out

'''
2. ShallowConvNet & DeepConvNet — Schirrmeister et al. (2017)
Preprocessing: None basically
Source: EEG Deep Learning (Neural Computation, 2017)
Architecture (ShallowConvNet): Two convolutional layers — a temporal convolution followed by a spatial convolution — with a mean-pooling layer and log-variance activation. Designed to approximate band-power features from the µ/β range.
Architecture (DeepConvNet): A deeper stack of 4 convolutional blocks (Conv → BatchNorm → ELU → MaxPool), trading interpretability for capacity.
ShallowConvNet and DeepConvNet were among the first architectures to decode MI tasks from raw EEG signals end-to-end, though DeepConvNet's large parameter count made it harder to interpret. Springer
Results on BCI IV-2a (4-class): ShallowConvNet ~75–76% accuracy; DeepConvNet slightly lower due to overfitting in small-data regimes. CTNet surpassed ShallowConvNet by 6.83% on BCI IV-2a, contextualizing its competitive standing. PubMed Central
'''

class DeepConvNet(nn.Module):
    def __init__(self):
        super(DeepConvNet, self).__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        
        # block 1
        self.temporal_conv = nn.Conv2d(1, 25, kernel_size=(1,10))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=(n_channels, 1)),
            nn.BatchNorm2d(25),
            nn.ELU(inplace=True),
        )

        # block 2
        self.block2_conv = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=(1,10)),
            nn.BatchNorm2d(50),
            nn.ELU(inplace=True),
        )
        # block 3
        self.block3_conv = nn.Sequential(
            nn.Conv2d(50, 100, kernel_size=(1,10)),
            nn.BatchNorm2d(100),
            nn.ELU(inplace=True),
        )
        # block 4
        self.block4_conv = nn.Sequential(
            nn.Conv2d(100, 200, kernel_size=(1,10)),
            nn.BatchNorm2d(200),
            nn.ELU(inplace=True),
        )
        # classification
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # collapses to (batch, 200, 1, 1)
            nn.Flatten(),                  # → (batch, 200)
            nn.Linear(200, n_classes),
            nn.Softmax(dim=1),
        )
                
    def forward(self, x):
        # block 1
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = self.max_pool(x)

        # blocks 2-4
        x = self.block2_conv(x)
        x = self.max_pool(x)
        x = self.block3_conv(x)
        x = self.max_pool(x)
        x = self.block4_conv(x)
        x = self.max_pool(x)

        # output
        x = self.classifier(x)
        return x
    
class ShallowConvNet(nn.Module):
    def __init__(self, n_channels=22, n_classes=4):
        super(ShallowConvNet, self).__init__()

        self.temporal_conv = nn.Conv2d(1, 40, kernel_size=(1, 25))
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(40, 40, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(40),
        )

        # squaring + avg pool + log — the FBCSP-equivalent nonlinearity
        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # collapses to (batch, 40, 1, 1)
            nn.Flatten(),                  # → (batch, 40)
            nn.Linear(40, n_classes),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x = self.temporal_conv(x)
        x = self.spatial_conv(x)
        x = x ** 2                  # squaring nonlinearity
        x = self.avg_pool(x)
        x = torch.log(x.clamp(min=1e-6))  # log activation, clamped for stability
        x = self.classifier(x)
        return x

'''
3. EEGNet — Lawhern et al. (2018)
Source: Journal of Neural Engineering, 15(5):056013
Architecture: Three-block compact CNN:

Block 1: 2D temporal convolution (captures frequency-specific filters)
Block 2: Depthwise convolution (learns spatial filters per temporal feature map) + BatchNorm + ELU + Average pooling
Block 3: Separable convolution (summarizes temporal features per depthwise filter) + BatchNorm + ELU + Average pooling + Dropout → Softmax

EEGNet uses one-dimensional and deep convolutional layers for real-time feature extraction, enabling training on limited datasets while achieving competitive decoding accuracy. Frontiers Two standard configurations exist: EEGNet-4,2 and EEGNet-8,2 (F1 filters, D depth multiplier).
Results on BCI IV-2a: ~73–76% (subject-specific). EEGNet-4,2 and EEGNet-8,2 achieve very competitive classification performance on both datasets and are highly compact. ScienceDirect
Why it matters: EEGNet is the de facto lightweight baseline in the field — nearly every subsequent paper benchmarks against it.
'''

class EEGNet(nn.Module):
    def __init__(self, num_temporal_filters = 8, num_spatial_filters = 10):
        super(EEGNet, self).__init__()
        '''
        C = num channels
        T = num time points
        F1 = num temporal filters
        D = num spatial filters
        F2 = num pointwise filtesrs
        N = num classes
        p = 0.5 for dropout

        https://arxiv.org/pdf/1611.08024
        '''
        # block 1        
        # input (C, T)
        # block one
        # reshape (1, C, T)
        # conv2d (F1, C, T)
        # batch norm (F1, C, T)
        # depthwise conv2d (D * F1, 1, T)
        # batch norm (D * F1, 1, T)
        # activation (D * F1, 1, T)
        # average pool 2d (D * F1, 1, T // 4)
        # drouput (D * F1, 1, T // 4)
        # 2 convolution steps in sequence
        self.block_one = nn.Sequential(
            nn.Conv2d(1, num_temporal_filters, kernel_size=(1, sampling_rate // 2), padding='same', bias=False),
            nn.BatchNorm2d(num_temporal_filters),
            nn.Conv2d(num_temporal_filters, num_spatial_filters * num_temporal_filters, kernel_size=(n_channels, 1), groups=num_temporal_filters, bias=False),
            nn.BatchNorm2d(num_spatial_filters * num_temporal_filters),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=0.5),
        )

        # block 2
        # separable conv2d (F2, 1, T // 4) which is a kind of depthwise conv2d
        # batch norm (F2, 1, T // 4)
        # activation (F2, 1, T // 4)
        # average pool 2d (F2, 1, T // 32)
        # droupout (F2, 1, T // 32)
        # flatten (F2 * (T // 32)
        # dense (N)

        f2 = num_spatial_filters * num_temporal_filters
        f2 = num_spatial_filters * num_temporal_filters
        self.block_two = nn.Sequential(
            # separable conv = depthwise + pointwise
            nn.Conv2d(f2, f2, kernel_size=(1, 16), groups=f2, bias=False, padding='same'),
            nn.Conv2d(f2, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=0.5),
            nn.Flatten(),
        )

        self.classification = nn.Sequential(
            nn.Linear(f2 * (n_timepoints // 32), n_classes),
            nn.Softmax(dim=1),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1) # (B, C, T) -> (B, 1, C, T)
        x = self.block_one(x)
        x = self.block_two(x)
        x = self.classification(x)
        return x

'''
4. ATCNet — Altaheri et al. (2022)
Source: IEEE Transactions on Neural Systems and Rehabilitation Engineering
Architecture: Three sequential modules:

Convolutional module (EEGNet-style): Temporal + depthwise + separable convolutions for local spatial-spectral feature extraction
Attention block: Multi-head self-attention (MSA) applied to the convolutional output to emphasize the most task-relevant temporal segments
TCN block: Temporal Convolutional Network with dilated causal convolutions and residual connections for high-level temporal feature extraction

Additionally uses a sliding window augmentation strategy at the convolutional module output to increase effective training data.
ATCNet integrates attention mechanisms with TCN and CNN architectures — the attention block applies multi-head self-attention to identify key features, while the TCN block extracts advanced temporal features. Frontiers
Results on BCI IV-2a: ~85–87% (subject-specific). ATCNet represents one of the most advanced currently available approaches combining attention mechanisms with temporal convolutional networks for MI-EEG decoding. arXiv
preprocessing: none
'''

class ATCAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, dropout=0.5):
        super(ATCAttentionBlock, self).__init__()

        self.norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

    def forward(self, x):
        # x: (B, F, 1, n)
        B, F, _, n = x.shape
        x = x.squeeze(2).permute(0, 2, 1)  # (B, n, F)

        # pre-norm → attention → residual
        normed = self.norm(x)
        attn_out, _ = self.attention(normed, normed, normed)
        x = x + attn_out                    # (B, n, F)

        return x

class DilatedCausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(DilatedCausalConv1d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.padding = (kernel_size - 1) * dilation
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=0, # Manual padding will be applied
            dilation=dilation
        )

    def forward(self, x):
        # Pad the input tensor on the left side
        # The padding format is (padding_left, padding_right) for the last dimension
        x_padded = F.pad(x, (self.padding, 0), 'constant', 0)
        
        # Apply the convolution
        output = self.conv(x_padded)
        
        return output

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ResidualBlock, self).__init__()

        self.conv_block = nn.Sequential(
            DilatedCausalConv1d(in_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
            DilatedCausalConv1d(out_channels, out_channels, kernel_size, dilation),
            nn.BatchNorm1d(out_channels),
            nn.ELU(inplace=True),
        )

        # 1x1 conv if dimensions don't match, otherwise identity
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        return self.conv_block(x) + self.residual(x)

class TCBlock(nn.Module):
    def __init__(self, in_channels=32, out_channels=32, kernel_size=4):
        super(TCBlock, self).__init__()

        # L=2 residual blocks, dilation doubles each block
        self.network = nn.Sequential(
            ResidualBlock(in_channels, out_channels, kernel_size, dilation=1),
            ResidualBlock(out_channels, out_channels, kernel_size, dilation=2),
        )

    def forward(self, x):
        # x: (B, F, n) — Conv1d expects (B, channels, length)
        x = self.network(x)     # (B, F, n)
        x = x[:, :, -1]        # take last timestep output → (B, F)
        return x

class ATCNet(nn.Module):
    def __init__(self, num_filters = 16, d = 2, p2 = 8, num_heads=2, block_one_dropout=0.3, attn_drouput=0.5):
        super(ATCNet, self).__init__()
        '''
        3 main blocks
        block 1: convolutional 
        - 3 layers
        block 2: attention
        - multi head attention
        block 3: temporal convolution
        '''

        # input (B,  1,  C,  T)
        self.block_one = nn.Sequential(
            # layer one
            nn.Conv2d(1, num_filters, kernel_size=(1, sampling_rate // 4), bias=False, padding='same'), # (B, 16,  C,  T)
            nn.BatchNorm2d(num_filters), # (B, 16,  C,  T)

            # layer two
            nn.Conv2d(num_filters, num_filters * d, kernel_size=(n_channels, 1), groups=num_filters, bias=False), # (B, 32,  1,  T)
            nn.BatchNorm2d(num_filters * d),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)), # (B, 32,  1,  T//4)
            nn.Dropout2d(p=block_one_dropout),
            
            # layer three
            nn.Conv2d(num_filters * d, num_filters * d, kernel_size = (1, sampling_rate // 8), bias=False), # (B, 32,  1,  T//4)
            nn.BatchNorm2d(num_filters * d),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, p2)), # (B, 32,  1,  T//32)
            nn.Dropout2d(p=block_one_dropout),
        )
        # output of block one: (B, 32,  1,  T//32)

        self.block_two = ATCAttentionBlock(
            embed_dim=num_filters * d, 
            num_heads=num_heads,
            dropout=attn_drouput,
        )
        # output is (B, n, F)

        self.block_three = TCBlock(
            in_channels=num_filters * d, 
            out_channels=num_filters * d, 
            kernel_size=4
        )

        self.classifier = nn.Sequential(
            nn.Linear(num_filters * d, n_classes),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.block_one(x)           # (B, F, 1, n)
        x = self.block_two(x)           # (B, n, F)
        x = x.permute(0, 2, 1)         # (B, F, n)
        x = self.block_three(x)         # (B, F)
        x = self.classifier(x)            # (B, n_classes)

        return x



'''
5. EEG Conformer — Song et al. (2022/2023)
- Source: IEEE Transactions on Neural Systems and Rehabilitation Engineering
- Architecture: Hybrid CNN + Transformer:
- Convolution module (ShallowConvNet-based): Temporal convolution → spatial depthwise convolution → average pooling; extracts local temporal-spatial features
- Self-attention (Transformer encoder) module: Projects convolutional features into a sequence, applies multi-head self-attention with positional encoding to model global temporal dependencies
- Classification module: Fully connected layer → Softmax
- BCI IV-2a: 78.66%
- BCI IV-2b: 84.63%
- SEED (emotion): 95.30%

preprocessing:
- bandpass filtering
- 6 order chebyshev filter 
- Z score standardization
 xo = (xi - mean) / sqrt(variance)
    xi = band pass filtered data
    xo = output of standardization

https://pubmed.ncbi.nlm.nih.gov/37015413/
'''

class PatchEmbedding(nn.Module):
    def __init__(self, emb_size=40):
        # self.patch_size = patch_size
        super().__init__()

        self.shallownet = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.Conv2d(40, 40, (n_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.AvgPool2d((1, 75), (1, 15)),  # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(0.5),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )


    def forward(self, x: Tensor) -> Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, emb_size, num_heads, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)
        self.att_drop = nn.Dropout(dropout)
        self.projection = nn.Linear(emb_size, emb_size)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)
        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)  
        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)

        scaling = self.emb_size ** (1 / 2)
        att = F.softmax(energy / scaling, dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out


class ResidualAdd(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        res = x
        x = self.fn(x, **kwargs)
        x += res
        return x


class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, drop_p):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(drop_p),
            nn.Linear(expansion * emb_size, emb_size),
        )


class GELU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return input*0.5*(1.0+torch.erf(input/math.sqrt(2.0)))


class TransformerEncoderBlock(nn.Sequential):
    def __init__(self,
                 emb_size,
                 num_heads=10,
                 drop_p=0.5,
                 forward_expansion=4,
                 forward_drop_p=0.5):
        super().__init__(
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                MultiHeadAttention(emb_size, num_heads, drop_p),
                nn.Dropout(drop_p)
            )),
            ResidualAdd(nn.Sequential(
                nn.LayerNorm(emb_size),
                FeedForwardBlock(
                    emb_size, expansion=forward_expansion, drop_p=forward_drop_p),
                nn.Dropout(drop_p)
            )
            ))


class TransformerEncoder(nn.Sequential):
    def __init__(self, depth, emb_size):
        super().__init__(*[TransformerEncoderBlock(emb_size) for _ in range(depth)])


class ClassificationHead(nn.Sequential):
    def __init__(self, emb_size, n_classes, fc_input_size):
        super().__init__()
        
        # global average pooling
        self.clshead = nn.Sequential(
            Reduce('b n e -> b e', reduction='mean'),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, n_classes)
        )
        self.fc = nn.Sequential(
            nn.Linear(fc_input_size, 256),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(256, 32),
            nn.ELU(),
            nn.Dropout(0.3),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = x.contiguous().view(x.size(0), -1)
        out = self.fc(x)
        return x, out


class Conformer(nn.Sequential):
    def __init__(self, emb_size=40, depth=6, n_classes=4):

        _t = n_timepoints - 24                                      # after Conv2d (1,25): 201
        _t = (_t - 75) // 15 + 1                                   # after AvgPool2d (1,75) stride (1,15): 9
        emb_size = 40
        fc_input_size = _t * emb_size                               # 360
        
        super().__init__(

            PatchEmbedding(emb_size),
            TransformerEncoder(depth, emb_size),
            ClassificationHead(emb_size, n_classes, fc_input_size)
        )




'''
6. CTNet — Zhao et al. (2024)
Source: Scientific Reports 14, 20237 (2024). Link
Architecture: CNN (improved EEGNet) + Transformer encoder:

Convolutional module: Modified EEGNet with temporal and depthwise spatial convolutions; outputs local feature representations
Transformer encoder: Multi-head self-attention over the convolutional features to extract global temporal correlations
Classifier: FC → Softmax

Key design goal: fewer trainable parameters than EEG Conformer, reducing overfitting.
In subject-specific experiments on BCI IV-2a, CTNet achieved an average accuracy of 82.52%, with the lowest standard deviation of 9.61% and a Kappa score of 0.7670 among evaluated models — indicating strong consistency across subjects. PubMed Central
Results:

BCI IV-2a (subject-specific): 82.52% (Kappa: 0.7670)
Outperforms EEG Conformer by 4.86% and DeepConvNet by 4.74%
'''

class CTNet(nn.Module):
    def __init__(self, num_temporal_filters=8, D=2, dropout=0.5, d=16):
        super(CTNet, self).__init__()
        # F1 = num_temporal_filters, F2 = F1*D after depthwise, d = output dim
        F1 = num_temporal_filters
        F1D = F1 * D  # channels after depthwise
        _t = n_timepoints                   # 225
        _t = (_t - (sampling_rate // 4) + 1)  # after temporal conv... wait, padding='same' so unchanged: 225
        _t = _t // 4                        # after AvgPool(1,4): 56
        _t = _t - 16 + 1                    # after spatial conv (1,16), no padding: 41
        Tc = _t // 2                        # after AvgPool(1,2): 20

        fc_input_size = Tc * d              # 320

        self.conv_layer = nn.Sequential(
            # temporal conv
            nn.Conv2d(1, F1, kernel_size=(1, sampling_rate // 4), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            # depthwise spatial conv — collapses channel dim
            nn.Conv2d(F1, F1D, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1D),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
            # spatial conv — extracts higher-level temporal features
            nn.Conv2d(F1D, d, kernel_size=(1, 16), bias=False),
            nn.BatchNorm2d(d),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 2)),   # Tc = 20
            nn.Dropout(p=dropout),
        ) # output is (B, d, 1, Tc) = (B, 16, 1, 20)

        self.transformer = TransformerEncoder(depth=6, emb_size=d)

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(d * Tc, n_classes)
        )

    def forward(self, x):
        x = self.conv_layer(x)          # (B, d, 1, Tc)
        cnn_out = x.squeeze(2).permute(0, 2, 1)   # (B, Tc, d)
        trans_out = self.transformer(cnn_out)      # (B, Tc, d)
        x = cnn_out + trans_out         # CTNet's residual add between CNN and transformer
        x = x.flatten(1)                # (B, Tc*d)
        x = self.classifier(x)
        return x




'''
7. MBMANet — Deng et al. (2024)
Architecture: Tri-branch structure combining parallel multi-head attention with Squeeze-and-Excitation (SE) blocks, Convolutional Block Attention Module (CBAM), EEGNet, and TCN. The branches run in parallel and are concatenated before classification.
MBMANet's tri-branch architecture combines parallel multi-head attention with SE and CBAM alongside EEGNet and TCN for MI-EEG classification. PubMed Central
Results on BCI IV-2a: 83.18% — a 3.43% improvement over prior models at time of publication.
'''
# ─── Attention modules ────────────────────────────────────────────────────────

class SE(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x).view(x.size(0), x.size(1), 1, 1)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 1)
        self.ch_mlp = nn.Sequential(
            nn.Linear(channels, mid), nn.ReLU(), nn.Linear(mid, channels)
        )
        self.sp_conv = nn.Conv2d(2, 1, kernel_size=(1, 7), padding=(0, 3))

    def forward(self, x):
        # channel attention
        avg = self.ch_mlp(x.mean(dim=[2, 3]))
        mx  = self.ch_mlp(x.amax(dim=[2, 3]))
        x = x * torch.sigmoid(avg + mx).view(x.size(0), x.size(1), 1, 1)
        # spatial attention
        sp = torch.cat([x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.sp_conv(sp))


class SegmentMHA(nn.Module):
    """MHA on a (B, C, 1, T) segment — treats T as sequence."""
    def __init__(self, channels, num_heads=2, dropout=0.1):
        super().__init__()
        self.mha  = nn.MultiheadAttention(channels, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        seq = x.squeeze(2).permute(0, 2, 1)          # (B, T, C)
        out, _ = self.mha(seq, seq, seq)
        out = self.norm(seq + out)
        return out.permute(0, 2, 1).unsqueeze(2)      # (B, C, 1, T)


# ─── TCN ─────────────────────────────────────────────────────────────────────

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation, padding=pad),
            nn.BatchNorm1d(out_ch),
            nn.ELU(),
            nn.Dropout(dropout),
        )
        self.shortcut = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ELU()

    def forward(self, x):
        out = self.conv(x)[..., :x.size(-1)]  # trim causal padding
        return self.act(out + self.shortcut(x))


class TCN(nn.Module):
    def __init__(self, in_ch, out_ch, n_layers=2, kernel_size=3, dropout=0.2):
        super().__init__()
        self.layers = nn.Sequential(*[
            TCNBlock(in_ch if i == 0 else out_ch, out_ch,
                     kernel_size, dilation=2**i, dropout=dropout)
            for i in range(n_layers)
        ])

    def forward(self, x):
        return self.layers(x)


# ─── EEGNet branch (no flatten/classifier) ───────────────────────────────────

class EEGNetBranch(nn.Module):
    """Single EEGNet branch, outputs (B, F2, 1, T//32)."""
    def __init__(self, F1=8, D=2, dropout=0.5):
        super().__init__()
        F2 = F1 * D
        self.block_one = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, sampling_rate // 2), padding='same', bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F2, kernel_size=(n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(p=dropout),
        )
        self.block_two = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16), groups=F2, padding='same', bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(inplace=True),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(p=dropout),
        )
        self.out_channels = F2

    def forward(self, x):
        return self.block_two(self.block_one(x))


# ─── MBMANet ─────────────────────────────────────────────────────────────────

_Tprime = n_timepoints // 32   # temporal length after EEGNet branches: 225//32 = 7

class MBMANet(nn.Module):
    def __init__(self, F1s=(4, 8, 16), D=2, tcn_channels=64, tcn_layers=2, dropout=0.5):
        super().__init__()

        # Module 2: three EEGNet branches with different F1 for diverse features
        self.branches = nn.ModuleList([EEGNetBranch(F1, D, dropout) for F1 in F1s])
        seg_channels   = [F1 * D for F1 in F1s]   # [8, 16, 32]
        total_channels = sum(seg_channels)          # 56

        # Module 3: one attention mechanism per segment (branch output)
        self.mha_attn  = SegmentMHA(seg_channels[0], num_heads=2)
        self.cbam_attn = CBAM(seg_channels[1])
        self.se_attn   = SE(seg_channels[2])

        # TCN on fused attended features
        self.tcn = TCN(total_channels, tcn_channels, n_layers=tcn_layers, dropout=dropout)

        # Module 4: classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(tcn_channels * _Tprime, n_classes),
        )

    def forward(self, x):
        # x: (B, 1, n_channels, n_timepoints)

        # Module 2: parallel EEGNet branches → (B, F2_i, 1, T')
        b0, b1, b2 = [branch(x) for branch in self.branches]

        # Module 3: different attention per segment
        b0 = self.mha_attn(b0)     # MHA
        b1 = self.cbam_attn(b1)    # CBAM
        b2 = self.se_attn(b2)      # SE

        # fuse + squeeze H dim → (B, total_channels, T')
        fused = torch.cat([b0, b1, b2], dim=1).squeeze(2)

        # TCN → (B, tcn_channels, T')
        out = self.tcn(fused)

        # classify
        return self.classifier(out)

'''
8. EMD+CWT+SPoC+CSP+ADBN — Mathiyazhagan & Devasena (2025)
Source: Scientific Reports / PubMed (2025). Link
Architecture: Fully hybrid classical + deep pipeline:

Preprocessing: EMD (Empirical Mode Decomposition) for intrinsic signal mode extraction + CWT (Continuous Wavelet Transform) for multi-resolution analysis
Spatial feature enhancement: Source Power Coherence (SPoC) + Common Spatial Patterns (CSP)
Classifier: Adaptive Deep Belief Network (ADBN) with parameters optimized via Far and Near Optimization (FNO) algorithm

On the BCI IV Dataset 2a, this approach achieves 95.7% accuracy, 96.2% recall, 95.9% precision, and 97.5% specificity; on PhysioNet it yields 94.1% accuracy, 94.0% recall, 93.6% precision, and 95.0% specificity. Nature
⚠️ Caveat: These very high numbers (95.7%) should be interpreted cautiously. They are from a single paper not yet broadly replicated, and the complexity of the preprocessing pipeline may limit real-world generalizability and real-time use. The BCI IV-2a community consensus sits closer to 80–87% for well-validated models.
'''

'''
9. Transformer-Based Model — Scientific Reports (2023)
Source: Scientific Reports (2023). Link
Architecture: Pure transformer network operating on raw EEG time series, without a CNN front-end. Uses multi-head self-attention over temporal windows with positional encoding.
This transformer-based architecture achieves 99.7% accuracy on the binary-class BCI III IVa dataset and 84% on the 4-class BCI IV-2a dataset. Nature
Note: The 99.7% figure is on a binary classification task (2-class, small dataset), which inflates performance vs. 4-class benchmarks.
'''

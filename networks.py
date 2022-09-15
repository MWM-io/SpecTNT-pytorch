import torch as th
import torch.nn as nn


class Res2DMaxPoolModule(nn.Module):
    def __init__(self, in_channels, out_channels, pooling=2):
        super(Res2DMaxPoolModule, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn_1 = nn.BatchNorm2d(out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn_2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool2d(tuple(pooling))

        # residual
        self.diff = False
        if in_channels != out_channels:
            self.conv_3 = nn.Conv2d(
                in_channels, out_channels, 3, padding=1)
            self.bn_3 = nn.BatchNorm2d(out_channels)
            self.diff = True

    def forward(self, x):
        out = self.bn_2(self.conv_2(self.relu(self.bn_1(self.conv_1(x)))))
        if self.diff:
            x = self.bn_3(self.conv_3(x))
        out = x + out
        out = self.mp(self.relu(out))
        return out


class ResFrontEnd(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResFrontEnd, self).__init__()
        self.input_bn = nn.BatchNorm2d(in_channels)
        self.layer1 = Res2DMaxPoolModule(in_channels, out_channels, pooling=(2, 2))
        self.layer2 = Res2DMaxPoolModule(out_channels, out_channels, pooling=(2, 2))
        self.layer3 = Res2DMaxPoolModule(out_channels, out_channels, pooling=(2, 1))

    def forward(self, hcqt):
        """
        Inputs:
            hcqt: [B, F, K, T]

        Outputs:
            out: [B, ^F, ^K, ^T]
        """
        # batch normalization
        out = self.input_bn(hcqt)

        # CNN
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)

        return out


class SpecTNTBlock(nn.Module):
    def __init__(
        self, n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, dropout, use_tct
    ):
        super().__init__()

        self.D = embed_dim
        self.F = n_frequencies
        self.K = n_channels
        self.T = n_times

        # TCT: Temporal Class Token
        if use_tct:
            self.T += 1

        # Shared frequency-time linear layers
        self.D_to_K = nn.Linear(self.D, self.K)
        self.K_to_D = nn.Linear(self.K, self.D)

        # Spectral Transformer Encoder
        self.spectral_linear_in = nn.Linear(self.F+1, spectral_dmodel)
        self.spectral_encoder_layer = nn.TransformerEncoderLayer(
            d_model=spectral_dmodel, nhead=spectral_nheads, dim_feedforward=spectral_dimff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.spectral_linear_out = nn.Linear(spectral_dmodel, self.F+1)

        # Temporal Transformer Encoder
        self.temporal_linear_in = nn.Linear(self.T, temporal_dmodel)
        self.temporal_encoder_layer = nn.TransformerEncoderLayer(
            d_model=temporal_dmodel, nhead=temporal_nheads, dim_feedforward=temporal_dimff, dropout=dropout, batch_first=True, activation="gelu", norm_first=True)
        self.temporal_linear_out = nn.Linear(temporal_dmodel, self.T)

    def forward(self, spec_in, temp_in):
        """
        Inputs:
            spec_in: spectral embedding input [B, T, F+1, K]
            temp_in: temporal embedding input [B, T, 1, D]

        Outputs:
            spec_out: spectral embedding output [B, T, F+1, K]
            temp_out: temporal embedding output [B, T, 1, D]
        """
        # Element-wise addition with TE
        spec_in = spec_in + self.D_to_K(temp_in)

        # Spectral Transformer
        spec_in = spec_in.flatten(0, 1).transpose(1, 2)  # [B*T, K, F+1]
        emb = self.spectral_linear_in(spec_in)  # [B*T, K, spectral_dmodel]
        spec_enc_out = self.spectral_encoder_layer(
            emb)  # [B*T, K, spectral_dmodel]
        spec_out = self.spectral_linear_out(spec_enc_out)  # [B*T, K, F+1]
        spec_out = spec_out.view(-1, self.T, self.K,
                                 self.F+1).transpose(2, 3)  # [B, T, F+1, K]

        # FCT slicing (first raw) + back to D
        temp_in = temp_in + self.K_to_D(spec_out[:, :, :1, :])  # [B, T, 1, D]

        # Temporal Transformer
        temp_in = temp_in.permute(0, 2, 3, 1).flatten(0, 1)  # [B, D, T]
        emb = self.temporal_linear_in(temp_in)  # [B, D, temporal_dmodel]
        temp_enc_out = self.temporal_encoder_layer(
            emb)  # [B, D, temporal_dmodel]
        temp_out = self.temporal_linear_out(temp_enc_out)  # [B, D, T]
        temp_out = temp_out.unsqueeze(1).permute(0, 3, 1, 2)  # [B, T, 1, D]

        return spec_out, temp_out


class SpecTNTModule(nn.Module):
    def __init__(
        self, n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, n_blocks, dropout, use_tct
    ):
        super().__init__()

        D = embed_dim
        F = n_frequencies
        K = n_channels
        T = n_times

        # Frequency Class Token
        self.fct = nn.Parameter(th.zeros(1, T, 1, K))

        # Frequency Positional Encoding
        self.fpe = nn.Parameter(th.zeros(1, 1, F+1, K))

        # TCT: Temporal Class Token
        if use_tct:
            self.tct = nn.Parameter(th.zeros(1, 1, 1, D))
        else:
            self.tct = None

        # Temporal Embedding
        self.te = nn.Parameter(th.rand(1, T, 1, D))

        # SpecTNT blocks
        self.spectnt_blocks = nn.ModuleList([
            SpecTNTBlock(
                n_channels,
                n_frequencies,
                n_times,
                spectral_dmodel,
                spectral_nheads,
                spectral_dimff,
                temporal_dmodel,
                temporal_nheads,
                temporal_dimff,
                embed_dim,
                dropout,
                use_tct
            )
            for _ in range(n_blocks)
        ])

    def forward(self, x):
        """
        Input:
            x: [B, T, F, K]

        Output:
            spec_emb: [B, T, F+1, K]
            temp_emb: [B, T, 1, D]
        """
        batch_size = len(x)

        # Initialize spectral embedding - concat FCT (first raw) + add FPE
        fct = th.repeat_interleave(self.fct, batch_size, 0)  # [B, T, 1, K]
        spec_emb = th.cat([fct, x], dim=2)  # [B, T, F+1, K]
        spec_emb = spec_emb + self.fpe
        if self.tct is not None:
            spec_emb = nn.functional.pad(
                spec_emb, (0, 0, 0, 0, 1, 0))  # [B, T+1, F+1, K]

        # Initialize temporal embedding
        temp_emb = th.repeat_interleave(self.te, batch_size, 0)  # [B, T, 1, D]
        if self.tct is not None:
            tct = th.repeat_interleave(self.tct, batch_size, 0)  # [B, 1, 1, D]
            temp_emb = th.cat([tct, temp_emb], dim=1)  # [B, T+1, 1, D]

        # SpecTNT blocks inference
        for block in self.spectnt_blocks:
            spec_emb, temp_emb = block(spec_emb, temp_emb)

        return spec_emb, temp_emb


class SpecTNT(nn.Module):
    def __init__(
        self, front_end_model,
        n_channels, n_frequencies, n_times,
        spectral_dmodel, spectral_nheads, spectral_dimff,
        temporal_dmodel, temporal_nheads, temporal_dimff,
        embed_dim, n_blocks, dropout, use_tct, n_classes
    ):
        super().__init__()
        
        # TCT: Temporal Class Token
        self.use_tct = use_tct

        # Front-end model
        self.fe_model = front_end_model

        # Main model
        self.main_model = SpecTNTModule(
            n_channels,
            n_frequencies,
            n_times,
            spectral_dmodel,
            spectral_nheads,
            spectral_dimff,
            temporal_dmodel,
            temporal_nheads,
            temporal_dimff,
            embed_dim,
            n_blocks,
            dropout,
            use_tct
        )
        
        # Linear layer
        self.linear_out = nn.Linear(embed_dim, n_classes)
        
    def forward(self, features):
        """
        Input:
            features: [B, K, F, T]
        
        Output:
            logits: 
                - [B, n_classes] if use_tct
                - [B, T, n_classes] otherwise
        """
        # Add channel dimension if None
        if len(features.size()) == 3:
            features = features.unsqueeze(1)
        # Front-end model
        fe_out = self.fe_model(features)            # [B, ^K, ^F, ^T]
        fe_out = fe_out.permute(0, 3, 2, 1)         # [B, T, F, K]
        # Main model
        _, temp_emb = self.main_model(fe_out)       # [B, T, 1, D]
        # Linear layer
        if self.use_tct:
            return self.linear_out(temp_emb[:, 0, 0, :])   # [B, n_classes]
        else:
            return self.linear_out(temp_emb[:, :, 0, :])   # [B, T, n_classes]

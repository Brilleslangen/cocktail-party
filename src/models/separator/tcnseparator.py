import torch
from torch import nn
from torch.autograd import Variable
from src.models.submodules import SubModule


class TCNSeparator(SubModule):
    """
    Temporal Convolutional Network separator for binaural feature streams.

    Args:
        input_dim (int): Number of input channels (2D + 3F fused features).
        output_dim (int): Number of output channels (2 D, for two speaker masks).
        bn_dim (int): Bottleneck channel dimension for TCN blocks.
        hidden_dim (int): Hidden channel dimension in depthwise conv blocks.
        num_layers (int): Number of conv blocks per stack.
        num_stacks (int): Number of repeated TCN stacks.
        kernel_size (int): Kernel size for depthwise convolutions.
        skip_connection (bool): Whether to use residual skip outputs.
        causal (bool): Use causal convolutions (no future context).
        dilated (bool): Use exponentially dilated convolutions.
    """

    def __init__(self, input_dim: int, output_dim: int, name: str, bn_dim: int, hidden_dim: int, num_layers: int,
                 num_stacks: int, kernel_size: int, skip_connection: bool, causal: bool, dilated: bool,
                 context_size_ms: float):
        super().__init__()

        # Store dims
        self.name = name
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.receptive_field = 0
        self.dilated = dilated
        self.skip = skip_connection
        self.context_size_ms = context_size_ms

        # Normalization and bottleneck 1x1 conv
        self.LN = cLN(input_dim) if causal else nn.GroupNorm(1, input_dim, eps=1e-8)
        self.BN = nn.Conv1d(input_dim, bn_dim, kernel_size=1, bias=False)

        # Build TCN: num_stacks × num_layers of DepthConv1d
        self.TCN = nn.ModuleList()
        for s in range(num_stacks):
            for i in range(num_layers):
                dilation = 2 ** i if dilated else 1
                padding = dilation * (kernel_size - 1) if dilated else (kernel_size // 2)
                block = DepthConv1d(
                    bn_dim, hidden_dim, kernel_size,
                    padding=padding,
                    dilation=dilation,
                    skip=skip_connection,
                    causal=causal
                )
                # Track total receptive field in frames
                self.receptive_field += (kernel_size - 1) * dilation if (i or s) else kernel_size
                self.TCN.append(block)

        # Final 1x1 conv to project back to output_dim channels
        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv1d(bn_dim, output_dim, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the TCN separator.

        Args:
            x (torch.Tensor): Input tensor of shape [batch, input_dim, time].

        Returns:
            torch.Tensor: Output tensor of shape [batch, output_dim, time].
        """
        # Initial normalization + bottleneck
        f = self.BN(self.LN(x))  # [B, bn_dim, T]

        # Pass through TCN blocks, accumulating skip connections if enabled
        if self.skip:
            skip_sum = 0.0
            for block in self.TCN:
                res, skip = block(f)
                f = f + res
                skip_sum = skip_sum + skip
            f = skip_sum
        else:
            for block in self.TCN:
                res = block(f)
                f = f + res

        # Final output projection
        return self.output(f)

    def get_input_dim(self) -> int:
        """Return the expected input channel dimension."""
        return self.input_dim

    def get_output_dim(self) -> int:
        """Return the output channel dimension (2×D masks)."""
        return self.output_dim


class DepthConv1d(nn.Module):
    """
    Single depthwise separable convolution block with optional skip/residual.

    Args:
        input_channel (int): Number of input channels.
        hidden_channel (int): Number of channels in depthwise conv.
        kernel (int): Kernel size.
        padding (int): Padding size.
        dilation (int): Dilation factor.
        skip (bool): Include skip branch.
        causal (bool): Truncate future context for causal conv.
    """

    def __init__(self, input_channel: int, hidden_channel: int, kernel: int, padding: int, dilation: int = 1,
                 skip: bool = True, causal: bool = False):
        super().__init__()
        self.causal = causal
        self.skip = skip

        # Pointwise conv + activation
        self.conv1d = nn.Conv1d(input_channel, hidden_channel, kernel_size=1)
        self.nonlinearity1 = nn.PReLU()

        # Depthwise conv
        self.padding = (kernel - 1) * dilation if causal else padding
        self.dconv1d = nn.Conv1d(
            hidden_channel, hidden_channel,
            kernel_size=kernel,
            dilation=dilation,
            groups=hidden_channel,
            padding=self.padding
        )
        self.nonlinearity2 = nn.PReLU()

        # Normalization after depthwise
        self.reg1 = cLN(hidden_channel) if causal else nn.GroupNorm(1, hidden_channel, eps=1e-8)
        self.reg2 = cLN(hidden_channel) if causal else nn.GroupNorm(1, hidden_channel, eps=1e-8)

        # Residual and skip 1x1 convs
        self.res_out = nn.Conv1d(hidden_channel, input_channel, kernel_size=1)
        if skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, kernel_size=1)

    def forward(self, x: torch.Tensor):
        """
        Forward through depthwise block.

        Args:
            x (torch.Tensor): [batch, input_channel, time]

        Returns:
            (residual, skip) or residual only if skip disabled:
            residual (torch.Tensor): [batch, input_channel, time]
            skip     (torch.Tensor): [batch, input_channel, time]
        """
        # 1) pointwise conv + norm
        y = self.nonlinearity1(self.conv1d(x))

        # 2) depthwise conv + norm
        if self.causal:
            y = self.dconv1d(y)[:, :, :-self.padding]
        else:
            y = self.dconv1d(y)
        y = self.nonlinearity2(self.reg2(y))

        # 3) residual
        res = self.res_out(y)
        if self.skip:
            skip = self.skip_out(y)
            return res, skip
        return res


class cLN(nn.Module):
    """
    Cumulative LayerNorm along the time axis for causal processing.

    Maintains running mean/var per frame to avoid peeking into the future.
    """

    def __init__(self, dimension: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(1, dimension, 1))
        self.bias = nn.Parameter(torch.zeros(1, dimension, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [batch, channels, time]
        Returns:
            torch.Tensor: same shape, normalized causally per channel.
        """
        B, C, T = x.size()
        # sum over channels → [B, T]
        sum_ = x.sum(dim=1)
        sum_sq = x.pow(2).sum(dim=1)

        # cumulative sums in time
        cum_sum = torch.cumsum(sum_, dim=1)
        cum_sqsum = torch.cumsum(sum_sq, dim=1)

        # entry counts: [1, T], repeated per batch
        cnt = torch.arange(C, C * (T + 1), C, device=x.device, dtype=x.dtype)
        cnt = cnt.unsqueeze(0).expand(B, T)

        # compute mean/var
        mean = cum_sum / cnt
        var = (cum_sqsum - 2 * mean * cum_sum) / cnt + mean.pow(2)
        std = (var + self.eps).sqrt()

        # reshape & apply
        mean = mean.unsqueeze(1)
        std = std.unsqueeze(1)
        norm = (x - mean) / std
        return norm * self.gain + self.bias

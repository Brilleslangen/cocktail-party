import torch
from torch import nn
from torch.autograd import Variable
from src.models.submodules import SubModule


class TCNSeparator(SubModule):
    def __init__(self, input_dim, output_dim, bn_dim, hidden_dim,
                 num_layers, num_stacks, kernel_size, skip_connection,
                 causal, dilated):
        super(TCNSeparator, self).__init__()

        # input is a sequence of features of shape (B, N, L)
        self.input_dim = input_dim
        self.output_dim = output_dim

        # normalization
        if not causal:
            self.LN = nn.GroupNorm(1, input_dim, eps=1e-8)
        else:
            self.LN = cLN(input_dim, eps=1e-8)

        self.BN = nn.Conv1d(input_dim, bn_dim, 1)

        # TCN for feature extraction
        self.receptive_field = 0
        self.dilated = dilated

        self.TCN = nn.ModuleList([])
        for s in range(num_stacks):
            for i in range(num_layers):
                if self.dilated:
                    self.TCN.append(DepthConv1d(bn_dim, hidden_dim, kernel_size, dilation=2 ** i, padding=2 ** i,
                                                skip=skip_connection,
                                                causal=causal))
                else:
                    self.TCN.append(
                        DepthConv1d(bn_dim, hidden_dim, kernel_size, dilation=1, padding=1, skip=skip_connection,
                                    causal=causal))
                if i == 0 and s == 0:
                    self.receptive_field += kernel_size
                else:
                    if self.dilated:
                        self.receptive_field += (kernel_size - 1) * 2 ** i
                    else:
                        self.receptive_field += (kernel_size - 1)

        # print("Receptive field: {:3d} frames.".format(self.receptive_field))

        # output layer
        self.output = nn.Sequential(nn.PReLU(),
                                    nn.Conv1d(bn_dim, output_dim, 1)
                                    )

        self.skip = skip_connection

    def forward(self, x):
        # input shape: (B, N, L)

        # normalization
        output = self.BN(self.LN(x))

        # pass to TCN
        if self.skip:
            skip_connection = 0.
            for i in range(len(self.TCN)):
                residual, skip = self.TCN[i](output)
                output = output + residual
                skip_connection = skip_connection + skip
        else:
            for i in range(len(self.TCN)):
                residual = self.TCN[i](output)
                output = output + residual

        # output layer
        if self.skip:
            output = self.output(skip_connection)
        else:
            output = self.output(output)

        return output

    def get_input_dim(self) -> int:
        return self.input_dim

    def get_output_dim(self) -> int:
        return self.output_dim


class DepthConv1d(nn.Module):

    def __init__(self, input_channel, hidden_channel, kernel, padding, dilation=1, skip=True, causal=False):
        super(DepthConv1d, self).__init__()

        self.causal = causal
        self.skip = skip

        self.conv1d = nn.Conv1d(input_channel, hidden_channel, 1)
        if self.causal:
            self.padding = (kernel - 1) * dilation
        else:
            self.padding = padding
        self.dconv1d = nn.Conv1d(hidden_channel, hidden_channel, kernel, dilation=dilation,
                                 groups=hidden_channel,
                                 padding=self.padding)
        self.res_out = nn.Conv1d(hidden_channel, input_channel, 1)
        self.nonlinearity1 = nn.PReLU()
        self.nonlinearity2 = nn.PReLU()
        if self.causal:
            self.reg1 = cLN(hidden_channel, eps=1e-08)
            self.reg2 = cLN(hidden_channel, eps=1e-08)
        else:
            self.reg1 = nn.GroupNorm(1, hidden_channel, eps=1e-08)
            self.reg2 = nn.GroupNorm(1, hidden_channel, eps=1e-08)

        if self.skip:
            self.skip_out = nn.Conv1d(hidden_channel, input_channel, 1)

    def forward(self, input):
        output = self.reg1(self.nonlinearity1(self.conv1d(input)))
        if self.causal:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)[:, :, :-self.padding]))
        else:
            output = self.reg2(self.nonlinearity2(self.dconv1d(output)))
        residual = self.res_out(output)
        if self.skip:
            skip = self.skip_out(output)
            return residual, skip
        else:
            return residual


class cLN(nn.Module):
    def __init__(self, dimension, eps=1e-8, trainable=True):
        super(cLN, self).__init__()

        self.eps = eps
        if trainable:
            self.gain = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step

        batch_size, channel, time_step = input.size()

        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T

        entry_cnt = torch.arange(channel, channel * (time_step + 1), channel, device=input.device,
                                 dtype=input.dtype)  # shape [time_step]
        entry_cnt = entry_cnt.unsqueeze(0).expand(batch_size, time_step)
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)

        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2 * cum_mean * cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T

        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)

        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain.expand_as(x).type(x.type()) + self.bias.expand_as(x).type(x.type())

import torch.nn as nn


def init_weights(m):
    """
    glorot initialization (xvaier)
    """
    if isinstance(m, nn.Conv1d):
        nn.init.xavier_uniform_(m.weight.data)


class CNNDeep(nn.Module):
    def __init__(self, channels, kernels, strides, pools, num_classes):
        assert len(channels) == len(kernels) == len(strides) == len(pools)
        super().__init__()

        self.conv_blocks = nn.ModuleList()
        prev_channel = 1
        for i in range(len(channels)):
            block = []
            for channel in channels[i]:
                conv_layer = nn.Conv1d(
                    in_channels=prev_channel, out_channels=channel, kernel_size=kernels[i], stride=strides[i])
                block.append(conv_layer)
                prev_channel = channel

                batch_norm_layer = nn.BatchNorm1d(prev_channel)
                block.append(batch_norm_layer)

                relu_activation = nn.ReLU()
                block.append(relu_activation)
            self.conv_blocks.append(nn.Sequential(*block))

        self.pool_blocks = nn.ModuleList()
        for i in range(len(pools)):
            if pools[i] == 1:
                max_pool_layer = nn.MaxPool1d(kernel_size=4)
                self.pool_blocks.append(max_pool_layer)

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev_channel, num_classes)

    def forward(self, input):
        for i in range(len(self.conv_blocks)):
            input = self.conv_blocks[i](input)
            if i < len(self.pool_blocks):
                input = self.pool_blocks[i](input)

        output = self.global_pool(input).squeeze()
        output = self.fc(output)
        return output.squeeze()


class ResBlock(nn.Module):
    def __init__(self, prev_channel, out_channel, kernel):
        super().__init__()
        self.res = nn.Sequential(
            nn.Conv1d(in_channels=prev_channel,
                      out_channels=out_channel, kernel_size=kernel),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(),
            nn.Conv1d(in_channels=out_channel,
                      out_channel=out_channel, kernel_size=kernel),
            nn.BatchNorm1d(out_channel)
        )
        self.batch_norm = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()

    def forward(self, input):
        identity = input
        input = self.res(input)
        if input.shape[1] == identity.shape[1]:
            input += identity

        elif input.shape[1] > identity.shape[1]:
            if input.shape[1] % identity.shape[1] == 0:
                input += identity.repeat(1,
                                         input.shape[1]//identity.shape[1], 1)
            else:
                raise RuntimeError(
                    "Dims in ResBlock needs to be divisible on the previous dims!")
        else:
            if identity.shape[1] % input.shape[1] == 0:
                identity += input.repeat(1,
                                         identity.shape[1]//input.shape[1], 1)
            else:
                raise RuntimeError(
                    "Dims in ResBlock needs to be divisible on the previous dims!")
            input = identity
        input = self.bn(input)
        input = self.relu(input)
        return input


class CNNRes(nn.Module):
    def __init__(self, channels, kernels, strides, pools, num_classes):
        assert len(channels) == len(kernels) == len(strides) == len(pools)
        super().__init__()

        prev_channel = 1
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_channels=prev_channel,
                      out_channels=channels[0][0], kernel_size=kernels[0], stride=strides[0]),
            nn.BatchNorm1d(channels[0][0]),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4)
        )

        prev_channel = channels[0][0]
        self.res_blocks = nn.ModuleList()
        for i in range(1, len(channels)):
            blocks = []
            for channel in channels[i]:
                blocks.append(ResBlock(prev_channel=prev_channel,
                              out_channel=channel, kernel=kernels[i]))
                prev_channel = channel
            self.res_blocks.append(nn.Sequential(*blocks))

        self.pool_blocks = nn.ModuleList()
        for i in range(1, len(pools)):
            if pools[i] == 1:
                self.pool_blocks.append(nn.MaxPool1d(kernel_size=kernels[i]))

        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(prev_channel, num_classes)

    def forward(self, input):
        input = self.conv_block(input)
        for i in range(len(self.res_blocks)):
            input = self.res_blocks[i](input)
            if i < len(self.pool_blocks):
                input = self.pool_blocks[i](input)
        output = self.global_pool(input).squeeze()
        output = self.fc(output)
        return output.squeeze()

import torch
import torch.nn as nn


class CorrTorch(nn.Module):
    """
    Pytorch implementation of correlation layer.
    Adapted from: https://github.com/limacv/CorrelationLayer
    """

    def __init__(self, max_disp=4):
        super().__init__()
        self.stride = 2
        self.max_disp = max_disp
        self.padding_layer = nn.ConstantPad2d(max_disp, 0)

    def forward(self, in1, in2):
        """
        Compute correlation between two feature maps.
        :param in1: first feature map
        :param in2: second feature map
        :return: correlation volume with shape D^2 x H x W
        """
        height, width = in1.shape[2], in1.shape[3]
        in2_padded = self.padding_layer(in2)

        # Compute offsets => neighborhood size D = 2 * max_disp + 1
        # Stride used to skip certain positions in neighborhood
        offset_y, offset_x = torch.meshgrid([torch.arange(0, 2 * self.max_disp + 1, self.stride),
                                             torch.arange(0, 2 * self.max_disp + 1, self.stride)])

        # For each offset pair, compute correlation between first feature map and "shifted" second feature map
        corr_results = [torch.mean(in1 * in2_padded[:, :, dy:dy + height, dx:dx + width], dim=1, keepdim=True)
                            for dx, dy in zip(offset_x.reshape(-1), offset_y.reshape(-1))]

        # Concatenate all results on channel dimension
        output = torch.cat(corr_results, dim=1)
        return output

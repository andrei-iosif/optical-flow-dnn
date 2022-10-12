import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

from flownet.model.corr_torch import CorrTorch


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_batch_norm=False, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2)

        if self.use_batch_norm:
            self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.use_batch_norm:
            return self.activation(self.batch_norm(self.conv(x)))
        else:
            return self.activation(self.conv(x))


class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, activation=F.relu):
        super().__init__()
        self.activation = activation
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=1)

    def forward(self, x):
        return self.activation(self.upconv(x))


class FlowNetCorr(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.conv_1     = ConvBlock(3, 64, kernel_size=7, stride=2)
        self.conv_2     = ConvBlock(64, 128, kernel_size=5, stride=2)
        self.conv_3     = ConvBlock(128, 256, kernel_size=5, stride=2)
        self.conv_redir = ConvBlock(256, 32, kernel_size=1, stride=1)

        self.corr_layer = CorrTorch(max_disp=20)

        # In the original paper, the conv layers with stride 2 are: conv_1, conv_2, conv_3, conv_4, conv_5, conv_6
        self.conv_3_1 = ConvBlock(473, 256, kernel_size=3)
        self.conv_4   = ConvBlock(256, 512, kernel_size=3, stride=2)
        self.conv_4_1 = ConvBlock(512, 512, kernel_size=3)
        self.conv_5   = ConvBlock(512, 512, kernel_size=3, stride=2)
        self.conv_5_1 = ConvBlock(512, 512, kernel_size=3)
        self.conv_6   = ConvBlock(512, 1024, kernel_size=3, stride=2)

        # Decoder
        self.deconv_5 = UpConvBlock(1024, 512, kernel_size=4, stride=2)
        self.deconv_4 = UpConvBlock(512 + 512, 256, kernel_size=4, stride=2)
        self.deconv_3 = UpConvBlock(256 + 512 + 2, 128, kernel_size=4, stride=2)
        self.deconv_2 = UpConvBlock(128 + 256 + 2, 64, kernel_size=4, stride=2)

        # Flow prediction
        self.flow_pred_5 = nn.ConvTranspose2d(512 + 512, 2, kernel_size=3, stride=1, padding=1)
        self.flow_pred_4 = nn.ConvTranspose2d(256 + 512 + 2, 2, kernel_size=3, stride=1, padding=1)
        self.flow_pred_3 = nn.ConvTranspose2d(128 + 256 + 2, 2, kernel_size=3, stride=1, padding=1)

        # Flow upsampling
        self.upsample_flow_5 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.upsample_flow_4 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)
        self.upsample_flow_3 = nn.ConvTranspose2d(2, 2, kernel_size=4, stride=2, padding=1)

        # Output
        self.out_conv = nn.Conv2d(64 + 128 + 2, 2, kernel_size=1, stride=1)
        self.upsample_bilinear = nn.UpsamplingBilinear2d(scale_factor=4)

    def forward(self, img1, img2):
        # Separate branches for image 1 and image 2
        conv_1_out_1 = self.conv_1(img1)
        conv_2_out_1 = self.conv_2(conv_1_out_1)
        conv_3_out_1 = self.conv_3(conv_2_out_1)

        conv_1_out_2 = self.conv_1(img2)
        conv_2_out_2 = self.conv_2(conv_1_out_2)
        conv_3_out_2 = self.conv_3(conv_2_out_2)

        # Correlation layer
        corr_out = self.corr_layer(conv_3_out_1, conv_3_out_2)

        # Concatenate with conv from top branch
        conv_redir_out = self.conv_redir(conv_3_out_1)
        conv_3_1_in = torch.cat([corr_out, conv_redir_out], dim=1)

        conv_3_1_out = self.conv_3_1(conv_3_1_in)
        conv_4_out   = self.conv_4(conv_3_1_out)
        conv_4_1_out = self.conv_4_1(conv_4_out)
        conv_5_out   = self.conv_5(conv_4_1_out)
        conv_5_1_out = self.conv_5_1(conv_5_out)
        conv_6_out   = self.conv_6(conv_5_1_out)

        deconv_5_out = self.deconv_5(conv_6_out)
        concat_5 = torch.cat([deconv_5_out, conv_5_1_out], dim=1)
        flow_5 = self.flow_pred_5(concat_5)
        up_flow_5 = self.upsample_flow_5(flow_5)

        deconv_4_out = self.deconv_4(concat_5)
        concat_4 = torch.cat([deconv_4_out, conv_4_1_out, up_flow_5], dim=1)
        flow_4 = self.flow_pred_4(concat_4)
        up_flow_4 = self.upsample_flow_4(flow_4)

        deconv_3_out = self.deconv_3(concat_4)
        concat_3 = torch.cat([deconv_3_out, conv_3_1_out, up_flow_4], dim=1)
        flow_3 = self.flow_pred_3(concat_3)
        up_flow_3 = self.upsample_flow_3(flow_3)

        deconv_2_out = self.deconv_2(concat_3)
        concat_2 = torch.cat([deconv_2_out, conv_2_out_1, up_flow_3], dim=1)

        out_flow = self.out_conv(concat_2)
        out_flow = self.upsample_bilinear(out_flow)

        return out_flow


if __name__ == "__main__":
    model = FlowNetCorr()
    # summary(model, input_size=((8, 3, 384, 448), (8, 3, 384, 448)))
    # summary(model, input_size=((8, 3, 375, 1242), (8, 3, 375, 1242)))
    summary(model, input_size=((8, 3, 384, 768), (8, 3, 384, 768)))


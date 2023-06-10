import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn='group', stride=1):
        """ Initialize residual block

        Args:
            in_channels (int): Number of channels of input tensor.
            out_channels (int): Number of channels of output tensor.
            norm_fn (str, optional): Type of normalization used. Defaults to 'group'.
            stride (int, optional): Stride value for first convolutional layer 
                (determines whether the block downsamples the input or not). Defaults to 1.
        """
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == 'group':
            num_groups = out_channels // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(out_channels)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels)
            self.norm2 = nn.InstanceNorm2d(out_channels)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(out_channels)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        # If stride is larger than 1, add another convolution -> downsample on the residual path
        if stride == 1:
            self.downsample = None        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), self.norm3)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, norm_fn='group', stride=1):
        """ Initialize bottleneck block

        Args:
            in_channels (int): Number of channels of input tensor.
            out_channels (int): Number of channels of output tensor.
            norm_fn (str, optional): Type of normalization used. Defaults to 'group'.
            stride (int, optional): Stride value for first convolutional layer 
                (determines whether the block downsamples the input or not). Defaults to 1.
        """
        super(BottleneckBlock, self).__init__()

        # Bottleneck blocks have a stack of 3 layers (1x1, 3x3, 1x1)
        # The purpose of the 1x1 layers is to reduce and then restore the size of channel dimension
        # => reduced number of parameters
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        if norm_fn == 'group':
            num_groups = out_channels // 8
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(out_channels//4)
            self.norm2 = nn.BatchNorm2d(out_channels//4)
            self.norm3 = nn.BatchNorm2d(out_channels)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(out_channels)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(out_channels//4)
            self.norm2 = nn.InstanceNorm2d(out_channels//4)
            self.norm3 = nn.InstanceNorm2d(out_channels)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(out_channels)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        # If stride is larger than 1, add another convolution -> downsample on the residual path
        if stride == 1:
            self.downsample = None
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride), self.norm4)

    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)


class BasicEncoder(nn.Module):
    """ Encoder module used in regular RAFT. """

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        """ Initialize module.

        Args:
            output_dim (int, optional): Number of channels in output tensor. Defaults to 128.
            norm_fn (str, optional): Type of normalization to use at input layer and in residual blocks. Defaults to 'batch'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn

        # Normalization for input layer
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Create input convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # Create residual blocks
        self.in_planes = 64
        self.layer1 = self._make_layer(64, stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # Create output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        # Optionally, add dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # Initialize parameters of convolutional and norm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """ Create group of 2 residual blocks. Optionally, the first block can downsample the input, if stride value is changed.

        Args:
            dim (int): Number of output channels of one residual block.
            stride (int, optional): Stride to be used for the first residual block. Defaults to 1.

        Returns:
            Group of residual blocks
        """
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # Feature encoder is applied to both images
        # => concatenate images on batch dimension and extract features for both in a single forward pass
        # => batch size for this module is doubled
        batch_dim = None
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # Input convolution [B, 3, H, W] -> [B, 64, H//2, W//2]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Residual blocks at 1/2 resolution [B, 64, H//2, W//2] -> [B, 64, H//2, W//2]
        x = self.layer1(x)

        # Residual blocks at 1/4 resolution [B, 64, H//2, W//2] -> [B, 96, H//4, W//4]
        x = self.layer2(x)

        # Residual blocks at 1/8 resolution [B, 96, H//4, W//4] -> [B, 128, H//8, W//8]
        x = self.layer3(x)

        # Output convolution [B, 128, H//8, W//8] -> [B, 256, H//8, W//8]
        x = self.conv2(x)

        # Optional dropout layer
        if self.dropout is not None:
            x = self.dropout(x)

        # If two images were given as input, split result feature maps on batch dimension
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


class SmallEncoder(nn.Module):
    """ Encoder module used in small version of RAFT. Residual blocks are replaced by bottleneck residual blocks. """

    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        """ Initialize module.

        Args:
            output_dim (int, optional): Number of channels in output tensor. Defaults to 128.
            norm_fn (str, optional): Type of normalization to use at input layer and in bottleneck blocks. Defaults to 'batch'.
            dropout (float, optional): Dropout probability. Defaults to 0.0.
        """
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        # Normalization for input layer
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        # Create input convolution
        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        # Create bottleneck blocks
        self.in_channels = 32
        self.layer1 = self._make_layer(32, stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        # Optionally, add dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        # Create output convolution
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        # Initialize parameters of convolutional and norm layers
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        """ Create group of 2 bottleneck blocks. Optionally, the first block can downsample the input, if stride value is changed.

        Args:
            dim (int): Number of output channels of one bottleneck block
            stride (int, optional): Stride to be used for the first bottleneck block. Defaults to 1.

        Returns:
            Group of bottleneck blocks
        """
        layer1 = BottleneckBlock(self.in_channels, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_channels = dim
        return nn.Sequential(*layers)

    def forward(self, x):
        # Feature encoder is applied to both images
        # => concatenate images on batch dimension and extract features for both in a single forward pass
        # => batch size for this module is doubled
        batch_dim = None
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        # Input convolution [B, 3, H, W] -> [B, 32, H//2, W//2]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        # Bottleneck blocks at 1/2 resolution [B, 32, H//2, W//2] -> [B, 32, H//2, W//2]
        x = self.layer1(x)

        # Bottleneck blocks at 1/4 resolution [B, 32, H//2, W//2] -> [B, 64, H//4, W//4]
        x = self.layer2(x)

        # Bottleneck blocks at 1/8 resolution [B, 64, H//4, W//4] -> [B, 96, H//8, W//8]
        x = self.layer3(x)

        # Output convolution [B, 96, H//8, W//8] -> [B, 128, H//8, W//8]
        x = self.conv2(x)

        # Optional dropout layer (not mentioned in paper)
        if self.training and self.dropout is not None:
            x = self.dropout(x)

        # If two images were given as input, split result feature maps on batch dimension
        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)

        return x


if __name__ == "__main__":
    from torchinfo import summary

    bottleneck_block = BottleneckBlock(32, 64, stride=2)
    residual_block = ResidualBlock(32, 64, stride=2)

    summary(bottleneck_block, input_size=(1, 32, 128, 128))
    summary(residual_block, input_size=(1, 32, 128, 128))

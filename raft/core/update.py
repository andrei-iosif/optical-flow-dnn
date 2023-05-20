import torch
import torch.nn as nn
import torch.nn.functional as F


class FlowHead(nn.Module):
    """ Flow decoder module. Receives as input the hidden state from the ConvGRU. """
    def __init__(self, input_dim=128, hidden_dim=256):
        """ Initialize module.

        Args:
            input_dim (int, optional): Number of input channels. Defaults to 128.
            hidden_dim (int, optional): Number of channels from first layer output. Defaults to 256.
        """
        super(FlowHead, self).__init__()
        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)
        self.conv2 = nn.Conv2d(hidden_dim, 2, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.conv2(self.relu(self.conv1(x)))


class FlowHeadWithUncertainty(nn.Module):
    """ Predicts residual flow, as a probability distribution.

    If we assume the output flow has Gaussian distribution, we predict both the mean and the variance of that distribution.
    """
    def __init__(self, input_dim=128, hidden_dim=256, log_variance=False):
        super(FlowHeadWithUncertainty, self).__init__()
        self.log_variance = log_variance

        self.conv1 = nn.Conv2d(input_dim, hidden_dim, 3, padding=1)

        # Double the number of output channels => mean and variance for both flow components
        self.conv2 = nn.Conv2d(hidden_dim, 4, 3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.elu = nn.ELU()

    def forward(self, x):
        x = self.conv2(self.relu(self.conv1(x)))
        mean, var = x[:, :2, :, :], x[:, 2:, :, :]

        if self.log_variance:
            # Predict log(var) => no special activation
            return mean, var
        else:
            # Predict variance => need exponential activation to ensure positive values
            # return mean, torch.exp(var)

            # ELU activation
            var = self.elu(var) + 1 + 1e-15
            return mean, var


class ConvGRU(nn.Module):
    """ GRU layer with convolutional inputs and hidden state. """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(ConvGRU, self).__init__()
        self.convz = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convr = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)
        self.convq = nn.Conv2d(hidden_dim+input_dim, hidden_dim, 3, padding=1)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)

        z = torch.sigmoid(self.convz(hx))
        r = torch.sigmoid(self.convr(hx))
        q = torch.tanh(self.convq(torch.cat([r*h, x], dim=1)))

        h = (1-z) * h + z * q
        return h


class SepConvGRU(nn.Module):
    """ ConvGru layer with separable convolutions. """
    def __init__(self, hidden_dim=128, input_dim=192+128):
        super(SepConvGRU, self).__init__()
        self.convz1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convr1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))
        self.convq1 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (1,5), padding=(0,2))

        self.convz2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convr2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))
        self.convq2 = nn.Conv2d(hidden_dim+input_dim, hidden_dim, (5,1), padding=(2,0))

    def forward(self, h, x):
        # horizontal
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz1(hx))
        r = torch.sigmoid(self.convr1(hx))
        q = torch.tanh(self.convq1(torch.cat([r*h, x], dim=1)))        
        h = (1-z) * h + z * q

        # vertical
        hx = torch.cat([h, x], dim=1)
        z = torch.sigmoid(self.convz2(hx))
        r = torch.sigmoid(self.convr2(hx))
        q = torch.tanh(self.convq2(torch.cat([r*h, x], dim=1)))       
        h = (1-z) * h + z * q

        return h


class SmallMotionEncoder(nn.Module):
    """ Encodes correlation feature map and estimated flow from previous iteration. """

    def __init__(self, args):
        super(SmallMotionEncoder, self).__init__()

        # Number of channels for correlation feature map
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2

        # Convolution for correlation feature map
        self.convc1 = nn.Conv2d(cor_planes, 96, 1, padding=0)

        # Convolutions for flow
        self.convf1 = nn.Conv2d(2, 64, 7, padding=3)
        self.convf2 = nn.Conv2d(64, 32, 3, padding=1)

        # Output convolution
        self.conv = nn.Conv2d(128, 80, 3, padding=1)

    def forward(self, flow, corr):
        """
        Args:
            flow (torch.Tensor): Optical flow estimation from previous iteration, shape [B, 2, H, W]
            corr (torch.Tensor): Correlation feature map, shape [B, corr_levels * (2*r+1)^2, H, W]

        Returns:
            Motion feature map, shape [B, 80, H, W]
        """
        # Correlation feature map
        cor = F.relu(self.convc1(corr))

        # Flow
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # Concatenate correlation feature map and flow
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))

        return torch.cat([out, flow], dim=1)


class BasicMotionEncoder(nn.Module):
    """ Encodes correlation feature map and estimated flow from previous iteration. """

    def __init__(self, args):
        super(BasicMotionEncoder, self).__init__()

        # Number of channels for correlation feature map
        cor_planes = args.corr_levels * (2*args.corr_radius + 1)**2

        # Convolutions for correlation feature map
        self.convc1 = nn.Conv2d(cor_planes, 256, 1, padding=0)
        self.convc2 = nn.Conv2d(256, 192, 3, padding=1)

        # Convolutions for flow
        self.convf1 = nn.Conv2d(2, 128, 7, padding=3)
        self.convf2 = nn.Conv2d(128, 64, 3, padding=1)

        # Output convolution
        self.conv = nn.Conv2d(64+192, 128-2, 3, padding=1)

    def forward(self, flow, corr):
        # Correlation features
        cor = F.relu(self.convc1(corr))
        cor = F.relu(self.convc2(cor))

        # Flow
        flo = F.relu(self.convf1(flow))
        flo = F.relu(self.convf2(flo))

        # Concatenate correlation feature map and flow
        cor_flo = torch.cat([cor, flo], dim=1)
        out = F.relu(self.conv(cor_flo))
        return torch.cat([out, flow], dim=1)


class SmallUpdateBlock(nn.Module):
    def __init__(self, args, hidden_dim=96, dropout=0.0):
        super(SmallUpdateBlock, self).__init__()
        self.encoder = SmallMotionEncoder(args)
        self.gru = ConvGRU(hidden_dim=hidden_dim, input_dim=82+64)
        self.flow_head = FlowHead(hidden_dim, hidden_dim=128)

        # Optionally, add dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, net, inp, corr, flow):
        motion_features = self.encoder(flow, corr)
        inp = torch.cat([inp, motion_features], dim=1)
        net = self.gru(net, inp)
        delta_flow = self.flow_head(net)

        return net, None, delta_flow


class BasicUpdateBlock(nn.Module):
    """ Update block used for iterative flow refinement in base RAFT. """

    def __init__(self, args, hidden_dim=128, input_dim=128, dropout=0.0):
        """
        Args:
            args (argparse.Namespace): Command-line arguments
            hidden_dim (int, optional): Number of channels for ConvGRU hidden dimension. Defaults to 128.
            input_dim (int, optional): Number of channels for input dimension. Defaults to 128.
            dropout (float, optional): Dropout probability. Defaults to 0.0
        """
        super(BasicUpdateBlock, self).__init__()
        self.args = args

        # Motion encoder
        self.encoder = BasicMotionEncoder(args)

        # ConvGRU with separable convolutions
        self.gru = SepConvGRU(hidden_dim=hidden_dim, input_dim=128+hidden_dim)

        if self.args.uncertainty:
            self.flow_head = FlowHeadWithUncertainty(hidden_dim, hidden_dim=256, log_variance=args.log_variance)
        else:
            self.flow_head = FlowHead(hidden_dim, hidden_dim=256)

        # Upsampling mask decoder
        self.mask = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64*9, 1, padding=0))
        
        # Optionally, add dropout
        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

    def forward(self, net, context_fmap, corr, flow):
        """ `
        Args:
            net (torch.Tensor): Hidden state of ConvGRU, shape [B, hidden_dim, H, W]
            context_fmap (torch.Tensor): Context feature map, shape [B, input_dim, H, W]
            corr (torch.Tensor): Correlation feature map, shape [B, corr_levels * (2*r+1)^2, H, W]
            flow (torch.Tensor): Optical flow estimation from previous iteration, shape [B, 2, H, W]

        Returns:
            ConvGRU hidden state, upsampling mask, residual flow
        """
        # Encode motion features
        motion_features = self.encoder(flow, corr)

        # Concatenate context features and motion features
        context_fmap = torch.cat([context_fmap, motion_features], dim=1)

        # Pass through GRU
        net = self.gru(net, context_fmap)

        # Optional dropout layer applied on GRU hidden state
        if self.dropout is not None:
            net = self.dropout(net)

        # Decode residual flow (and optionally, the uncertainty)
        delta_flow = self.flow_head(net)

        # Not sure if this is really necessary
        # scale mask to balence gradients
        # mask = .25 * self.mask(net)

        # Decode upsampling mask
        mask = self.mask(net)

        return net, mask, delta_flow

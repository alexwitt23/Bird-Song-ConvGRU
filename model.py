""" Combination of CNN and GRU inspired by
https://www.merl.com/publications/docs/TR2018-137.pdf """

from typing import Tuple

import torch
from torch import nn
import torchvision


class SongIdentifier(nn.Module):
    """ This class brings together a feature extractor, ResNet is used here, 
    plus a ConvGRU. """
    
    def __init__(
        self,
        num_classes: int,
        hidden_channels: int = 32,
        num_of_frames: int = 5,
    ) -> None:

        super().__init__()
        # Use a prebult resnet model. Take off the last AdaptiveAvgPooling and linear layer to
        # preserve dimensionality of features.
        self.cnn = torch.nn.Sequential(
            *list(torchvision.models.resnet18(pretrained=False).children())[:-2]
        )
        cnn_output_channels = list(self.cnn.children())[-1][-1].conv2.out_channels

        # Create ConvGRU model
        self.gru = ConvGRU(
            channels_in=cnn_output_channels,
            hidden_channels=hidden_channels,
            kernel_size=(3, 3),
            num_layers=1,
        )

        # TODO(alex) is there anyway to determine these output feature dimensions programatically?
        # The FC/softmax layer to make class predictions
        self.head = nn.Sequential(
            nn.Linear(hidden_channels * 15 * 4, num_classes), nn.LogSoftmax(dim=1),
        )

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Comes out as [frames, features]. Need to permute to [features, frames]
        assert (
            len(x.shape) == 5
        ), "Need input in [batch, num_frames, channels, width, height]"

        cnn_output = []
        for i in range(x.shape[0]):
            cnn_output.append(self.cnn(x[i]).unsqueeze(0))

        cnn_output = torch.cat(cnn_output)
        gru_output = self.gru(cnn_output)
        gru_output = gru_output.reshape(gru_output.shape[0], -1)
        prediction = self.head(gru_output)
        return prediction


class ConvGRUCell(nn.Module):
    def __init__(
        self,
        channels_in: int,
        hidden_channels: int,
        kernel_size: Tuple[int, int],
        bias: bool = True,
    ) -> None:
        """ A ConvGRU cell.
        
        Inputs:
            channels_in: Number of channels of input tensor, or outputted from
                the feature extraction.
            hidden_channels: Number of convolutional output channels in the GRU layer.
            kernel_size: Convolutional kernel dimensions.
            bias: Whether or not to add the bias in the GRU convolutions.
        """
        super().__init__()
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_channels = hidden_channels
        self.bias = bias

        # Update gate. This is really 3 separate conv layers
        # but it is combined into one layer for simplification.
        self.input_update = nn.Conv2d(
            in_channels=channels_in,  # Convolve over input x
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )
        self.input_reset = nn.Conv2d(
            in_channels=channels_in,  # Convolve over input x
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )
        self.input_new = nn.Conv2d(
            in_channels=channels_in,  # Convolve over input x
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

        # Hidden layer weights
        self.hidden_update = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )
        self.hidden_reset = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )
        self.hidden_new = nn.Conv2d(
            in_channels=self.hidden_channels,
            out_channels=self.hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding,
            bias=True,
        )

    def __call__(self, x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        """ The forward pass on a ConvGRU.
        
        Inputs:
            x: The extracted features from the cnn backbone
            h: The previous hidden state

        """
        # Perform the multiplcation of weights on inputs
        input_update = self.input_update(x)
        input_reset = self.input_reset(x)
        input_new = self.input_new(x)

        # Perform the multiplcation of weights on incoming hidden layer
        hidden_update = self.hidden_update(h)
        hidden_reset = self.hidden_reset(h)
        hidden_new = self.hidden_new(h)

        reset_gate = torch.sigmoid(input_reset + hidden_reset)
        update_gate = torch.sigmoid(input_update + hidden_update)
        new_gate = torch.tanh(input_new + reset_gate * hidden_new)

        # New hidden state
        h_new = update_gate * h + (1 - update_gate) * new_gate

        return h_new


class ConvGRU(nn.Module):
    def __init__(
        self,
        channels_in: int,
        hidden_channels: int,
        kernel_size: Tuple[int, int],
        num_layers: int,
        cuda: bool = True,
    ) -> None:
        """ Initialize the Convolutional GRU model.

        Args:
            input_size: The number of filters from the cnn backbone.
            hidden_channels: The number of convolutional filters in the hidden layer
            kernel_size: The kernel size of the hidden cnn.
            num_layers: the number of stacked ConvGRUs.
        """

        super().__init__()
        self.channels_in = channels_in
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.cuda = cuda

        # Create the list of number of cells specified
        self.cell_list: torch.nn.ModuleList[ConvGRUCell] = nn.ModuleList([])
        for _ in range(self.num_layers):
            self.cell_list.append(
                ConvGRUCell(
                    channels_in=self.channels_in,
                    hidden_channels=self.hidden_channels,
                    kernel_size=self.kernel_size,
                    bias=True,
                )
            )

    def __call__(self, x: torch.tensor, h: torch.Tensor = None) -> torch.Tensor:
        """ Forward pass through the ConvGRU model.
        Args: 
            x: A tensor in (batch, sequence_len, channels, height, width)
            h: A tensor representing the hidden state.
        """

        # Initialize the hidden layer if this is the first pass through.
        if h is None:
            h = torch.zeros(
                size=(x.shape[0], self.hidden_channels, x.shape[3], x.shape[4])
            )
            if self.cuda:
                h = h.cuda()

        # Loop over the number of ConvGRU layers specified
        for layer_idx in range(self.num_layers):
            # Loop over the number of items in the sequence, i.e. the number of frames.
            for seq in range(x.shape[1]):
                h = self.cell_list[layer_idx](x[:, seq, :, :, :], h)

        return h

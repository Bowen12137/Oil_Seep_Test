import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse


class UNet(nn.Module):
    def __init__(self, input_channels, output_classes):
        """
        Initialize the U-Net model with the given input channels and output classes.

        :param input_channels: Number of channels in the input image.
        :param output_classes: Number of output classes for segmentation.
        """
        super(UNet, self).__init__()

        # Encoder: Convolutional blocks to extract features
        self.encoder_block1 = self.conv_block(input_channels, 64)
        self.encoder_block2 = self.conv_block(64, 128)
        self.encoder_block3 = self.conv_block(128, 256)
        self.encoder_block4 = self.conv_block(256, 512)

        # Bottleneck layer
        self.bottleneck = self.conv_block(512, 1024)

        # Decoder: Transposed convolutions for upsampling
        self.upconv_block4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder_block4 = self.conv_block(1024, 512)
        self.upconv_block3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder_block3 = self.conv_block(512, 256)
        self.upconv_block2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder_block2 = self.conv_block(256, 128)
        self.upconv_block1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder_block1 = self.conv_block(128, 64)

        # Final output layer
        self.final_conv = nn.Conv2d(64, output_classes, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """
        A helper function to create a convolutional block consisting of two convolution layers,
        each followed by batch normalization and ReLU activation.

        :param in_channels: Number of input channels.
        :param out_channels: Number of output channels.
        :return: A sequential model with two Conv-BatchNorm-ReLU layers.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the U-Net model.

        :param x: Input tensor.
        :return: Output tensor after the final convolution.
        """
        # Encoding
        enc1 = self.encoder_block1(x)
        enc2 = self.encoder_block2(F.max_pool2d(enc1, 2))
        enc3 = self.encoder_block3(F.max_pool2d(enc2, 2))
        enc4 = self.encoder_block4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Decoding
        dec4 = self.upconv_block4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder_block4(dec4)
        dec3 = self.upconv_block3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder_block3(dec3)
        dec2 = self.upconv_block2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder_block2(dec2)
        dec1 = self.upconv_block1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder_block1(dec1)

        # Final output layer
        output = self.final_conv(dec1)
        return output


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.

    :param model: The neural network model.
    :return: Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Set up argparse for configuration
    parser = argparse.ArgumentParser(description='U-Net Model Configuration')
    parser.add_argument('--input_channels', type=int, default=1, help='Number of input channels. Default is 1.')
    parser.add_argument('--output_classes', type=int, default=8, help='Number of output classes. Default is 8.')
    parser.add_argument('--show_params', action='store_true', help='Whether to show model parameters.')
    parser.add_argument('--show_config', action='store_true', help='Whether to show model configuration.')

    args = parser.parse_args()

    # Initialize model
    model = UNet(input_channels=args.input_channels, output_classes=args.output_classes).cuda()

    # Display model configuration
    if args.show_config:
        print("Model configuration:")
        print(f"Input channels: {args.input_channels}")
        print(f"Output classes: {args.output_classes}")

    # Display parameter count
    if args.show_params:
        num_params = count_parameters(model)
        print(f"Total number of trainable parameters: {num_params}")

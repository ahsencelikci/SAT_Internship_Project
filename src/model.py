import torch.nn as nn
import torch.nn.functional as F
from constant import HEIGHT, WIDTH, N_CLASS

class FoInternNet(nn.Module):
    def __init__(self, input_size, n_classes):
        super(FoInternNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes

        # LAYERS
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=n_classes, kernel_size=1, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        """This function feeds input data into the model layers defined.
        Args:
            x : input data
        """
        #################################################################
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Decoder
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        x = self.conv4(x)
        ###################################################################
        
        return x

if __name__ == '__main__':
    model = FoInternNet(input_size=(HEIGHT, WIDTH), n_classes=N_CLASS)
    print(model)

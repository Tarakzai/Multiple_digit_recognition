import torch
import torch.nn as nn
from torch.nn import functional as F

class Attention(nn.Module):
    def __init__(self, in_features, out_features):
        super(Attention, self).__init__()
        self.attention = nn.Linear(in_features, out_features)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, in_features)
        attention_weights = F.softmax(self.attention(x), dim=1)
        return attention_weights * x

class DigitModel_improved(nn.Module):
    def __init__(self, num_chars):
        super(DigitModel_improved, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=(3,3), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(128)
        self.max_pool_1 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv2 = nn.Conv2d(128, 64, kernel_size=(3,3), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool_2 = nn.MaxPool2d(kernel_size=(2,2))

        self.conv3 = nn.Conv2d(64, 32, kernel_size=(3,3), padding=(1,1))
        self.bn3 = nn.BatchNorm2d(32)
        self.max_pool_3 = nn.MaxPool2d(kernel_size=(2,2))

        self.linear_1 = nn.Linear(288, 64)
        self.drop_1 = nn.Dropout(0.2)

        self.gru = nn.GRU(64, 32, bidirectional=True, num_layers=3, dropout=0.25)
        self.attention = Attention(64, 64)

        self.output = nn.Linear(64, num_chars+1)

    def forward(self, images, labels=None):
        n, c, h, w = images.size()
        print(n,c,h,w)
        x = F.relu(self.bn1(self.conv1(images)))
        print(x.size())
        x = self.max_pool_1(x)
        print(x.size())

        x = F.relu(self.bn2(self.conv2(x)))
        print(x.size())
        x = self.max_pool_2(x)
        print(x.size())

        x = F.relu(self.bn3(self.conv3(x)))
        print(x.size())
        x = self.max_pool_3(x)
        print(x.size())

        x = x.permute(0, 3, 1, 2)
        print(x.size())
        x = x.view(n, x.size(1), -1)
        print(x.size())

        x = F.relu(self.linear_1(x))
        
        x = self.drop_1(x)
        print(x.size())

        x, _ = self.gru(x)
        print(x.size())
        x = self.attention(x)
        print(x.size())
        x = self.output(x)
        print(x.size())
        x = x.permute(1, 0, 2)
        print(x.size())

        if labels is not None:
            log_probs = F.log_softmax(x, 2)

            # Exclude the padding symbol '10' from the CTC loss calculation
            non_padding_labels = labels != 10

            input_lengths = torch.full(
                size=(n,), fill_value=log_probs.size(0), dtype=torch.int32
            )

            target_lengths = torch.sum(non_padding_labels, dim=1, dtype=torch.int32)

            loss = nn.CTCLoss(blank=10)(
                log_probs, labels, input_lengths, target_lengths
            )

            return x, loss

        return x, None

if __name__ == "__main__":
    dm = DigitModel_improved(11)
    img = torch.rand((1, 3, 28, 140))
    x, _ = dm(img, torch.rand((1, 5)))
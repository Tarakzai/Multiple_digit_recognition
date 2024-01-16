import torch
import torch.nn as nn
from torch.nn import functional as F

class DigitModel(nn.Module):
    def __init__(self,num_chars):
       super(DigitModel,self).__init__()
       self.conv1 = nn.Conv2d(3,128, kernel_size=(3,3),padding=(1,1))
       self.max_pool_1= nn.MaxPool2d(kernel_size=(2,2))
       self.conv2 = nn.Conv2d(128,64, kernel_size=(3,3),padding=(1,1))
       self.max_pool_2= nn.MaxPool2d(kernel_size=(2,2))
       self.linear_1= nn.Linear(448,64)
       self.drop_1 = nn.Dropout(0.1)
       self.gru = nn.GRU(64,32, bidirectional=True,num_layers=2,dropout=0.25)
       self.output = nn.Linear(64,num_chars+1)
    def forward(self,images,labels=None):
        n , c , h , w = images.size()
        
        x = F.relu(self.conv1(images))
        
        x = self.max_pool_1(x)
        
        x = F.relu(self.conv2(x))
        
        x = self.max_pool_2(x)
        
        x = x.permute(0,3,1,2)
        x = x.view(n, x.size(1),-1)
 
        x = self.linear_1(x)
        
        x = self.drop_1(x)
        x,_ = self.gru(x)
 
        x = self.output(x)
      
        x = x.permute(1,0 ,2)
        
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




#ss DigitModel(nn.Module):
# def __init__(self, num_chars):
#     super(DigitModel, self).__init__()
#     self.conv1 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
#     self.bn1 = nn.BatchNorm2d(128)
#     self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
#     self.bn2 = nn.BatchNorm2d(128)
#     self.max_pool_1 = nn.MaxPool2d(kernel_size=2)
#
#     self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
#     self.bn3 = nn.BatchNorm2d(64)
#     self.max_pool_2 = nn.MaxPool2d(kernel_size=2)
#
#     self.linear_1 = nn.Linear(448, 64)
#     self.drop_1 = nn.Dropout(0.2)
#
#     self.lstm = nn.LSTM(64, 32, bidirectional=True, num_layers=2, dropout=0.25)
#     self.output = nn.Linear(64, num_chars + 1)
#
# def forward(self, images, labels=None):
#     n, c, h, w = images.size()
#
#     x = F.relu(self.bn1(self.conv1(images)))
#     x = self.max_pool_1(x)
#
#     x = F.relu(self.bn2(self.conv2(x)))
#     x = self.max_pool_1(x)  # Reusing max pooling layer
#
#     x = F.relu(self.bn3(self.conv3(x)))
#     x = self.max_pool_2(x)
#
#     x = x.permute(0, 3, 1, 2)
#     x = x.view(n, x.size(1), -1)
#
#     x = self.linear_1(x)
#     x = self.drop_1(x)
#
#     x, _ = self.lstm(x)
#     x = self.output(x)
#     x = x.permute(1, 0, 2)
#
#     if labels is not None:
#         log_probs = F.log_softmax(x, 2)
#         input_lengths = torch.full(size=(n,), fill_value=log_probs.size(0), dtype=torch.int32)
#         target_lengths = torch.sum(labels != 10, dim=1, dtype=torch.int32)  # Assuming 10 is the padding value
#         loss = nn.CTCLoss(blank=10)(log_probs, labels, input_lengths, target_lengths)
#         return x, loss
#
#     return x, None









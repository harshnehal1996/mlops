from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self):
        super(MyAwesomeModel, self).__init__()
        self.model =  nn.Sequential(nn.Conv2d(1, 2, kernel_size=3, padding=3),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.BatchNorm2d(2),
                      nn.ReLU(),
                      nn.Conv2d(2, 4, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.BatchNorm2d(4),
                      nn.ReLU(),
                      nn.Conv2d(4, 8, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.BatchNorm2d(8),
                      nn.ReLU(),
                      nn.Conv2d(8, 16, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.BatchNorm2d(16),
                      nn.ReLU(),
                      nn.Conv2d(16, 32, kernel_size=3, padding=1),
                      nn.MaxPool2d(kernel_size=2, stride=2),
                      nn.ReLU(),
                      nn.Flatten(),
                      nn.Linear(32, 32),
                      nn.ReLU(),
                      nn.Linear(32, 10),
                      nn.LogSoftmax(dim=-1))
    
    def forward(self, inputs):
        return self.model(inputs)

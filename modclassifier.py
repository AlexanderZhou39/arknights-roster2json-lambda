import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 3, 2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(900, 300)
        self.fc2 = nn.Linear(300, 200)
        self.fc3 = nn.Linear(200, 2)

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.load_state_dict(torch.load('models/modclassifier.pth'))
net.eval()

@torch.no_grad()
def classify_module(img):
    predictions = net.forward(transform(img))
    pred_class = torch.argmax(predictions).item()
    if pred_class == 0:
        return "x"
    return "y"
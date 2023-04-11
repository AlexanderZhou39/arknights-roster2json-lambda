import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((30, 30)),
    transforms.ToTensor()
])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(576, 90)
        self.fc2 = nn.Linear(90, 90)
        self.fc3 = nn.Linear(90, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 0) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()
net.load_state_dict(torch.load('models/potclassifier.pth'))
net.eval()

@torch.no_grad()
def classify_potential(img):
    predictions = net.forward(transform(img))
    pred_class = torch.argmax(predictions).item()
    return pred_class + 2
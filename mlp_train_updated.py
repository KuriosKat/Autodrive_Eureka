
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import torch.nn.functional as F

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class MLP(nn.Module):  # 랜덤 시드 추가
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),  # 입력 차원 변경
            nn.LeakyReLU(),
            nn.Linear(32, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)
    

class BottleneckFCBlock(nn.Module):
    def __init__(self, in_dim, bottleneck_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, bottleneck_dim)
        self.bn1 = nn.BatchNorm1d(bottleneck_dim)
        self.fc2 = nn.Linear(bottleneck_dim, bottleneck_dim)
        self.bn2 = nn.BatchNorm1d(bottleneck_dim)
        self.fc3 = nn.Linear(bottleneck_dim, in_dim)
        self.bn3 = nn.BatchNorm1d(in_dim)
        self.relu = nn.LeakyReLU()

    def forward(self, x):
        identity = x

        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.fc3(out)
        out = self.bn3(out)

        out += identity  # Skip connection
        out = self.relu(out)
        return out

class ResNet50MLP(nn.Module):
    def __init__(self, input_dim=6):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, 64)  # 입력 확장
        self.relu = nn.LeakyReLU()

        # ResNet-50은 (3 + 4 + 6 + 3) = 16개의 bottleneck block
        self.layer1 = self._make_layer(64, 16, num_blocks=3)
        self.layer2 = self._make_layer(64, 16, num_blocks=4)
        self.layer3 = self._make_layer(64, 16, num_blocks=6)
        self.layer4 = self._make_layer(64, 16, num_blocks=3)

        self.fc_out1 = nn.Linear(64, 16)
        self.fc_out2 = nn.Linear(16, 1)

    def _make_layer(self, in_dim, bottleneck_dim, num_blocks):
        layers = []
        for _ in range(num_blocks):
            layers.append(BottleneckFCBlock(in_dim, bottleneck_dim))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fc_in(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.fc_out1(x)
        x = self.relu(x)
        x = self.fc_out2(x)
        return x



# 데이터 로딩
data = pd.read_csv("driving_data.csv")
X = torch.tensor(data[["front", "left", "right", "speed", "dx_to_goal", "dy_to_goal"]].values, dtype=torch.float32)
y = torch.tensor(data["steering"].values.reshape(-1, 1), dtype=torch.float32)


dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = ResNet50MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 랜덤 시드 추가=42
loss_fn = nn.MSELoss()

# 학습 루프
for epoch in range(100):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mlp_model.pt")
print("MLP 모델 저장 완료")

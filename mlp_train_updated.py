
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

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
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.net(x)

# 데이터 로딩
data = pd.read_csv("driving_data.csv")
X = torch.tensor(data[["front", "left", "right", "speed", "dx_to_goal", "dy_to_goal"]].values, dtype=torch.float32)
y = torch.tensor(data["steering"].values.reshape(-1, 1), dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 랜덤 시드 추가=42
loss_fn = nn.MSELoss()

# 학습 루프
for epoch in range(200):
    for xb, yb in loader:
        pred = model(xb)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "mlp_model.pt")
print("MLP 모델 저장 완료")

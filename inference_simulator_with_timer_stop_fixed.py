
import pygame
import math
import sys
import joblib
import torch
import torch.nn as nn
import numpy as np
import time

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
BLUE  = (0, 128, 255)
GREEN = (0, 255, 0)

class Obstacle:
    def __init__(self, x, y, w, h):
        self.rect = pygame.Rect(x, y, w, h)

    def draw(self, win):
        pygame.draw.rect(win, BLACK, self.rect)

    def collide_point(self, x, y):
        return self.rect.collidepoint(x, y)

class Radar:
    def __init__(self):
        self.angles = [0, -30, 30]
        self.length = 150
        self.readings = []

    def scan(self, x, y, car_angle, obstacles):
        self.readings = []
        for offset in self.angles:
            angle = math.radians(car_angle + offset)
            for i in range(self.length):
                rx = x + math.cos(angle) * i
                ry = y + math.sin(angle) * i
                hit = False
                for obs in obstacles:
                    if obs.collide_point(rx, ry):
                        hit = True
                        break
                if hit:
                    self.readings.append(i)
                    break
            else:
                self.readings.append(self.length)
        return self.readings

    def draw(self, win, x, y, car_angle):
        for offset, dist in zip(self.angles, self.readings):
            angle = math.radians(car_angle + offset)
            end_x = x + math.cos(angle) * dist
            end_y = y + math.sin(angle) * dist
            pygame.draw.line(win, RED, (x, y), (end_x, end_y), 2)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
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




class Car:
    def __init__(self, x, y, goal, model_type="knn"):
        self.x = x
        self.y = y
        self.angle = -90
        self.base_speed = 2
        self.speed = 2
        self.radar = Radar()
        self.goal = goal
        self.model_type = model_type
        self.reached_goal = False
        self.crashed = False

        if model_type == "knn":
            self.model = joblib.load("knn_model.pkl")
        elif model_type == "rf":
            self.model = joblib.load("rf_model.pkl")
        elif model_type == "et":
            self.model = joblib.load("et_model.pkl")
        elif model_type == "gb":
            self.model = joblib.load("gb_model.pkl")
        elif model_type == "mlp":
            self.model = ResNet50MLP()
            self.model.load_state_dict(torch.load("mlp_model.pt"))
            self.model.eval()
        else:
            raise ValueError("Unknown model_type")

    def move(self, steering):
        self.angle += steering
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def get_features(self, obstacles):
        radar = self.radar.scan(self.x, self.y, self.angle, obstacles)
        dx = self.goal[0] - self.x
        dy = self.goal[1] - self.y
        return radar + [self.speed, dx, dy]

    def control(self, obstacles):
        if self.reached_goal or self.crashed:
            return
        features = self.get_features(obstacles)
        if self.model_type in ["knn", "rf", "et", "gb"]:
            steering = self.model.predict([features])[0]
        elif self.model_type == "mlp":
            inp = torch.tensor([features], dtype=torch.float32)
            steering = self.model(inp).item()
        else:
            steering = 0
        self.move(steering)

        for obs in obstacles:
            if obs.collide_point(self.x, self.y):
                self.crashed = True

    def draw(self, win):
        pygame.draw.circle(win, BLUE, (int(self.x), int(self.y)), 10)
        self.radar.draw(win, self.x, self.y, self.angle)

    def check_goal(self):
        dist = math.hypot(self.x - self.goal[0], self.y - self.goal[1])
        if dist < 15:
            self.reached_goal = True

def run_simulation(model_type="knn"):
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption(f"AI 자율주행 시뮬레이터 ({model_type})")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 36)

    goal = (700, 100)
    car = Car(100, 500, goal, model_type=model_type)

    border_thickness = 10
    obstacles = [
        Obstacle(0, 0, WIDTH, border_thickness),
        Obstacle(0, HEIGHT - border_thickness, WIDTH, border_thickness),
        Obstacle(0, 0, border_thickness, HEIGHT),
        Obstacle(WIDTH - border_thickness, 0, border_thickness, HEIGHT),
        Obstacle(300, 250, 200, 20),
        Obstacle(150, 150, 120, 20),
        Obstacle(500, 200, 180, 20),
        Obstacle(400, 300, 20, 150),
        Obstacle(100, 400, 250, 20),
        Obstacle(500, 400, 200, 20),
        Obstacle(250, 100, 20, 100),
        Obstacle(600, 100, 20, 100),
        Obstacle(650, 300, 20, 150),
        Obstacle(200, 350, 100, 20),

        Obstacle(350, 450, 150, 20),     # 아래쪽 가로 장애물
        Obstacle(100, 100, 20, 100),     # 왼쪽 위 세로 장애물
        Obstacle(720, 150, 20, 200),     # 오른쪽 세로 장애물
        Obstacle(450, 50, 100, 20),      # 위쪽 가로 장애물
        Obstacle(300, 500, 20, 100),     # 맨 아래 세로 장애물
        Obstacle(550, 500, 150, 20),     # 오른쪽 아래 가로 장애물
        Obstacle(400, 150, 20, 100),     # 중앙 위 세로 장애물
        Obstacle(200, 200, 80, 20),      # 중간 왼쪽 가로 장애물
        Obstacle(600, 250, 100, 20),     # 중간 오른쪽 가로 장애물
        Obstacle(100, 500, 200, 20),     # 왼쪽 아래 가로 장애물

        # 새로 추가된 장애물들 (빈 공간 채우기)
        Obstacle(350, 600, 100, 20),       # 중앙 아래쪽 가로 장애물
        # Obstacle(700, 50, 20, 80),         # 오른쪽 위 세로 장애물
        Obstacle(50, 250, 80, 20),         # 왼쪽 중간 가로 장애물
        Obstacle(50, 50, 50, 20),          # 왼쪽 위 구석 가로 장애물
        Obstacle(750, 550, 20, 80),        # 오른쪽 아래 세로 장애물
    ]

    start_time = time.time()
    elapsed = 0

    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            pygame.quit()
            sys.exit()

        car.control(obstacles)
        car.check_goal()

        if not (car.reached_goal or car.crashed):
            elapsed = time.time() - start_time

        win.fill(WHITE)
        for obs in obstacles:
            obs.draw(win)
        car.draw(win)
        pygame.draw.circle(win, GREEN, goal, 15)

        timer_text = font.render(f"Time: {elapsed:.2f}s", True, BLACK)
        win.blit(timer_text, (10, 10))

        result_font = pygame.font.SysFont(None, 48)
        if car.reached_goal:
            result = result_font.render("GOAL REACHED!", True, GREEN)
            win.blit(result, (WIDTH // 2 - 100, HEIGHT // 2 - 30))
        elif car.crashed:
            result = result_font.render("FAILED!", True, RED)
            win.blit(result, (WIDTH // 2 - 70, HEIGHT // 2 - 30))

        pygame.display.update()

if __name__ == "__main__":
    run_simulation("mlp")

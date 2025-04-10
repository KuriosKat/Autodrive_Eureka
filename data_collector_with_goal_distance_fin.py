
import pygame
import math
import csv
import sys

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED   = (255, 0, 0)
BLUE  = (0, 128, 255)
GREEN = (0, 255, 0)

# 장애물 설정(필요시 장애물 추가 가능)
# 장애물은 pygame.Rect 객체로 정의
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

# DummyCar 클래스는 자동차의 위치, 속도, 각도 및 레이더를 포함
# 자동차의 이동 및 상태를 업데이트하는 메서드를 포함
# 자동차는 pygame.Rect 객체로 정의
class DummyCar:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.angle = -90
        self.base_speed = 2
        self.speed = 2
        self.radar = Radar()

    def move(self, control, boost=False):
        self.angle += control
        self.speed = self.base_speed * (2 if boost else 1)
        rad = math.radians(self.angle)
        self.x += math.cos(rad) * self.speed
        self.y += math.sin(rad) * self.speed

    def get_state(self, obstacles):
        return self.radar.scan(self.x, self.y, self.angle, obstacles)

    def draw(self, win):
        pygame.draw.circle(win, BLUE, (int(self.x), int(self.y)), 10)
        self.radar.draw(win, self.x, self.y, self.angle)

    def reached_goal(self, goal):
        return math.hypot(self.x - goal[0], self.y - goal[1]) < 30

def run_data_collection():
    pygame.init()
    WIDTH, HEIGHT = 800, 600
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("데이터 수집 (속도 + 목표거리 포함)")
    clock = pygame.time.Clock()

    car = DummyCar(100, 500)
    goal = (700, 100)
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


    collected_data = []
    running = True
    reached = False

    while running:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            print("ESC 눌림 - 종료합니다.")
            break

        steering = 0
        if keys[pygame.K_LEFT]:
            steering = -5
        elif keys[pygame.K_RIGHT]:
            steering = 5

        boost = keys[pygame.K_b]

        state = car.get_state(obstacles)
        car.move(steering, boost=boost)
        dx = goal[0] - car.x
        dy = goal[1] - car.y
        collected_data.append(state + [steering, car.speed, dx, dy])

        if car.reached_goal(goal):
            reached = True
            running = False

        win.fill(WHITE)
        for obs in obstacles:
            obs.draw(win)
        car.draw(win)
        pygame.draw.circle(win, GREEN, goal, 15)

        if reached:
            font = pygame.font.SysFont(None, 48)
            text = font.render("GOAL REACHED!", True, GREEN)
            win.blit(text, (WIDTH // 2 - 100, HEIGHT // 2 - 20))

        pygame.display.update()

    pygame.quit()
    with open("driving_data.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["front", "left", "right", "steering", "speed", "dx_to_goal", "dy_to_goal"])
        writer.writerows(collected_data)

    print("데이터 저장 완료: driving_data.csv")


# 메인 함수
# pygame을 실행하고 데이터를 수집하는 함수
if __name__ == "__main__":
    run_data_collection()

    

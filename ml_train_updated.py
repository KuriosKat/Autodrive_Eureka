import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
import joblib
from tqdm import tqdm

# 데이터 불러오기
data = pd.read_csv("driving_data.csv")
X = data[["front", "left", "right", "speed", "dx_to_goal", "dy_to_goal"]]
y = data["steering"]

# 데이터 분할
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 모델 리스트
models = {
    "knn_model.pkl": KNeighborsRegressor(n_neighbors=3),
    "rf_model.pkl": RandomForestRegressor(n_estimators=100, random_state=42),
    "et_model.pkl": ExtraTreesRegressor(n_estimators=1000, random_state=42),
    "gb_model.pkl": GradientBoostingRegressor(n_estimators=100, learning_rate=0.001, random_state=42)
}

# tqdm으로 모델 학습 진행 상황 표시
for name in tqdm(models, desc="모델 학습 중"):
    model = models[name]
    model.fit(X_train, y_train)
    joblib.dump(model, name)

print("모델 저장 완료 (knn_model.pkl, rf_model.pkl, et_model.pkl, gb_model.pkl)")

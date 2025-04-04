
import pandas as pd
from sklearn.model_selection import train_test_split
from lazypredict.Supervised import LazyRegressor

# 데이터 불러오기
data = pd.read_csv("driving_data.csv")

# 특징과 타깃 정의
X = data[["front", "left", "right", "speed", "dx_to_goal", "dy_to_goal"]]
y = data["steering"]

# 학습/테스트 분리
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LazyRegressor 실행
reg = LazyRegressor(verbose=1, ignore_warnings=True, custom_metric=None)
models, predictions = reg.fit(X_train, X_test, y_train, y_test)

# 결과 저장
models.to_csv("lazyregressor_results.csv")
print("모든 회귀 모델 결과를 lazyregressor_results.csv에 저장하였습니다.")

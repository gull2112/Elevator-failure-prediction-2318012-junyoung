# 필요한 라이브러리 임포트
import os  # 파일 및 디렉토리 관리
import pandas as pd  # 데이터 처리 및 분석
import matplotlib.pyplot as plt  # 시각화
import seaborn as sns  # 시각화 (스타일 및 추가 기능)
from sklearn.model_selection import train_test_split  # 데이터 분할 (학습용/테스트용)
from sklearn.ensemble import RandomForestClassifier  # 랜덤 포레스트 분류 모델
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve  # 모델 성능 평가 지표
from sklearn.preprocessing import StandardScaler  # 데이터 표준화
from imblearn.over_sampling import RandomOverSampler  # 데이터 불균형 해소를 위한 오버샘플링

# 데이터 경로 설정 및 Excel 파일 불러오기
data_path = './data/2.elevator_failure_prediction.xlsx'
data = pd.read_excel(data_path, sheet_name='data')  # Excel 파일의 'data' 시트 불러오기

# 결과 저장 폴더 생성
results_folder = './results'
os.makedirs(results_folder, exist_ok=True)  # 폴더가 없으면 생성

# 데이터 요약 및 결측치 확인
summary = data.describe()  # 수치형 데이터의 기본 통계 요약
data_head = data.head()  # 데이터의 상위 5개 행 미리보기
missing_values = data.isnull().sum()  # 각 열별 결측치 개수

# 요약 및 결측치 정보를 CSV 파일로 저장
summary.to_csv(f"{results_folder}/data_summary.csv")
data_head.to_csv(f"{results_folder}/data_head.csv", index=False)
missing_values.to_csv(f"{results_folder}/missing_values.csv", header=["Missing Count"])

# 결측치 처리: Temperature와 Sensor2의 결측치를 중앙값으로 채움
data['Temperature'] = data['Temperature'].fillna(data['Temperature'].median())
data['Sensor2'] = data['Sensor2'].fillna(data['Sensor2'].median())

# 불필요한 열(Time) 삭제
data = data.drop(columns=['Time'])

# 독립변수(X)와 종속변수(y) 분리
X = data.drop(columns=['Status'])  # 'Status' 열 제외한 나머지는 X
y = data['Status']  # 목표 변수 'Status'

# 클래스 불균형 확인 및 저장
class_distribution_before = y.value_counts()  # 클래스별 개수 확인
class_distribution_before.to_csv(f"{results_folder}/class_distribution_before.csv", header=["Count"])

# 클래스 불균형 해소: 오버샘플링 수행
ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X, y)  # 랜덤 오버샘플링 적용

# 오버샘플링 후 클래스 분포 저장
class_distribution_after = pd.Series(y_resampled).value_counts()
class_distribution_after.to_csv(f"{results_folder}/class_distribution_after.csv", header=["Count"])

# 데이터를 학습용/테스트용으로 분할
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# 데이터 표준화 (StandardScaler 적용)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # 학습 데이터 기준으로 표준화
X_test = scaler.transform(X_test)  # 테스트 데이터 변환

# 랜덤 포레스트 모델 학습
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)  # 모델 훈련

# 테스트 데이터 예측 수행
y_pred = model.predict(X_test)  # 클래스 예측
y_pred_proba = model.predict_proba(X_test)  # 클래스별 확률 예측

# 모델 성능 평가
accuracy = accuracy_score(y_test, y_pred)  # 정확도 평가
roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')  # ROC AUC 점수 (다중 클래스)

# 평가 지표를 DataFrame으로 저장
report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)  # 분류 보고서 생성
metrics = pd.DataFrame({"Metric": ["Accuracy", "ROC AUC Score"], "Value": [accuracy, roc_auc]})
metrics.to_csv(f"{results_folder}/evaluation_metrics.csv", index=False)  # 평가 지표 저장
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(f"{results_folder}/classification_report.csv")  # 분류 보고서 저장

# ROC 곡선 시각화 및 저장
plt.figure(figsize=(8, 6))
for i in range(y_pred_proba.shape[1]):  # 각 클래스에 대해 ROC 곡선 생성
    fpr, tpr, _ = roc_curve((y_test == i).astype(int), y_pred_proba[:, i])
    plt.plot(fpr, tpr, label=f"Class {i} ROC Curve")
plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")  # 무작위 예측 기준선
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(f"{results_folder}/roc_curve.png")  # ROC 곡선 저장
plt.close()

# 특성 중요도 확인 및 저장
importances = model.feature_importances_  # 특성 중요도 계산
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importances}).sort_values(by='Importance', ascending=False)
feature_importance.to_csv(f"{results_folder}/feature_importance.csv", index=False)  # 특성 중요도 저장

# 특성 중요도 시각화 및 저장
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title("Feature Importances")
plt.savefig(f"{results_folder}/feature_importance.png")  # 특성 중요도 그래프 저장
plt.close()

# 최종 완료 메시지 출력
print("Results saved to 'results' folder.")

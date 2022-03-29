# MNIST #
# 데이터셋에서 이미지 하나 확인
import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)
plt.imshow(some_digit_image, cmap=mpl.cm.binary)
plt.axis("off")

plt.show()

# 데이터를 자세히 조사하기 전에 항상 테스트 세트를 만들고 따로 떼어놓아야
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# 이진 분류기 훈련 #
# 이진 분류기(binary classifier)를 위해 타깃 벡터 생성
y_train_5 = (y_train == 5) # 5는 True고, 다른 숫자는 모두 False
y_test_5 = (y_test == 5)

# 확률적 경사 하강법(Stochastic Gradient Descent(SGD)) 분류기 모델을 만들고 전체 훈련 세트를 사용해 훈련
from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
sgd_clf.fit(X_train, y_train_5)

# 성능 측정 #
# 교차 검증을 사용한 정확도 측정 #
# 정확도(accuracy): 정확한 예측의 비율
# 모든 이미지를 '5 아님' 클래스로 분류하는 더미 분류기를 만들어 비교
from sklearn.base import BaseEstimator
class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass
    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)
      
# 불균형한 데이터셋을 다룰 때(어떤 클래스가 다른 것보다 월등히 많은 경우) 정확도를 분류기의 성능 측정 지표로 선호 x

# 오차 행렬 #
# 오차 행렬(confusion matrix)을 만들려면 실제 타깃과 비교할 수 있도록 먼저 예측값을 만들어야
from sklearn.model_selection import cross_val_predict

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

# 오차 행렬의 행은 실제 클래스를 나타내고 열은 예측한 클래스를 나타낸다
# 이 행렬의 첫 번째 행은 음성 클래스(negative class)에 대한 것 - 진짜 음성(true negative) vs 거짓 양성(false positive)
# 두 번째 행은 양성 클래스(positive class)에 대한 것 - 거짓 음성(false negative) vs 진짜 양성(true positive)

# 정밀도-재현율 트레이드오프 #
# 적절한 임곗값을 어떻게 정할 수 있을까?
# cross_val_predict() 함수를 사용해 예측 결과가 아니라 결정 함수를 반환받도록 지정하여 훈련 세트에 있는 모든 샘플의 점수를 구해야
y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3,
                             method="decision_function")

# 이 점수로 precision_recall_curve() 함수를 사용하여 가능한 모든 임곗값에 대해 정밀도와 재현율 계산
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)

# 맷플롯립을 이용해 임곗값의 함수로 정밀도와 재현율 표현
def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    # 임곗값을 표시하고 범례, 축 이름, 그리드 추가
    plt.legend(loc="center right", fontsize=16)
    plt.xlabel("Threshold", fontsize=16)
    plt.grid(True)
    plt.axis([-50000, 50000, 0, 1])

plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

# 조금 더 정확하게 최소한 90% 정밀도가 되는 가장 낮은 임곗값을 찾을 수 있다
threshold_90_precision = thresholds[np.argmax(precisions >= 0.90)]

# 예측을 만들려면 분류기의 predict() 메서드를 호출하는 대신
y_train_pred_90 = (y_scores >= threshold_90_precision)

# ROC 곡선 #
# ROC 곡선을 그리려면 먼저 roc_curve() 함수를 사용해 여러 임곗값에서 TPR과 FPR을 계산
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)

# 맷플롯립을 사용해 TPR에 대한 FPR 곡선을 그린다
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # 대각 점선
    # 축 이름, 그리드 추가
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (Fall-Out)', fontsize=16)
    plt.ylabel('True Positive Rate (Recall)', fontsize=16)
    plt.grid(True)

plot_roc_curve(fpr, tpr)
plt.show()

# 곡선 아래의 면적(area under the curve(AUC))을 측정하면 분류기들을 비교

# RandomForestClassifier를 훈련시켜 SGDClassifier의 ROC 곡선과 ROC AUC 점수를 비교
from sklearn.ensemble import RandomForestClassifier
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3,
                                    method="predict_proba")

# 양성 클래스 확률을 점수로 사용
y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

# 비교를 위해 첫 번째 ROC 곡선도 함께 표현
plt.plot(fpr, tpr, "b:", linewidth=2, label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right", fontsize=16)
plt.show()

# 에러 분석 #
# 오차 행렬을 맷플롯립의 matshow() 함수를 사용해 이미지로 표현하면 보기에 편리할 때가 많다
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()

# 그래프의 에러 부분에 초점
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

# 다른 항목은 그대로 유지하고 주대각선만 0으로 채워서 그래프 표현
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()

# 개개의 에러를 분석해보면 분류기가 무슨 일을 하고, 왜 잘못되었는지에 대해 통찰을 얻을 수 있지만, 더 어렵고 시간이 오래 걸린다
# 예를 들어 3와 5의 샘플
cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8,8))
plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
plt.show()

# 다중 레이블 분류 #
# 다중 레이블 분류의 간단한 예
from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)

# 레이블에 클래스의 지지도(support)(타깃 레이블에 속한 샘플 수)를 가중치로 주려면 average="weighted"로 설정

# 다중 출력 분류 #
# 먼저 MNIST의 이미지에서 추출한 훈련 세트와 테스트 세트에 넘파이의 randint() 함수를 사용하여 픽셀 강도에 잡음을 추가
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

# 분류기를 훈련시켜 이미지를 깨끗하게 
knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)

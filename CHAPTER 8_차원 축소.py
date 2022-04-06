# PCA #
# 주성분 #
# 넘파이의 svd() 함수를 사용해 훈련 세트의 모든 주성분을 구한 후 처음 두 개의 PC 정의하는 두 개의 단위 벡터 추출
X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

# d차원으로 투영하기 #
# 첫 두 개의 주성분으로 정의된 평면에 훈련 세트 투영
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)

# 사이킷런 사용하기 #
# PCA 모델을 사용해 데이터의 차원을 2로 줄임
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

# 적절한 차원 수 선택하기 #
# 차원을 축소하지 않고 PCA를 계산한 뒤 훈련 세트의 분산을 95%로 유지하는 데 필요한 최소한의 차원 수 계산
pca = PCA()
pca.fit(X_train)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

# 압축을 위한 PCA #
# MNIST 데이터셋을 154차원으로 압축하고 inverse_transform() 메서드를 사용해 784차원으로 복원
pca = PCA(n_components = 154)
X_reduced = pca.fit_transform(X_train)
X_recovered = pca.inverse_transform(X_reduced)

# 랜덤 PCA #
rnd_pca = PCA(n_components=154, svd_solver="randomized")
X_reduced = rnd_pca.fit_transform(X_train)

# 점진적 PCA #
# MNIST 데이터셋을 100개의 미니배치로 나누고 사이킷런의 IncrementalPCA 파이썬 클래스에 주입하여 MNIST 데이터셋의 차원을 154개로 줄임
from sklearn.decomposition import IncrementalPCA

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
  inc_pca.partial_fit(X_batch)

X_reduced = inc_pca.transform(X_train)

# 넘파이의 memmap 파이썬 클래스를 사용해 하드 디스크의 이진 파일에 저장된 매우 큰 배열을 메모리에 들어 있는 것처럼 다룸
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m,n))

batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)

# 커널 PCA #
# 사이킷런의 KernelPCA를 사용해 RBF 커널로 kPCA 적용
from sklearn.decomposition import KernelPCA

rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 0.04)
X_reduced = rbf_pca.fit_transform(X)

# 커널 선택과 하이퍼파라미터 튜닝 #
# 두 단계의 파이프라인
# - kPCA를 사용해 차원을 2차원으로 축소하고 분류를 위해 로지스틱 회귀 적용
# - 파이프라인 마지막 단계에서 가장 높은 분류 정확도 얻기 위해 GridSearchCV를 사용해 kPCA의 가장 좋은 커널과 gamma 파라미터 찾음
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

clf = Pipeline([
                ("kpca", KernelPCA(n_components=2)),
                ("log_reg", LogisticRegression())
])

param_grid = [{
    "kpca__gamma": np.linspace(0.03, 0.05, 10),
    "kpca__kernel": ["rbf", "sigmoid"]
}]

grid_search = GridSearchCV(clf, param_grid, cv=3)
grid_search.fit(X, y)

# 사이킷런에서는 fit_inverse_transform=True로 지정하면 재구성을 자동으로 수행
rbf_pca = KernelPCA(n_components = 2, kernel = "rbf", gamma = 0.0433,
                    fit_inverse_transform = True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced)

# LLE #
# 사이킷런의 LocalyLinearEmbedding을 사용해 스위스 롤 펼침
from sklearn.manifold import LocallyLinearEmbedding

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X_reduced = lle.fit_transform(X)

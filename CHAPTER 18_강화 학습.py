# OpenAI 짐 #
def basic_policy(obs):
    angle = obs[2]
    return 0 if angle < 0 else 1

totals = []
for episode in range(500):
    episode_rewards = 0
    obs = env.reset()
    for step in range(200):
        action = basic_policy(obs)
        obs, reward, done, info = env.step(action)
        episode_rewards += reward
        if done:
            break
    totals.append(episode_rewards)
    
# 신경망 정책 #
'''
다음은 tf.keras를 사용하여 신경망 정책을 구현하는 코드이다.
'''

import tensorflow as tf
from tensorflow import keras

n_inputs = 4 # == env.observation_space.shape[0]

model = keras.models.Sequential([
    keras.layers.Dense(5, activation="elu", input_shape=[n_inputs]),
    keras.layers.Dense(1, activation="sigmoid"),
])

'''
필요한 라이브러리를 임포트한 후에 간단한 Sequential 모델을 사용해 정책 네트워크를 정의한다. 입력의 개수는 관측 공간의 크기이다. 간단한 문제이므로 은닉 유닛 5개를 사용한다. 마지막으로 하나의 확률이 필요하므로 시그모이드 활성화 함수를 사용한 하나의 출력 뉴런을 둔다. 만약 가능한 행동이 두 개보다 많으면 행동마다 하나의 출력 뉴런을 두고 소프트맥스 활성화 함수를 사용해야 한다.
'''

# 정책 그레이디언트 #
def play_one_step(env, obs, model, loss_fn):
    with tf.GradientTape() as tape:
        left_proba = model(obs[np.newaxis])
        action = (tf.random.uniform([1, 1]) > left_proba)
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0, 0].numpy()))
    return obs, reward, done, grads

'''
- GradientTape 블록 안에서 하나의 관측과 함께 먼저 모델을 호출한다. 이 모델은 왼쪽으로 이동할 확률을 출력한다.
- 그다음 0에서 1 사이에서 랜덤한 실수를 샘플링한다. 이 값이 left_proba보다 큰지를 확인한다. action은 left_proba 확률로 False가 되고 1-left_proba 확률로 True가 될 것이다. 이 불리언 값을 숫자로 변환하면 액션은 출력된 확률에 맞게 0 또는 1이 된다.
- 이제 왼쪽으로 이동할 타깃 확률을 정의한다. 이 값은 1 빼기 행동이다. 행동이 0이면 왼쪽으로 이동할 타깃 확률은 1이 될 것이다. 행동이 1이면 타깃 확률이 0이 될 것이다.
- 그런 다음 주어진 손실 함수를 사용해 손실을 계산하고 테이프를 사용해 모델의 훈련 가능한 변수에 대한 손실의 그레이디언트를 계산한다. 이 그레이디언트도 나중에 적용하기 전에 이 행동이 좋은지 나쁜지에 따라 조정될 것이다.
- 마지막으로 선택한 행동을 플레이하고 새로운 관측, 보상, 에피소드 종료 여부, 계산한 그레이디언트를 반환한다.

이제 play_one_step() 함수를 사용해 여러 에피소드를 플레이하고 전체 보상과 각 에피소드와 스텝의 그레이디언트를 반환하는 또 다른 함수를 만들어보겠다.
'''

def play_multiple_episodes(env, n_episodes, n_max_steps, model, loss_fn):
    all_rewards = []
    all_grads = []
    for episode in range(n_episodes):
        current_rewards = []
        current_grads = []
        obs = env.reset()
        for step in range(n_max_steps):
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            current_rewards.append(reward)
            current_grads.append(grads)
            if done:
                break
        all_rewards.append(current_rewards)
        all_grads.append(current_grads)
    return all_rewards, all_grads
  
'''
이 알고리즘은 play_multiple_episodes() 함수를 사용하여 여러 번 게임을 플레이한다. 그다음 처음부터 모든 보상을 살펴서 각 보상을 할인하고 정규화한다. 이렇게 하기 위해 함수 몇 개가 더 필요하다. 첫 번째 함수는 각 스텝에서 할인된 미래 보상의 합을 계산한다. 두 번째 함수는 여러 에피도스에 걸쳐 계산된 할인된 이 모든 보상에서 평균을 빼고 표준편차를 나누어 정규화한다.
'''

def discount_rewards(rewards, discount_rate):
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

def discount_and_normalize_rewards(all_rewards, discount_rate):
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]
  
n_iterations = 150
n_episodes_per_update = 10
n_max_steps = 200
discount_rate = 0.95

optimizer = keras.optimizers.Adam(learning_rate=0.01)
loss_fn = keras.losses.binary_crossentropy

for iteration in range(n_iterations):
    all_rewards, all_grads = play_multiple_episodes(
        env, n_episodes_per_update, n_max_steps, model, loss_fn)
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                       discount_rate)
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

'''
- 각 훈련 반복에서 play_multiple_episodes() 함수를 호출한다. 이 함수는 게임을 10번 플레이하고 각 에피소드와 스텝에 대한 모든 보상과 그레이디언트를 반환한다.
- 그다음 discount_and_normalize_rewards() 함수를 호출하여 각 행동의 정규화된 이익을 계산한다. 이 값은 각 행동이 실제로 얼마나 좋은지 나쁜지를 알려준다.
- 그다음 각 훈련가능한 변수를 순회하면서 모든 에피소드와 모든 스텝에 대한 각 변수의 그레이디언트를 final_reward로 가중치를 두어 평균한다.
- 마지막으로 이 평균 그레이디언트를 옵티마이저에 적용한다. 모델의 훈련 가능한 변수가 변경되고 아마 정책은 조금 더 나아질 것이다.
'''

# 마르코프 결정 과정 #
transition_probabilities = [ # shape=[s, a, s']
        [[0.7, 0.3, 0.0], [1.0, 0.0, 0.0], [0.8, 0.2, 0.0]],
        [[0.0, 1.0, 0.0], None, [0.0, 0.0, 1.0]],
        [None, [0.8, 0.1, 0.1], None]]
rewards = [ # shape=[s, a, s']
        [[+10, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 0, -50]],
        [[0, 0, 0], [+40, 0, 0], [0, 0, 0]]]
possible_actions = [[0, 1, 2], [0, 2], [1]]

'''
예를 들어, 행동 a1을 플레이한 후 s2에서 s0으로 전이할 확률을 알기 위해서는 transition_probabilities[2][1][0]을 참조한다. 비슷하게 이에 해당하는 보상을 얻으려면 rewards[2][1][0]을 참조한다. s2에서 가능한 행동의 리스트를 얻으려면 possible_actions[2]를 참조한다. 그다음 모든 Q-가치를 0으로 초기화해야 한다.
'''

Q_values = np.full((3, 3), -np.inf) # -np.inf for impossible actions
for state, actions in enumerate(possible_actions):
    Q_values[state, actions] = 0.0  # for all possible actions
    
gamma = 0.90  # the discount factor

history1 = [] # Not shown in the book (for the figure below)
for iteration in range(50):
    Q_prev = Q_values.copy()
    history1.append(Q_prev) # Not shown
    for s in range(3):
        for a in possible_actions[s]:
            Q_values[s, a] = np.sum([
                    transition_probabilities[s][a][sp]
                    * (rewards[s][a][sp] + gamma * np.max(Q_prev[sp]))
                for sp in range(3)])

# Q-러닝 #
def step(state, action):
    probas = transition_probabilities[state][action]
    next_state = np.random.choice([0, 1, 2], p=probas)
    reward = rewards[state][action][next_state]
    return next_state, reward
  
def exploration_policy(state):
    return np.random.choice(possible_actions[state])
  
alpha0 = 0.05 # initial learning rate
decay = 0.005 # learning rate decay
gamma = 0.90 # discount factor
state = 0 # initial state
history2 = [] # Not shown in the book

for iteration in range(10000):
    history2.append(Q_values.copy()) # Not shown
    action = exploration_policy(state)
    next_state, reward = step(state, action)
    next_value = np.max(Q_values[next_state]) # greedy policy at the next step
    alpha = alpha0 / (1 + iteration * decay)
    Q_values[state, action] *= 1 - alpha
    Q_values[state, action] += alpha * (reward + gamma * next_value)
    state = next_state
    
# 심층 Q-러닝 구현하기 #
env = gym.make("CartPole-v0")
input_shape = [4] # == env.observation_space.shape
n_outputs = 2 # == env.action_space.n

model = keras.models.Sequential([
    keras.layers.Dense(32, activation="elu", input_shape=input_shape),
    keras.layers.Dense(32, activation="elu"),
    keras.layers.Dense(n_outputs)
])

def epsilon_greedy_policy(state, epsilon=0):
    if np.random.rand() < epsilon:
        return np.random.randint(n_outputs)
    else:
        Q_values = model.predict(state[np.newaxis])
        return np.argmax(Q_values[0])
      
from collections import deque

replay_memory = deque(maxlen=2000)

'''
각 경험은 원소 5개로 구성된다. 상태, 에이전트가 선택한 행동, 결과 보상, 도달한 다음 상태, 마지막으로 이 에피소드가 이때 종료되었는지 가리키는 불리언 값. 재생 버퍼에서 경험을 랜덤하게 샘플링하기 위해 작은 함수를 만들겠다. 이 함수는 경험 원소 5개에 상응하는 넘파이 배열 5개를 반환한다.
'''

def sample_experiences(batch_size):
    indices = np.random.randint(len(replay_memory), size=batch_size)
    batch = [replay_memory[index] for index in indices]
    states, actions, rewards, next_states, dones = [
        np.array([experience[field_index] for experience in batch])
        for field_index in range(5)]
    return states, actions, rewards, next_states, dones
  
def play_one_step(env, state, epsilon):
    action = epsilon_greedy_policy(state, epsilon)
    next_state, reward, done, info = env.step(action)
    replay_memory.append((state, action, reward, next_state, done))
    return next_state, reward, done, info
  
batch_size = 32
discount_rate = 0.95
optimizer = keras.optimizers.Adam(learning_rate=1e-2)
loss_fn = keras.losses.mean_squared_error

def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    max_next_Q_values = np.max(next_Q_values, axis=1)
    target_Q_values = (rewards +
                       (1 - dones) * discount_rate * max_next_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
'''
- 먼저 하이퍼파라미터 몇 개를 정의하고 옵티마이저와 손실 함수를 만든다.
- 그다음 training_step() 함수를 만든다. 이 함수는 경험 배치를 샘플링한 다음 DQN을 사용하여 각 경험의 다음 상태에서 가능한 모든 행동에 대한 Q-가치를 예측한다. 에이전트가 최적으로 플레이한다고 가정하므로 다음 상태에 대한 최대 Q-가치만 저장한다. 그다음 각 경험의 상태-행동 쌍에 대한 타깃 Q-가치를 계산한다.
- DQN이 경험한 각 상태-행동 쌍의 Q-가치를 계산하길 원한다. 하지만 이 DQN은 에이전트가 실제로 선택한 행동뿐만 아니라 다른 가능한 행동에 대한 Q-가치도 출력할 것이다. 따라서 필요하지 않은 모든 Q-가치를 마스크 처리해야 한다. tf.one_hot() 함수는 행동 인덱스의 배열을 마스크로 쉽게 변환해준다. 예를 들어 처음 3개의 경험이 행동 1, 1, 0을 각각 담고 있다면 마스크는 [[0,1],[0,1],[1,0],...]와 같을 것이다. 이 마스크를 DQN의 출력과 곱하여 원하지 않은 Q-가치를 0으로 만들 수 있다. 그다음 0을 없애기 위해 열 방향으로 덧셈하여 경험된 상태-행동 쌍의 Q-가치만 남긴다. 결국 배치에 있는 각 경험에 대해 예측된 Q-가치 하나를 담은 텐서인 Q_values를 얻는다.
- 그다음 손실을 계산한다. 경험된 상태-행동 쌍에 대한 타깃과 예측된 Q-가치 사이의 평균 제곱 오차이다.
- 마지막으로 모델의 훈련 가능 변수에 관한 손실을 최소화하기 위해 경사 하강법을 수행한다.
'''

for episode in range(600):
    obs = env.reset()    
    for step in range(200):
        epsilon = max(1 - episode / 500, 0.01)
        obs, reward, done, info = play_one_step(env, obs, epsilon)
        if done:
            break
    if episode > 50:
        training_step(batch_size)
        
'''
최대 스텝 200번으로 이루어진 에피소드 600개를 실행한다. 각 스텝에서 먼저 epsilon-그리디 정책에 대한 epsilon 값을 계산한다. 이 값은 500 에피소드 직전까지 1에서 0.01로 선형적으로 줄어든다. 그다음 play_one_step() 함수를 호출한다. 이 함수는 epsilon-그리디 정책을 사용해 행동을 선택하여 실행하고 그 경험을 재생 버퍼에 기록한다. 에피소드가 종료되면 반복을 끝낸다. 마지막으로 50번째 에피소드 이후에는 training_step() 함수를 호출해 재생 버퍼에서 샘플링한 배치로 모델을 훈련한다. 훈련 없이 에피소드를 50번 플레이하는 이유는 재생 버퍼가 채워질 시간을 주기 위해서이다.
'''

# 심층 Q-러닝의 변종 #
## 고정 Q-가치 타깃 ##
target = keras.models.clone_model(model)
target.set_weights(model.get_weights())

'''
그다음 training_step() 함수에서 다음 상태의 Q-가치를 계산할 때 온라인 모델 대신 타깃 모델을 사용하도록 한 줄을 바꾸어야 한다.
'''

next_Q_values = model.predict(next_states)

if episode % 50 == 0:
  target.set_weights(model.get_weights())
  
'''
타깃 모델이 온라인 모델보다 자주 업데이트되지 않으므로, Q-가치 타깃이 더 안정적이며 피드백 반복을 완화하고 이에 대한 영향이 감소된다. 이 방식이 딥마인드 연구자들이 2013년 논문에서 달성한 주요 성과 중 하나이다. 논문에서 에이전트가 원시 픽셀로부터 아타리 게임을 플레이하는 방법을 학습했다. 연구자들은 안정적으로 훈련하기 위해 0.00025라는 작은 학습률을 사용하고 10,000 스텝마다 타깃 모델을 업데이트했다. 그리고 100만 경험을 저장할 수 있는 매우 큰 재생 버퍼를 사용했다. epsilon을 100만 스텝 동안 1에서 0.1까지 매우 천천히 감소하고 5천만 스텝 동안 알고리즘을 실행했다.
'''

## 더블 DQN ##
def training_step(batch_size):
    experiences = sample_experiences(batch_size)
    states, actions, rewards, next_states, dones = experiences
    next_Q_values = model.predict(next_states)
    best_next_actions = np.argmax(next_Q_values, axis=1)
    next_mask = tf.one_hot(best_next_actions, n_outputs).numpy()
    next_best_Q_values = (target.predict(next_states) * next_mask).sum(axis=1)
    target_Q_values = (rewards + 
                       (1 - dones) * discount_rate * next_best_Q_values)
    target_Q_values = target_Q_values.reshape(-1, 1)
    mask = tf.one_hot(actions, n_outputs)
    with tf.GradientTape() as tape:
        all_Q_values = model(states)
        Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
        loss = tf.reduce_mean(loss_fn(target_Q_values, Q_values))
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    
## 듀얼링 DQN ##
K = keras.backend
input_states = keras.layers.Input(shape=[4])
hidden1 = keras.layers.Dense(32, activation="elu")(input_states)
hidden2 = keras.layers.Dense(32, activation="elu")(hidden1)
state_values = keras.layers.Dense(1)(hidden2)
raw_advantages = keras.layers.Dense(n_outputs)(hidden2)
advantages = raw_advantages - K.max(raw_advantages, axis=1, keepdims=True)
Q_values = state_values + advantages
model = keras.models.Model(inputs=[input_states], outputs=[Q_values])

# TF-Agents 라이브러리 #
## 환경 래퍼와 아타리 전처리 ##
'''
래핑한 환경을 만들려면 래핑할 환경을 생성자에 전달하여 래퍼를 만들어야 한다. 예를 들어 다음 코드는 모든 행동을 4번씩 반복하기 위해 앞서 만든 환경을 ActionRepeat 래퍼로 감싼다.
'''

from tf_agents.environments.wrappers import ActionRepeat

repeating_env = ActionRepeat(env, times=4)

'''
OpenAI 짐은 gym.wrappers 패키지에 자체적으로 래퍼를 가지고 있다. 이 래퍼는 TF-Agents 환경을 위한 것이 아니라 짐 환경을 래핑한다. 따라서 이 래퍼를 사용하려면 먼저 짐 환경을 짐 래퍼로 감싸고 그다음 만들어진 환경을 TF-Agents 래퍼로 감싸야 한다. suite_gym.wrap_env() 함수에 짐 환경과 짐 래퍼 리스트, TF-Agents 래퍼 리스트를 제공하면 이런 작업을 처리한다. 또는 suite_gym.load() 함수가 짐 환경을 만들고 래퍼를 받아 처리도 해준다. 래퍼는 매개변수 없이 만들어지기 때문에 매개변수를 지정하고 싶으면 lambda로 전달해야 한다. 예를 들어 다음 코드는 각 에피소드에서 최대 10,000번 스텝을 실행하고 각 행동이 네 번 반복되는 <브레이크아웃> 환경을 만든다.
'''

from gym.wrappers import TimeLimit

limited_repeating_env = suite_gym.load(
    "Breakout-v4",
    gym_env_wrappers=[partial(TimeLimit, max_episode_steps=10000)],
    env_wrappers=[partial(ActionRepeat, times=4)],
)

'''
아타리 환경을 사용하는 경우 대부분의 논문에서 표준적인 전처리 단계가 있다. TF-Agents는 이를 구현하여 간편한 AtariPreprocessing 래퍼를 제공한다.
기본 아타리 환경은 이미 랜덤 프레임 스킵과 맥스 풀링이 적용되어 있기 때문에 스킵되지 않는 원본 환경 "BreakoutNoFrameskip-v4"를 로드했다. 또한 <브레이크아웃>에서 하나의 프레임을 가지고는 공의 방향과 속도를 알지 못한다. 이 때문에 에이전트가 게임을 적절히 플레이하기 매우 어렵다. 이 문제를 다루는 한 방법은 채널 차원으로 프레임을 여러 개 쌓아 관측으로 출력하는 환경 래퍼를 사용하는 것이다. FrameStack4 래퍼에 이 방법이 구현되어 있으며 프레임 4개를 쌓아 반환한다. 그럼 래핑된 이타리 환경을 만들어보겠다.
'''

from tf_agents.environments import suite_atari
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
from tf_agents.environments.atari_wrappers import FrameStack4

max_episode_steps = 27000 # <=> 108k ALE frames since 1 step = 4 frames
environment_name = "BreakoutNoFrameskip-v4"

env = suite_atari.load(
    environment_name,
    max_episode_steps=max_episode_steps,
    gym_env_wrappers=[AtariPreprocessingWithAutoFire, FrameStack4])

'''
끝으로 TFPyEnvironment 안에 이 환경을 감쌀 수 있다.
'''

from tf_agents.environments.tf_py_environment import TFPyEnvironment

tf_env = TFPyEnvironment(env)

'''
이렇게 하면 텐서플로 그래프 안에 이 환경을 사용할 수 있다. TFPyEnvironment 클래스 덕분에 TF-Agents가 순수한 파이썬 환경과 텐서플로 기반 환경을 모두 지원한다. 더 일반적으로 TF-Agents는 순수한 파이썬 컴포넌트와 텐서플로 컴포넌트를 지원하고 제공한다.
'''

## 심층 Q-네트워크 만들기 ##
'''
TF-Agents 라이브러리는 tf_agents.networks 패키지와 서브 패키지에 많은 네트워크를 제공한다. 여기에서는 tf_agents.networks.q_network.QNetwork 클래스를 사용한다.
'''

from tf_agents.networks.q_network import QNetwork

preprocessing_layer = keras.layers.Lambda(
                          lambda obs: tf.cast(obs, np.float32) / 255.)
conv_layer_params=[(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 1)]
fc_layer_params=[512]

q_net = QNetwork(
    tf_env.observation_spec(),
    tf_env.action_spec(),
    preprocessing_layers=preprocessing_layer,
    conv_layer_params=conv_layer_params,
    fc_layer_params=fc_layer_params)

'''
이 QNetwork는 옵저버를 입력으로 받고 행동마다 하나의 Q-가치를 출력한다. 따라서 관측과 행동의 스펙을 전달해야 한다. 먼저 관측을 32비트 실수로 변환하고 정규화하는 간단한 Lambda 층인 전처리 층이 등장한다. 이 관측은 32비트 실수보다 4배나 작은 공간을 차지하는 부호가 없는 바이트이다. 이 때문에 재생 버퍼의 RAM을 절약하려고 미리 관측을 32비트 실수로 변환하지 않았다. 그다음 네트워크는 합성곱 층 3개를 적용한다. 첫 번째는 8*8 필터 32개와 스트라이드 4를 사용하고, 두 번째는 8*8 필터 32개와 스트라이드 2, 세 번째는 8*8 필터 32개와 스트라이드 1을 사용한다. 마지막으로 유닛 512개를 가진 밀집 층을 적용한다. 그 뒤에는 출력할 Q-가치마다 하나씩 유닛 4개를 가진 밀집 층이 놓인다. 출력층을 제외한 모든 합성곱 층과 밀집 층은 기본적으로 ReLU 활성화 함수를 사용한다. 출력층은 활성화 함수를 사용하지 않는다.
'''

## DQN 에이전트 만들기 ##
'''
TF-Agents는 tf_agents.agents 패키지와 서브 패키지에 많은 종류의 에이전트를 구현해 놓았다. 여기에서는 tf_agents.agents.dqn.dqn_agent.DqnAgent 클래스를 사용하겠다.
'''

from tf_agents.agents.dqn.dqn_agent import DqnAgent

train_step = tf.Variable(0)
update_period = 4 # run a training step every 4 collect steps
optimizer = keras.optimizers.RMSprop(learning_rate=2.5e-4, rho=0.95, momentum=0.0,
                                     epsilon=0.00001, centered=True)
epsilon_fn = keras.optimizers.schedules.PolynomialDecay(
    initial_learning_rate=1.0, # initial ε
    decay_steps=250000 // update_period, # <=> 1,000,000 ALE frames
    end_learning_rate=0.01) # final ε
agent = DqnAgent(tf_env.time_step_spec(),
                 tf_env.action_spec(),
                 q_network=q_net,
                 optimizer=optimizer,
                 target_update_period=2000, # <=> 32,000 ALE frames
                 td_errors_loss_fn=keras.losses.Huber(reduction="none"),
                 gamma=0.99, # discount factor
                 train_step_counter=train_step,
                 epsilon_greedy=lambda: epsilon_fn(train_step))
agent.initialize()

'''
- 먼저 훈련 스텝 횟수를 헤아릴 변수를 만든다.
- 그다음 2015년 DQN 논문과 같은 하이퍼파라미터를 사용해 옵티마이저를 만든다.
- 그리고 현재 훈련 스텝이 주어졌을 때 epsilon-그리디 수집 정책을 위한 epsilon값을 계산하는 PolynomialDecay 객체를 만든다. 100만 ALE 프레임 동안 1.0에서 0.01로 줄어들 것이다. 4 프레임마다 사용하므로 250,000 스텝에 해당한다. 또한 에이전트가 4 스텝마다 훈련되므로 epsilon은 62,500 훈련 스텝에 걸쳐 감쇠된다.
- 그다음 DQNAgent를 만들 때 타임 스텝과 행동 스펙, 훈련할 QNetwork, 옵티마이저, 타깃 모델을 업데이트할 훈련 스텝 간격, 손실 함수, 할인 계수, train_step 변수를 전달한다. 그리고 epsilon값을 반환하는 함수를 전달한다.
- 손실 함수는 평균이 아니라 샘플마다 하나의 오차를 반환해야 하기 때문에 reduction="none"으로 지정한다.
- 마지막으로 이 에이전트를 초기화한다.
'''

## 재생 버퍼와 옵저버 만들기 ##
'''
TF-Agents는 tf_agents.replay_buffers 패키지로 다양한 재생 버퍼 구현을 제공한다. 순수하게 파이썬으로 작성된 것과 텐서플로 기반으로 작성된 것이 있다. 여기에서는 tf_agents.replay_buffers.tf_uniform_replay_buffer 패키지에 있는 TFUniformReplayBuffer 클래스를 사용하겠다. 이 클래스는 균등 샘플링을 수행하는 고성능 재생 버퍼 구현이다.
'''

from tf_agents.replay_buffers import tf_uniform_replay_buffer

replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=tf_env.batch_size,
    max_length=100000) # reduce if OOM error

'''
이제 재생 버퍼에 경로를 저장할 옵저버를 만들 수 있다. 옵저버는 경로 매개변수를 받는 단순한 함수이기 때문에 옵저버로 add_batch() 메서드를 직접 사용할 수 있다.
'''

replay_buffer_observer = replay_buffer.add_batch

'''
사용자 정의 옵저버를 만들고 싶다면 trajectory 매개변수를 가진 어떤 함수도 가능하다. 상태를 가져야 한다면 __call__(self, trajectory) 메서드를 포함한 클래스를 만들 수 있다. 예를 들어 다음은 호출될 때마다 카운터를 증가시키는 간단한 옵저버이다. 100번 증가될 때마다 주어진 총계에 대한 진행 과정을 표시한다.
'''

class ShowProgress:
    def __init__(self, total):
        self.counter = 0
        self.total = total
    def __call__(self, trajectory):
        if not trajectory.is_boundary():
            self.counter += 1
        if self.counter % 100 == 0:
            print("\r{}/{}".format(self.counter, self.total), end="")
            
## 훈련 측정 지표 만들기 ##
'''
TF-Agent는 tf_agents.metrics 패키지에 여러 가지 강화 학습 측정 지표를 구현해놓았다. 일부는 순수한 파이썬이고 일부는 텐서플로 기반이다. 그중에 몇 가지를 사용해 에피소드 횟수, 가장 중요한 에피소드당 평균 대가, 평균 에피소드 길이를 구해보겠다.
'''
from tf_agents.metrics import tf_metrics

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

'''
언제든지 각 지표의 값을 확인하고 싶으면 result() 메서드를 호출한다. 또는 log_metrics(train_metrics)를 호출하여 모든 지표를 로그에 기록할 수 있다.
'''

## 수집 드라이버 만들기 ##
'''
주요한 드라이버 클래스가 두 개 있다. DynamicStepDriver와 DynamicEpisodeDriver이다. 전자는 주어진 스텝 횟수에 대한 경험을 수집한다. 후자는 주어진 에피소드 횟수에 대한 경험을 수집한다. 여기에서는 각 훈련 반복에서 스텝 4개에 대한 경험을 수집하려 하므로 DynamicStepDriver를 만든다.
'''

from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver

collect_driver = DynamicStepDriver(
    tf_env,
    agent.collect_policy,
    observers=[replay_buffer_observer] + train_metrics,
    num_steps=update_period) # collect 4 steps for each training iteration

'''
드라이버에 환경, 에이전트의 수집 정책, 옵저버 리스트, 마지막으로 실행할 스텝 횟수를 전달한다. 이제 run() 메서드를 호출해 실행할 수 있다. 하지만 완전한 랜덤 정책을 사용해 수집된 경험으로 재생 버퍼를 사전에 채워놓는 것이 좋다. 이를 위해 RandomTFPolicy 클래스를 사용해 20,000 스텝 동안 이 정책을 실행하는 두 번째 드라이버를 만든다. ShowProgress 옵저버로 진행 과정을 볼 수 있다.
'''

from tf_agents.policies.random_tf_policy import RandomTFPolicy

initial_collect_policy = RandomTFPolicy(tf_env.time_step_spec(),
                                        tf_env.action_spec())
init_driver = DynamicStepDriver(
    tf_env,
    initial_collect_policy,
    observers=[replay_buffer.add_batch, ShowProgress(20000)],
    num_steps=20000) # <=> 80,000 ALE frames
final_time_step, final_policy_state = init_driver.run()

## 데이터셋 만들기 ##
'''
주 훈련 반복문에서 get_next() 메서드를 호출하는 대신 tf.data.Dataset을 사용하겠다. 이렇게 하면 Data API의 장점을 사용할 수 있다. 이를 위해 재생 버퍼의 as_dataset() 메서드를 호출한다.
'''

dataset = replay_buffer.as_dataset(
    sample_batch_size=64,
    num_steps=2,
    num_parallel_calls=3).prefetch(3)

## 훈련 반복 만들기 ##
'''
훈련 속도를 높이기 위해 주 함수를 텐서플로 함수로 변경하겠다. 이를 위해 tf.function()를 감싸고 실험적인 옵션을 추가한 tf_agents.utils.common.function() 함수를 사용한다.
'''

from tf_agents.utils.common import function

collect_driver.run = function(collect_driver.run)
agent.train = function(agent.train)

'''
n_iterations 동안 훈련 반복을 실행할 작은 함수를 만든다.
'''

def train_agent(n_iterations):
    time_step = None
    policy_state = agent.collect_policy.get_initial_state(tf_env.batch_size)
    iterator = iter(dataset)
    for iteration in range(n_iterations):
        time_step, policy_state = collect_driver.run(time_step, policy_state)
        trajectories, buffer_info = next(iterator)
        train_loss = agent.train(trajectories)
        print("\r{} loss:{:.5f}".format(
            iteration, train_loss.loss.numpy()), end="")
        if iteration % 1000 == 0:
            log_metrics(train_metrics)
            
'''
이 함수는 먼저 수집 정책에서 초기 상태를 얻는다. 이 정책은 상태가 없기 때문에 빈 튜플을 반환한다. 그다음 데이터셋으로 반복자를 만들고 훈련 반복문을 시작한다. 반복마다 현재의 타임 스텝과 현재 정책 상태를 전달하여 드라이버의 run() 메서드를 실행한다. 이 메서드는 수집 정책을 실행하고 4 스텝 동안 경험을 수집한다. 수집된 경로를 재생 버퍼와 지표로 전달할 것이다. 그다음 데이터셋에서 경로의 배치 하나를 샘플링하여 에이전트의 train() 메서드에 전달한다. 이 메서드는 train_loss 객체를 반환한다. 이 객체는 에이전트의 종류에 따라 다르다. 그다음 반복 횟수와 훈련 손실을 출력하고 1,000번 반복마다 모든 지표를 로그에 기록한다. 이제 얼마간 반복 횟수를 지정하여 train_agent()를 호출해서 에이전트가 조금씩 <브레이크아웃>을 플레이하는 법을 배웠는지 확인할 수 있다.
'''

train_agent(n_iterations=50000)


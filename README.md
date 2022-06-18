# Alpha Holdem -  Playing Texas hold 'em AI with DRL

# I. Introduction

Deep Reinforcement Learning을 이용한 홀덤 에이전트 구현 및 결과 분석

- 포커의 일종인 홀덤은 총 52장의 카드로 진행하며, 개인 카드 2장과 커뮤니티 카드 5장으로 족보를 맞춰서 높은 쪽이 승리하는 게임이다. 처음 개인 카드가 2장 주어지고 베팅을 한다. 그 후 커뮤니티 카드 3장, 1장, 1장이 주어지고 카드가 주어질 때마다 베팅을 하는 시스템이다. 플레이어는 각 차례마다 60초 이상의 시간제한이 주어진다.
- 알파고와는 달리 홀덤은 불완전 정보 게임이며, 바둑은 완전 정보 게임이다.
- 하지만, 홀덤 역시 경우의 수가 매우 큰 Continuous Action Space를 가져 일반적인 강화학습 알고리즘으로는 학습이 어렵다. 그러므로 딥러닝 기반 강화학습(DRL) 알고리즘을 사용해야 한다.
- 최근 여러 연구를 살펴보면, 불완전 정보 환경에서 CFR 알고리즘이 주로 쓰이며, 2017년 카네기 맬런 대학에서 연구한 리브라투스는 강화학습과 CFR(Counterfactual Regret) 알고리즘을 사용해 좋은 성과를 이루어냈다.
- DRL 알고리즘 중 정책 기반 에이전트는 Contiuous Action Space에서도 학습 성능이 좋기 때문에 정책 기반 에이전트 중에서도 좋은 성능을 Actor-Critic 알고리즘을 사용할 것이다.
- Actor-Critic 알고리즘은 가치함수와 정책함수를 두 개의 네트워크를 사용하는 것이 특징으로 $\pi (s,a)$값과 $V(s)$값을 이용해 학습을 진행한다.
- A2C(Advantage Actor-Critic)은 ****Actor-Critic의 Actor의 기대출력으로 Advantage를 사용하면 A2C
가 된다. Advantage는 예상했던 것, $V(s)$보다 얼마나 더 좋은 값인지를 판단하는 값으로, 이는 분산을 줄이는 효과가 있어 A2C 알고리즘을 이용해 학습을 진행해보았다.

# II. Purpose

- Python3를 이용해 홀덤 Environment를 구현한다.
- Pytorch를 이용해 홀덤 환경에서 학습할 수 있는 Acotr-Critic 기반 Agent와 Model을 구현한다.
- Actor-Critic 알고리즘 중 A2C의 두 가지 타입의 Architecture의 성능을 비교 분석한다.
- 추가적으로, 사람과 대결할 수 있는 Application을 제작할 예정이다.

# III. Software Design

## III-1. Environment

**MDP of the holdem Environment**

                                                             $< S, A, P, R, γ >$

$S$ : My hand, Community hand, Pot, My Stack, Opponent Stack, Opponent Bet

⇒ 총 11개의 state

$A$ : Fold, Check, Call, Raise(Continuous Action space = $10^{161}$ in No-Limit Betting)

⇒ 베팅 금액은 연속적이므로 방대한 Actions space를 가진다

$P$ ****: $P_{ss'}^a$ $=  1(∀a, ∀s, ∀s’ )$

$R$ : 라운드 승패 $\pm$ 1, 라운드 마다 칩의 변화  * 0.005

$**γ ∈ [0.05, 0.99]**$ : discount factor (Generally, 0.99)

## III-2. Functional Requirement

1. Visualized Environment
    
    Python을 이용해 게임을 시각화한다. 이전에 tkinter 라이브러리를 이용해 환경에 대한 정보와 GUI를 멀티스레딩으로 구현했지만, 정보 만을 이용해 학습하는 강화학습엔 어려움이 있어 문자열 만을 이용해 환경을 시각화했다.
    
2. A2C(Advantaged Actor-Critic) Model
    
    Actor-Critic 논문과 강화학습 책을 참조하여 홀덤 환경에 알맞는 A2C 모델을 개발한다. Pytorch를 이용하여 Network의 Hidden Layer수, Layer Dimension을 포함한 hyperparameter를 조정하며 최적을 hyperparameter를 구한다.
    
3. 사람과 대결할 수 있는 홀덤 Application
    
    1번 과정에서 제작한 Visualized Environment에서 사용자와 학습된 모델 간의 대결을 할 수 있는 기능을 추가해 불완전 정보 게임에서 구현된 모델이 사람과 상대가 가능한지 성능을 분석한다.
    

## III-3 Architecture

**Actor-Critic Architecture**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/7354f557-6233-4417-8c2f-7757ef6cf86e/Untitled.png)

입력을 해석하는 파라미터를 공유하느냐 그렇지 않느냐에 따라 Actor-Critic 알고리즘의 구조는 크게 Share-A2C와 Not Share-A2C 2가지로 나눌 수 있다.

![Critic-Network.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/c2955652-dad2-48d5-ad8a-44cb90638f29/Critic-Network.jpg)

![Actor-Nework.jpg](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/d64e8cd0-ab1b-4889-bfe4-27a579f33bdf/Actor-Nework.jpg)

**Adversarial A2C (2 agents)**

![Untitled](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/5fbc7e54-b055-4577-86c3-59a221da98af/Untitled.jpeg)

# IV. Implementation

**Critic Network**

```python
class Critic(nn.Module):
    def __init__(self, state_space=None):
        super(Critic, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_space, 64))
        self.layers.append(nn.Linear(64, 64))
        self.layers.append(nn.Linear(64, 64))
        self.layers.append(nn.Linear(64, 1))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = self.layers[-1](x)
        return out
```

**Actor Network**

```python
class Actor(nn.Module):
		def __init__(self, state_space=None, action_space=None):
        super(Actor, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_space, 128))
        self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.Linear(128, 128))
        self.layers.append(nn.Linear(128, action_space))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = F.softmax(self.layers[-1](x), dim=0)
        return out
```

**A2CAgent**

```python
class A2CAgent():
		critic = Critic(state_space=env.observation_space.shape[0]).to(device)
		actor = Actor(state_space=env.observation_space.shape[0],
	                action_space=env.action_space.n).to(device)
	
	  # Set Optimizer
		critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
		actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
```

**Main Flow**

```python
for epi in range(episodes):
		s = env.reset()
    done = False
    score = 0
    step = 0
    while (not done) and (step < max_step):
        # Get action
        a_prob = actor(torch.from_numpy(s).float().to(device))
        a_distrib = Categorical(a_prob)
        a = a_distrib.sample()

        # Interaction with Environment
        s_prime, r, done, _ = env.step(a.item())
        done_mask = 0 if done is True  else 1
        batch.append([s,r/100,s_prime,a_prob[a],done_mask])
```

**Train**

```python
v_s = critic(s_buf)
v_prime = critic(s_prime_buf)

Q = r_buf + gamma * v_prime.detach() * done_buf # value target
A =  Q - v_s                      # Advantage

# Update Critic
critic_optimizer.zero_grad()
critic_loss = F.mse_loss(v_s, Q.detach())
critic_loss.backward()
critic_optimizer.step()

# Update Actor
actor_optimizer.zero_grad()
actor_loss = 0
for idx, prob in enumerate(prob_buf):
    actor_loss += -A[idx].detach() * torch.log(prob)
actor_loss /= len(prob_buf) 
actor_loss.backward()
actor_optimizer.step()
```

**Environment**

```python
def render(self, mode='human', close=False):
	  print("├─────────────────────────────────")
	  print('│ total pot: {}'.format(self._totalpot))
	  if self._last_actions is not None:
		    pid = self._last_player.player_id
		    print('│ last action by player {}:'.format(pid))
		    print("│", format_action(self._last_player, self._last_actions[pid]))

	  (player_states, community_states) = self._get_current_state()
	  (player_infos, player_hands) = zip(*player_states)
	  (community_infos, community_cards) = community_states
	  print('│ community:')
	  print('│ -' + hand_to_str(community_cards))
	  print('│ players:')
	  for idx, hand in enumerate(player_hands):
		    print('│ {}{}stack: {}'.format(idx, hand_to_str(hand), self._seats[idx].stack))
```

                                          **Visualized Environment**

![visualized environment.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/66159018-5b6d-4ebc-9985-ab3a62a608f7/visualized_environment.png)

# V. Experimental Result

![Share_A2C(layer-128).PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/bf492ede-dcba-4798-9f72-198209fe66ad/Share_A2C(layer-128).png)

**Share-A2C(hidden-layer-dimension-128)**

![NS_A2C.PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e7869d61-2deb-44d4-83a7-91c68962ab00/NS_A2C.png)

**Not Share-A2C(hidden-layer-dimension-128)**

![Share_A2C(layer-64).PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/2b17bfc2-f139-490d-a6a1-7351e41bbadf/Share_A2C(layer-64).png)

**Share-A2C(hidden-layer-dimension-64)**

![NS_A2C(layer-64).PNG](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/1360691c-2d1a-488b-a9d3-c17d6b45a66e/NS_A2C(layer-64).png)

**Not Share-A2C(hidden-layer-dimension-64)**

# VI. Conclusion

- 네 가지 모델의 경우 정책 에이전트 알고리즘 특성 상 아무리 학습해도 랜덤 확률의 액션이 존재하기 때문에 reward가 수렴하지 못하는 것으로 보인다.
- Actor-Critic 알고리즘의 두 가지 타입 분석 결과, 입력을 해석하는 네트워크를 공유하는 타입의 구조 Share-A2C는 학습 속도는 느리지만 비교적 안정적으로 reward가 높은 값으로 수렴한다.
- Not Share-A2C의 경우 actor 네트워크와 critic 네트워크 모두 입력을 따로 해석하는 구조로 비교적 빠른 시간에 높은 reward를 달성했다. 하지만 에피소드가 진행될 수록 reward의 편차가 크다.
- hidden-layer-dimension-64의 경우 128의 경우보다 학습이 진행될 수록 reward가 수렴하는 경향이 나타났다.
- 분석 결과 두 가지 구조 중 입력을 해석하는 하이퍼파라미터를 공유하는 Share-A2C가 Not Share-A2C보다 좋은 성능을 보인다.
- 멀티 프로세싱을 하는 과정에 어려움이 있어 구현은 못했지만 추 후 State-of-the-art에 기여했던 A3C 알고리즘으로 구현할 예정이다.

# Reference

- 강화학습 이론

[바닥부터 배우는 강화 학습 - YES24](http://www.yes24.com/Product/Goods/92337949)

- Baseline code of Texas hold ‘em Environment

[https://github.com/wenkesj/holdem](https://github.com/wenkesj/holdem)

- Pytorch (deep-learning framework)

[점프 투 파이썬](https://wikidocs.net/book/2788)

- Actor-Critic for poker paper

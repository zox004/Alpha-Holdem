# Alpha Holdem -  Playing Texas hold 'em AI with DRL
## 주제

**강화학습을 이용한 홀덤 AI 구현**

- 규칙은 총 52장의 카드로 진행하며, 개인 카드 2장과 커뮤니티 카드 5장으로 족보를 맞춰서 높은 쪽이 승리하는 게임이다. 처음 개인 카드가 2장 주어지고 베팅을 한다. 그 후 커뮤니티 카드 3장, 1장, 1장이 주어지고 베팅을 하는 시스템이다. 플레이어는 각 차례마다 60초 이상의 시간제한이 주어진다.
- 알파고와는 달리 홀덤은 불완전 정보 게임이며, 바둑은 완전 정보 게임이다.
- 하지만, 홀덤 역시 경우의 수가 매우 큰 Continuous Action Space를 가져 일반적인 강화학습 알고리즘으로는 학습이 어렵다. 그러므로 딥러닝 기반 강화학습(DRL) 알고리즘을 사용해야 한다.
- 그 예시로, 2017년 카네기 맬런 대학에서 연구한 리브라투스는 강화학습과 CFR(Counterfactual Regret) 알고리즘을 사용해 좋은 성과를 이루어냈다.
- DRL 알고리즘 중 정책 기반 에이전트는 Contiuous Action Space에서도 학습 성능이 좋기 때문에 정책 기반 에이전트 중에서도 좋은 성능을 보여준 A3C 알고리즘을 사용할 것이다.

## Algorithm

### A3C

Multi processing 필요

## Environment

**MDP of the holdem Environment**

$< S, A, P, R, γ >$

$S$ : 현재 칩 개수, 현재 내 카드, 커뮤니티 카드, 현재 팟 상태, 방금 전 상대의 베팅 크기

$A$ : Fold, Check, Call, Raise, Bet, All-in (Continuous Action space = $10^{161}$ in No-Limit Betting)

$P : P_{ss'}^a = 1(∀a, ∀s, ∀s’ )$ 

$R$ : 라운드 승리 +2, 라운드 패배 -2, 최종 승리 +10, 최종 패배 -10, 낮은 밸류로 블러프에 성공했을 때 +1, 칩의 변화 +-0.01

$γ ∈ [0.05, 0.95]$ : discount factor

## 이해해야할 게임이론

**제로섬 게임**

- (1:1 상황) 플레이어A와 플레이어B가 있을 때 A가 돈을 따게 되면 B는 돈을 잃어 두 플레이어 간 이득과 손실의 합이 0이 된다.

**Nash Equilibrium**

- 게임 이론에서 경쟁자 대응에 따라 최선의 선택을 하면 서로가 자신의 선택을 바꾸지 않는 균형상태를 말한다. 상대방이 현재 전략을 유지한다는 전제 하에 나 자신도 현재 전략을 바꿀 유인이 없는 상태를 말하는 것으로 죄수의 딜레마와 밀접한 관계가 있다.

**CFR(CounterFactual Regret)**

- ~하지 않았다면(원인) ~했을텐데
    
    → 현재 state에서 실행했던 action이 아닌 other action을 했다면 other reward를 받았을텐데
    
- In Rock scissor paper
    
    strategy : normalization을 하며 현재의 strategy로 어떤 action을 할지 choice
    
    strategy_sum : target_policy
    
    regret : 현재의 state에서 other action을 choice했다면 어땠을지에 대한 value
    
    regret_sum : strategy을 뽑아내기 위해 episode마다 regret을 계속 sum하며 update
    
    **Process :** regret_sum으로부터 get_strategy를 통해 strategy를 얻는다. strategy_sum += strategy now_strategy의 probability로 action을 choice한다. action evalution 후 reward를 얻는다. 현재 state에서 다른 action을 했다면 어땠을지에 대한 식
    
    my_regret = self.get_reward(a, opponent_action) - my_reward을 통해 regret_sum을 update
    
    이 과정을 iteration
    
    → regret_sum을 통해 하지 말아야 할 action들을 배제하는 strategy를 만든다. 그 외의 action들을 normalize 후 확률을 계산해 action을 선택. iteration을 하며 현재 strategy에 대해 계속 regret_sum을 update하며 최선이 아닌 차선의 action을 선택하는 것.
    

## 기술 동향

## Libratus

2017년 카네기 멜런 대학이 개발한 리브라투스는 미국 피츠버그 슈퍼 컴퓨팅 센터의 브릿지 컴퓨터가 동원됐으며, 브릿지가 보유한 846개 컴퓨트 노드 중 600여개가 사용된다. 여기서 얻는 컴퓨팅 파워는 초당 1.35 페타플롭스 속도를 낸다. 최고 사양 노트북과 비교해 7천250배 빠르며, 메모리 용량은 274테라바이트에 달한다. 인공신경망을 쓰지 않는 대신 강화학습을 활용했고, 이를 통해 스스로 포커게임을 반복하면서 시행착오를 겪는 방법을 사용하였다. 상대방의 심리를 읽어낸다기보다는 매일 자신이 치렀던 게임 중 상대방 선수들이 치고 들어왔던 자신이 가진 취약점을 분석해 보완하는 작업을 거친다. 그 결과, 1:1로 세계 최고의 프로 선수 4인(지미 초우, 다니엘 맥컬리 등)을 상대로 게임에서 승리했다.

Libratus의 한계점은 1:1 대전 방식인 Heads-Up의 대전 방식을 선택한 것이다. 포커는 정보 비대칭적인 게임이지만 1:1로 하는 경우 다른 외부적 영향을 전혀 받지 않는다. 실제로 전문적인 포커 게임을 2명을 초과하는 인원들 끼리의 올인 베팅, 예상치 못한 플레이어로부터에서 나타나는 과감한 베팅과 블러핑 등으로 외부적 환경 요소가 존재하며, 이에 따른 영향을 받을 수밖에 없다.

## Pluribus

2019년에는 페이스북과 카네기 멜런 대학이 공동으로 개발한 플루리버스(Pluribus)가 6명이 붙은 포커 게임에서 프로 선수를 격파해 불완전 정보 게임에서도 인간을 능가하게 됐다. 다만 완전 정보 게임에 비하면 여전히 인간에 고전하고 있다. 플루리버스는 신념 기반 회귀학습 알고리즘을 이용했다. ReBeL(Recursive Belief-based Learning)이란 강화 학습과 검색으로 훈련된 인공지능 모델이다. 불완전 정보 게임의 해결책이라는 ReBeL은 자기 강화 학습을 통해 두 AI 모델인 가치 네트워크와 정책 네트워크를 훈련해 인간을 상대할 수 있는 유연한 알고리즘이라고 한다. 기존 포커 AI는 게임 할 때 생기는 변수를 다시 학습했지만 ReBeL은 게임 중 베팅 크기 등 변경 사항이 있어도 실시간으로 바로 학습한다. 하지만 악용 방지를 위해 ReBeL의 코드는 공개하지 않았다. 플루리버스는 각 플레이어가 가질 수 있는 다양한 신념의 확률 분포를 계산해 행동을 결정하도록 했고, 그 결과 인간 톱 플레이어도 뛰어넘는 성적을 거두었다. 리브라투스와는 다르게 6인 게임이 진행되었고 그 중 월드 시리즈 포커 챔피언인 크리스 퍼거슨을 포함해 다수를 상대로 승리를 따냈다. 이러한 AI의 성공은 포커게임 뿐만 아니라 비즈니스협상, 군사전략, 사이버보안, 의학적 치료 등 분야에서도 이러한 AI를 활용할 수 있다.

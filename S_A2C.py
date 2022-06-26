import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import holdemEnv
import numpy as np
import random

learning_rate = 0.0002
gamma = 0.98
n_rollout = 10
seed = 1

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.data = []

        self.fc1 = nn.Linear(11, 64)
        self.hidden1 = nn.Linear(64, 64)
        self.hidden2 = nn.Linear(64, 64)
        self.fc_pi = nn.Linear(64, 4)
        self.fc_v = nn.Linear(64, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def actor(self, x, softmax_dim = 0):
        x = F.relu(self.fc1(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = self.fc_pi(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob
    
    def critic(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        v = self.fc_v(x)
        return v
    
    def put_data(self, transition):
        self.data.append(transition)
    
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, done_lst = [], [], [], [], []
        for transition in self.data:
            s,a,r,s_prime,done = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r/100.0])
            s_prime_lst.append(s_prime)
            done_mask = 0.0 if done else 1.0
            done_lst.append([done_mask])

        s_batch, a_batch, r_batch, s_prime_batch, done_batch = [torch.tensor(s_lst, dtype=torch.float),
                                                                torch.tensor(a_lst, ),
                                                                torch.tensor(r_lst, dtype=torch.float),
                                                                torch.tensor(s_prime_lst, dtype=torch.float),
                                                                torch.tensor(done_lst, dtype=torch.float)]
        self.data = []
        return s_batch, a_batch, r_batch, s_prime_batch, done_batch

    def train(self):
        s, a, r, s_prime, done = self.make_batch()
        Q = r + gamma * self.critic(s_prime) * done
        A = Q - self.critic(s)

        actor = self.actor(s, softmax_dim=1)
        action = actor.gather(1,a)
        loss = -torch.log(action) * A.detach() + F.smooth_l1_loss(self.critic(s), Q.detach())

        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

max_step = 20000

if __name__ == "__main__":
    env = holdemEnv.TexasHoldemEnv(2)
    env.add_player(0, stack=1000000)
    env.add_player(1, stack=1000000)
    model = ActorCritic()
    print_interval = 20
    score = 0.0
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)    
    for epi in range(100):
        done = False
        (player_states, (community_infos, community_cards)) = env.reset()
        (player_infos, player_hands) = zip(*player_states)
        done = False
        score = 0
        my_hands, opp_hands = player_hands
        commu_cards = community_cards
        pot = community_infos[3]
        last_opp_bet = community_infos[-2]            
        s = my_hands + opp_hands + commu_cards + [pot] + [last_opp_bet] # pot 과 last_opp_bet은 int형이라 list 변환 후 더함
        s = np.array(s)
        start_stack = player_infos[0][2]
        step = 0
        env.render(mode='human')
        r = 0
        while (not done) and (step < max_step):
            for t in range(n_rollout):
                prob = model.actor(torch.from_numpy(s).float())
                distrib = Categorical(prob)
                a = distrib.sample()
                (player_states, (community_infos, community_cards)), rews, done, info = env.step(a.item())
                (player_infos, player_hands) = zip(*player_states)
                my_hands, opp_hands = player_hands
                commu_cards = community_cards
                pot = community_infos[3]
                last_opp_bet = community_infos[-2]            
                s_prime = my_hands + opp_hands + commu_cards + [pot] + [last_opp_bet] # pot 과 last_opp_bet은 int형이라 list 변환 후 더함
                s_prime = np.array(s_prime)
                end_stack = player_infos[1][2]
                player = player_infos[1][1]
                if player==1:
                    diff = end_stack - start_stack
                    # s_prime, r, done, _ = env.step(a.item()) # a.item() action의 확률 중 하나를 뽑아 그거를 int형으로 변환
                    if end_stack - start_stack > 0:
                        r += 1
                    else:
                        r -= 1
                    r += diff * 0.00001
                model.put_data((s,a,r,s_prime,done))

                s = s_prime
                score += r

                if done:
                    break
                env.render()
            model.train()
        
        print("epsiode: {}, score: {}".format(epi, score))
        score = 0.0
    env.close()


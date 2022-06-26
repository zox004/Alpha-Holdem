import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
# from torch.utils.tensorboard import SummaryWriter
from holdemEnv import TexasHoldemEnv

import numpy as np
import random

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

class Actor(nn.Module):
    def __init__(self, state_space=None, action_space=None):
        super(Actor, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_space, 64))
        self.layers.append(nn.Linear(64, 64))
        self.layers.append(nn.Linear(64, 64))
        self.layers.append(nn.Linear(64, action_space))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        out = F.softmax(self.layers[-1](x), dim=0)
        return out

def train(actor, critic, 
          critic_optimizer, actor_optimizer,
          gamma, batches, device):
          
    s_buf = []
    s_prime_buf = []
    r_buf = []
    prob_buf = []
    done_buf = []

    for batch in batches:
        s_buf.append(batch[0])
        r_buf.append(batch[1])
        s_prime_buf.append(batch[2])
        prob_buf.append(batch[3])
        done_buf.append(batch[4])
        
    s_buf = torch.FloatTensor(s_buf).to(device)
    r_buf = torch.FloatTensor(r_buf).unsqueeze(1).to(device)
    s_prime_buf = torch.FloatTensor(s_prime_buf).to(device)
    done_buf = torch.FloatTensor(done_buf).unsqueeze(1).to(device)

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

def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True

def save_model(model, path='default.pth'):
        torch.save(model.state_dict(), path)

if __name__ == "__main__":
    # Determine seeds
    seed = 1
    # Set gym environment
    env = TexasHoldemEnv(2)
    env.add_player(0, stack=1000000)
    env.add_player(1, stack=1000000)
    env_state_space = 11
    env_action_space = 4
    if torch.cuda.is_available():
        device = torch.device("cuda")
    device = torch.device("cpu")
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default 'log_dir' is "runs" - we'll be more specific here
    # writer = SummaryWriter('runs/')

    # set parameters
    actor_lr = 1e-3
    critic_lr = 1e-3
    episodes = 5000
    print_per_iter = 100
    max_step = 20000
    gamma = 0.98

    batch = []
    batch_size = 5 # 5 is best until now

    critic = Critic(state_space=env_state_space).to(device)
    
    actor = Actor(state_space=env_state_space,
                  action_space=env_action_space).to(device)
    
    # pre-trained model load
    # actor.load_state_dict(torch.load("Actor-Critic100_.pth"))
    # actor.eval()

    # Set Optimizer
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)
       
    for epi in range(100):
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
            # Get action
            a_prob = actor(torch.from_numpy(s).float().to(device))
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()
            # Interaction with Environment
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
                done_mask = 0 if done is True  else 1

                batch.append([s,r/100,s_prime,a_prob[a],done_mask])
                
                if len(batch) >= batch_size:
                    train(actor, critic, 
                        critic_optimizer, actor_optimizer, 
                        gamma,
                        batch,
                        device)
                    batch = []

                s = s_prime
                score += r
                step += 1
                env.render(mode='human')
        # Logging
        print("epsiode: {}, score: {}".format(epi, score))
        # writer.add_scalar('Rewards per epi', score, epi)

        # trained-model save
        # if epi%50==0 and epi!=0:
        #     save_model(actor, model_name+"{}_.pth".format(epi))

    # writer.close()
    env.close()

import gym

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

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
    model_name = "Actor-Critic"
    env_name = "CartPole-v1"
    seed = 1
    exp_num = 'SEED_'+str(seed)

    # Set gym environment
    env = gym.make(env_name)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    device = torch.device("cpu")
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    # default 'log_dir' is "runs" - we'll be more specific here
    writer = SummaryWriter('runs/'+env_name+"_"+model_name+"_"+exp_num)

    # set parameters
    actor_lr = 1e-4
    critic_lr = 1e-3
    episodes = 5000
    print_per_iter = 100
    max_step = 20000
    discount_rate = 0.99

    batch = []
    batch_size = 5 # 5 is best until now

    critic = Critic(state_space=env.observation_space.shape[0]).to(device)
    
    actor = Actor(state_space=env.observation_space.shape[0],
                  action_space=env.action_space.n).to(device)
    
    actor.load_state_dict(torch.load("Actor-Critic100_.pth"))
    actor.eval()

    # Set Optimizer
    critic_optimizer = optim.Adam(critic.parameters(), lr=critic_lr)
    actor_optimizer = optim.Adam(actor.parameters(), lr=actor_lr)

    for epi in range(episodes):
        s = env.reset()

        done = False
        score = 0
        

        step = 0
        while (not done) and (step < max_step):
            # if epi%print_per_iter == 0:
            #     env.render()

            # Get action
            a_prob = actor(torch.from_numpy(s).float().to(device))
            a_distrib = Categorical(a_prob)
            a = a_distrib.sample()

            # Interaction with Environment
            s_prime, r, done, _ = env.step(a.item())

            done_mask = 0 if done is True  else 1

            batch.append([s,r/100,s_prime,a_prob[a],done_mask])
            
            if len(batch) >= batch_size:
                train(actor, critic, 
                    critic_optimizer, actor_optimizer, 
                    discount_rate,
                    batch,
                    device)
                batch = []

            s = s_prime
            score += r
            step += 1


        # Logging
        print("epsiode: {}, score: {}".format(epi, score))
        writer.add_scalar('Rewards per epi', score, epi)
        # if epi%50==0 and epi!=0:
        #     save_model(actor, model_name+"{}_.pth".format(epi))

    writer.close()
    env.close()

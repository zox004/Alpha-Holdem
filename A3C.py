import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import holdemEnv
import utils
import numpy as np
import random
import time


# Critic Network
class Critic(nn.Module):
    def __init__(self, state_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(Critic, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        
        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(state_space, hidden_dim))

        # Add hidden layers
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, 1))

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = self.layers[-1](x)

        return out

# Actor Network
class Actor(nn.Module):
    def __init__(self, state_space=None,
                       action_space=None,
                       num_hidden_layer=2,
                       hidden_dim=None):

        super(Actor, self).__init__()

        # space size check
        assert state_space is not None, "None state_space input: state_space should be assigned."
        assert action_space is not None, "None action_space input: action_space should be assigned"
        

        if hidden_dim is None:
            hidden_dim = state_space * 2

        self.layers = nn.ModuleList()

        # Add input layer
        self.layers.append(nn.Linear(state_space, hidden_dim))

        # Add hidden layers
        for i in range(num_hidden_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.layers.append(nn.Linear(hidden_dim, action_space))

    def forward(self, x):

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))

        out = F.softmax(self.layers[-1](x), dim=0)

        return out


def train(global_Actor, global_Critic, device, rank):

    env = holdemEnv.TexasHoldemEnv(2)
    # env_state_space = env.observation_space.shape[0]
    env_state_space = 11
    # env_action_space = env.action_space.n
    env_action_space = 4
    env.add_player(0, stack=2000) # add a player to seat 0 with 2000 "chips"
    env.add_player(1, stack=2000) # add another player to seat 1 with 2000 "chips"    

    np.random.seed(seed+rank)
    random.seed(seed+rank)
    seed_torch(seed+rank)
    env.seed(seed+rank)

    local_Actor1 = Actor(state_space=env_state_space,
                  action_space=env_action_space,
                  num_hidden_layer=hidden_layer_num,
                  hidden_dim=hidden_dim_size).to(device)
    local_Critic1 = Critic(state_space=env_state_space,
                    num_hidden_layer=hidden_layer_num,
                    hidden_dim=hidden_dim_size).to(device)
    local_Actor2 = Actor(state_space=env_state_space,
                  action_space=env_action_space,
                  num_hidden_layer=hidden_layer_num,
                  hidden_dim=hidden_dim_size).to(device)
    local_Critic2 = Critic(state_space=env_state_space,
                    num_hidden_layer=hidden_layer_num,
                    hidden_dim=hidden_dim_size).to(device)


    local_Actor1.load_state_dict(global_Actor.state_dict())
    local_Critic1.load_state_dict(global_Critic.state_dict())
    local_Actor2.load_state_dict(global_Actor.state_dict())
    local_Critic2.load_state_dict(global_Critic.state_dict())

    batch = []

    # Set Optimizer
    actor_optimizer1 = optim.Adam(global_Actor.parameters(), lr=actor_lr)
    critic_optimizer1 = optim.Adam(global_Critic.parameters(), lr=critic_lr)
    actor_optimizer2 = optim.Adam(global_Actor.parameters(), lr=actor_lr)
    critic_optimizer2 = optim.Adam(global_Critic.parameters(), lr=critic_lr)

    for epi in range(1):
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
        step = 0
        env.render(mode='human')
        while (not done) and (step < max_step):
            # Get action
            a_prob = local_Actor1(torch.from_numpy(s).float().to(device))
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
            r = 0
            # s_prime, r, done, _ = env.step(a.item()) # a.item() action의 확률 중 하나를 뽑아 그거를 int형으로 변환

            done_mask = 0 if done is True  else 1

            batch.append([s,r/100,s_prime,a_prob[a],done_mask])
            
            if len(batch) >= batch_size:
                    s_buf = []
                    s_prime_buf = []
                    r_buf = []
                    prob_buf = []
                    done_buf = []

                    for item in batch:
                        s_buf.append(item[0])
                        r_buf.append(item[1])
                        s_prime_buf.append(item[2])
                        prob_buf.append(item[3])
                        done_buf.append(item[4])

                    s_buf = torch.FloatTensor(s_buf).to(device)
                    r_buf = torch.FloatTensor(r_buf).unsqueeze(1).to(device)
                    s_prime_buf = torch.FloatTensor(s_prime_buf).to(device)
                    done_buf = torch.FloatTensor(done_buf).unsqueeze(1).to(device)

                    v_s = local_Critic1(s_buf)
                    v_prime = local_Critic1(s_prime_buf)

                    Q = r_buf+discount_rate*v_prime.detach()*done_buf # value target
                    A =  Q - v_s                              # Advantage
                    
                    # Update Critic
                    critic_optimizer1.zero_grad()
                    critic_loss = F.mse_loss(v_s, Q.detach())
                    critic_loss.backward()
                    for global_param, local_param in zip(global_Critic.parameters(), local_Critic1.parameters()):
                        global_param._grad = local_param.grad
                    critic_optimizer1.step()

                    # Update Actor
                    actor_optimizer1.zero_grad()
                    actor_loss = 0
                    for idx, prob in enumerate(prob_buf):
                        actor_loss += -A[idx].detach() * torch.log(prob)
                    actor_loss /= len(prob_buf) 
                    actor_loss.backward()

                    for global_param, local_param in zip(global_Actor.parameters(), local_Actor1.parameters()):
                        global_param._grad = local_param.grad
                    actor_optimizer1.step()

                    local_Actor1.load_state_dict(global_Actor.state_dict())
                    local_Critic1.load_state_dict(global_Critic.state_dict())

                    batch = []

            s = s_prime
            
            score += r
            step += 1
            env.render(mode='human')
        
    env.close()
    print("Process {} Finished.".format(rank))



def test(global_Actor, device, rank):
    
    env = holdemEnv.TexasHoldemEnv(2)

    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)
    
    score = 0

    for epi in range(1):
        (player_states, (community_infos, community_cards)) = env.reset()
        (player_infos, player_hands) = zip(*player_states)
        done = False
        score = 0
        my_hands, opp_hands = player_hands
        commu_cards = community_cards
        pot = community_infos[3]
        last_opp_bet = community_infos[-2]            
        s = my_hands + opp_hands + commu_cards + [pot] + [last_opp_bet] # pot 과 last_opp_bet은 int형이라 list 변환 후 더함
        step = 0

        while (not done) and (step < max_step):
            # Get action
            a_prob = global_Actor(torch.from_numpy(s).float().to(device))
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
            r = 0

            s = s_prime
            score += r
            step += 1

        if epi % print_interval == 0:
            print("EPISODES:{}, SCORE:{}".format(epi, score/print_interval))
            score = 0

    env.close()
        


def seed_torch(seed):
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


# Global variables
model_name = "Actor-Critic"
env_name = "LunarLander-v2"
seed = 1
exp_num = 'SEED_'+str(seed)
print_interval = 10

# Global parameters
actor_lr = 1e-4
critic_lr = 1e-3
episodes = 10000
print_per_iter = 100
max_step = 20000
discount_rate = 0.99
batch_size = 5

hidden_layer_num = 2
hidden_dim_size = 128

# state = [card, commu_card, value, my chip, opp_chip, pot, opp_action, flop, dealer_btn]
# action = [fold, check, call, bet(33%, 50%, 100%), raise(2bet, 3bet), all_in]
# state가 dict로 표현해도 괜찮은지 모르겠음
# state = {"card":(None,None), "commu_card":(None,None,None,None,None), "value":(None,None,None), "my_chip":0,
#          "opp_chip":0, "pot":0, "opp_action":0, "flop":(0,0,0,0), "dealer_btn":False}

# state 간소화 -> state = [my_card1, my_card1_suit, my_card2, my_card1_suit, commu_card1, commu_card1_suit,
#                         commu_card2, commu_card2_suit, commu_card3, commu_card3_suit, commu_card4, commu_card4_suit,
#                         commu_card5, commu_card5_suit, pot, my_chip, opp_chip, opp_action]
state = [0] * 11
# action = [Check, Call : 상대 betting 금액과 동일, Raise : 상대 betting 금액의 2배, All_in]
action = [0.25, 0.25, 0.25, 0.25]
# reward = round win +1, chips fluctuation +- 0.005, round count -0.001
reward = {"round_win":1, "chips_fluctuation":0.005, "round_count":0.001}
if __name__ == "__main__":
    # Set gym environment
    env = holdemEnv.TexasHoldemEnv(2)
    
    # env_state_space = env.observation_space.shape[0]
    env_state_space = 11
    # env_action_space = env.action_space.n
    env_action_space = 4
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    device = torch.device("cpu")
    np.random.seed(seed)
    random.seed(seed)
    seed_torch(seed)
    env.seed(seed)

    global_Actor1 = Actor(state_space=env_state_space,
                  action_space=env_action_space,
                  num_hidden_layer=hidden_layer_num,
                  hidden_dim=hidden_dim_size).to(device)
    global_Critic1 = Critic(state_space=env_state_space,
                    num_hidden_layer=hidden_layer_num,
                    hidden_dim=hidden_dim_size).to(device)
    global_Actor2 = Actor(state_space=env_state_space,
                  action_space=env_action_space,
                  num_hidden_layer=hidden_layer_num,
                  hidden_dim=hidden_dim_size).to(device)
    global_Critic2 = Critic(state_space=env_state_space,
                    num_hidden_layer=hidden_layer_num,
                    hidden_dim=hidden_dim_size).to(device)

    env.close()

    global_Actor1.share_memory()
    global_Critic1.share_memory()
    global_Actor2.share_memory()
    global_Critic2.share_memory()

    processes = []
    process_num = 5
    # rank=1
    # train(global_Actor1, global_Critic1, device, rank)
    mp.set_start_method('spawn') # Must be spawn
    print("MP start method:",mp.get_start_method())

    for rank in range(process_num): 
        # if rank == 0:
        #     p1 = mp.Process(target=test, args=(global_Actor1, device, rank, ))
        #     p2 = mp.Process(target=test, args=(global_Actor2, device, rank, ))
        # else:
        p1 = mp.Process(target=train, args=(global_Actor1, global_Critic1, device, rank, ))
        # p2 = mp.Process(target=train, args=(global_Actor2, global_Critic2, device, rank, ))

        p1.start()
        processes.append(p1)
        # p2.start()
        # processes.append(p2)
    for p in processes:
        p.join() # waiting for processes's completion

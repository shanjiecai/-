import numpy as np
import gym
import os
import math
import random
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

MAX_EPISODE = 500
REWARD_THRESHOLD = 200
MAX_EP_STEPS = 5000
SAMPLE_NUMS = 30
RENDER = False
GAMMA = 0.9  # reward discount
LR_A = 0.001 # learning rate for actor
LR_C = 0.001 # learning rate for critic

env = gym.make("CartPole-v0")
init_state = env.reset()
N_F = env.observation_space.shape[0]
N_A = env.action_space.n
#Actor与Critic网络的搭建，均采用两层全连接神经网络
class Actor_net(nn.Module):
    def __init__(self, n_features,n_actions):
        super(Actor_net,self).__init__()
        self.l1 = nn.Linear(n_features,40)
        self.l2 = nn.Linear(40,n_actions)
    def forward(self,x):
        out1 = F.relu(self.l1(x))
        out2 = F.log_softmax(self.l2(out1),dim=1)
        return out2

class Critic_net(nn.Module):
    def __init__(self, n_input,n_output):
        super(Critic_net,self).__init__()
        self.l1 = nn.Linear(n_input,40)
        self.l2 = nn.Linear(40,n_output)
    def forward(self,x):
        out1 = F.relu(self.l1(x))
        out2 = F.relu(self.l2(out1))
        return out2

#辅助函数。包括网络的搭建，观测值的采样，环境的重置等
def init_env(env, actor_net, critic_net, sample_nums, init_state):
    states = []
    actions = []
    rewards = []
    is_done = False
    final_r = 0
    state = init_state

    for i in range(sample_nums):
        states.append(state)
        log_softmax_action = actor_net(Variable(torch.Tensor([state])))
        softmax_action = torch.exp(log_softmax_action)
        action = np.random.choice(N_A, p=softmax_action.cpu().data.numpy()[0])
        one_hot_action = [int(k == action) for k in range(N_A)]
        next_state, reward, done, _ = env.step(action)

        actions.append(one_hot_action)
        rewards.append(reward)
        final_state = next_state
        state = next_state
        if done:
            is_done = True
            state = env.reset()
            break
    if not is_done:
        final_r = critic_net(Variable(torch.Tensor([final_state]))).cpu().data.numpy()
    return states, actions, rewards, final_r, state


def discount_reward(r, gamma, final_r):
    discount_r = np.zeros_like(r)
    running_add = final_r
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discount_r[t] = running_add
    return discount_r

#actor critic网络及优化器初始化
actor = Actor_net(n_features=N_F,n_actions=N_A)
actor_optim = optim.Adam(actor.parameters(),lr=LR_A)
critic = Critic_net(n_input=N_F,n_output=1)
critic_optim = optim.Adam(critic.parameters(),lr=LR_C)

steps = []
test_results = []

for step in range(MAX_EP_STEPS):
    states,actions,rewards,final_r,current_state = init_env(env,actor,critic,SAMPLE_NUMS,init_state)
    init_state = current_state
    actions_var = Variable(torch.Tensor(actions).view(-1,N_A))
    states_var = Variable(torch.Tensor(states).view(-1,N_F))

    #训练策略网络
    actor_optim.zero_grad()
    log_softmax_actions = actor(states_var)
    vs = critic(states_var).detach()
    qs = Variable(torch.Tensor(discount_reward(rewards,GAMMA,final_r)))

    advantages = qs - vs
    actor_loss = -torch.mean(torch.sum(log_softmax_actions*actions_var,1)*advantages)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),0.5)
    actor_optim.step()

    #训练值网络
    critic_optim.zero_grad()
    target_values = qs
    values = critic(states_var)
    critic_loss = nn.MSELoss()(values,target_values)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(),0.5)
    critic_optim.step()

    if (step+1) % 10 == 0:
        result = 0
        state = env.reset()
        #每10步更新状态
        for test_epi in range(10):
            state=env.reset()
            for test_step in range(200):
                softmax_action = torch.exp(actor(Variable(torch.Tensor([state]))))

                action = np.argmax(softmax_action.data.numpy()[0])
                next_state,reward,done,_ = env.step(action)
                result += reward
                state = next_state
                if done:
                    break
        print("step:",step+1,"test result:",result/10.0)
        steps.append(step+1)
        test_results.append(result/10)

plt.figure(1)
plt.plot(steps,test_results)
plt.grid(True)
plt.xlabel('episodes')
plt.ylabel('reward')
plt.show()
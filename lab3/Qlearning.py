import gym
import random
import numpy as np


class q_learning():

    def __init__(self):
        # q_table是一个36*36*2的二维数组
        # 离散化后的状态共有2592种可能的取值，每种状态对应一个行动
        self.q_value=np.zeros((36*36+1, 2))

    def choose_action(self, observation, i):
        if random.random()>0.1*(0.99**i):
            num = self.state_number(observation)
            return np.argmax(self.q_value[num])
        else:
            return random.choice([0,1])

    def linspace_n(self, clip_min, clip_max, num):
        return np.linspace(clip_min, clip_max, num + 1)[1:-1]

    def state_number(self, observation):
        dig = [np.digitize(observation[0], bins=self.linspace_n(-2.4, 2.4, 6)),
               np.digitize(observation[1], bins=self.linspace_n(-3.0, 3.0, 6)),
               np.digitize(observation[2], bins=self.linspace_n(-0.5, 0.5, 6)),
               np.digitize(observation[3], bins=self.linspace_n(-2.0, 2.0, 6))]
        return sum([x * (6 ** i) for i, x in enumerate(dig)])

    def value_update(self, observation, reward, action, observation_):
        obs = self.state_number(observation)
        obs_ = self.state_number(observation_)
        self.q_value[obs][action] = 0.9*self.q_value[obs][action] + 0.1*(reward+0.9*max(self.q_value[obs_]))

    def game_start(self):
        env = gym.make('CartPole-v0')
        for i in range(2000):
            print(self.q_value)
            count = 0
            observation = env.reset()
            while True:
                action = self.choose_action(observation, i)
                observation_, reward, done, info = env.step(action)
                count += reward
                env.render()
                if done:
                    reward -= 200
                self.value_update(observation, reward, action, observation_)
                if done:
                    break
                observation = observation_
            print(i,':',count)
        env.close()
        print(self.q_value)


if __name__ == '__main__':
    game = q_learning()
    game.game_start()

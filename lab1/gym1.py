import gym
def decision(observation):
    if observation[0]<-1:
        return 1
    if observation[0]> 1:
        return 0
    if observation[2]>0.2:
        return 1
    if observation[2]<-0.2:
        return 0
    if observation[3] < 0.0:
        return 0
    else:
        return 1


env = gym.make('CartPole-v0')
for i_episode in range(20):
    observation = env.reset()
    for step in range(100):
        env.render()
        #print(observation)
        action = decision(observation)
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(step+1))
            break
    env.close()

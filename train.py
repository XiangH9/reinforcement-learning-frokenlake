import math
import pickle
# 将python对象序列化和反序列化，序列化：
# pickle.dump(obj, file) 方法将 Python 对象 obj 转换为字节流，并将其写入文件（或其他可写入对象）。
# 反：pickle.load(file) 方法从文件中读取字节流并将其反序列化为 Python 对象。

# 好处：实现持久化保存，在需要时可以重新加载对象，无需重新创建或者初始化
# 可以存储绝大多数对象，但是仍然存在一些对象无法存储
import gym
from tqdm import tqdm
# tqdm模块的作用是将我们的进程进行实时的展示

from terminaltables import AsciiTable
# 这一个主要用来美化我们的输出结果，创建各种表格，完成结果输出的美化

from .tabular_q_agent import TabularQAgent
# 最最主要的算法模块
from .map import map_config
# 记录了我们需要使用的地图的信息
import time

import matplotlib.pyplot as plt
import numpy as np

def exponential_decay(starter_learning_rate, global_step, decay_step,
                      decay_rate, mini_value=0.0):
    decayed_learning_rate = starter_learning_rate * math.pow(decay_rate,
                                                             math.floor(
                                                                 global_step / decay_step))
    return decayed_learning_rate if decayed_learning_rate > mini_value else mini_value
# 这里就是将一开始学习率做一个衰减，估计会不断的传入某一个参数是变化的，然后就能够更好的计算出价值函数
# 这里是学习率随时间衰减
# 更好的利用强化学习的算法


def save_rewards_as_pickle(rewards, filename='q_learning-rewards.pkl'):
    with open(filename, 'wb') as file:
        pickle.dump(rewards, file)

# 回报输出为字节流函数，保存下来

def train(tabular_q_agent, env, train_episodes=300000):
    table_header = ['Episode', 'learning_rate', 'eps_rate', 'reward', 'step']
    rewards = []

    table_data = [table_header]

    for episode in tqdm(range(train_episodes)):
        learning_rate = exponential_decay(0.9, episode, 1000, 0.99)
        eps_rate = exponential_decay(1.0, episode, 1000, 0.97, 0.001)
        # 让学习率发生变化
        # 探索率也会随着每一轮时间的变化而发生变化

        all_reward, step_count = tabular_q_agent.learn(env, learning_rate,
                                                       eps_rate)
        rewards.append(all_reward)  # 记录每个Episode的累计奖励
        if not episode % 1000:
            table_data.append([episode, round(learning_rate,3), round(eps_rate,3), round(all_reward,3), step_count])
            table = AsciiTable(table_data)
            # 这里语句的作用就是直观显示每一轮训练过程的数据
            tqdm.write(table.table)
            # print(DataFrame(tabular_q_agent.q))
    save_rewards_as_pickle(rewards)
    window_size = 500
    smoothed_data = np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')
    # 通过和归一化做卷积函数实现回报的归一化，并且最终将结果展示出来

    plt.plot(smoothed_data)
    plt.title("Smoothed Data Using Moving Average(window_size=500)")
    plt.xlabel("Episodes")
    plt.ylabel("Reward")
    plt.show()

from q_learning.command.SerialThread import SerialThread as Se
st = Se("COM8")


# 接下来就是主函数，直接调用我们编写的函数
def main():
    env = gym.make('FrozenLake-v1',desc=map_config['desc'],map_name=map_config['map_name'],is_slippery=map_config['is_slippery'])
    env.seed(0)  # make sure result is reproducible
    tabular_q_agent = TabularQAgent(env.observation_space, env.action_space, n_iter=100, discount=1, st=st)
    # tabular_q_agent = TabularQAgent(env.observation_space, env.action_space, n_iter=100, discount=1)

    train(tabular_q_agent, env)

    
    # st.send().led(2,0,0,255)
    # time.sleep(2)

    tabular_q_agent.test(env)
    # tabular_q_agent.export()


if __name__ == "__main__":
    main()
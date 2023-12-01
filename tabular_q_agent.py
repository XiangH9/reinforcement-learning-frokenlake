from collections import defaultdict
#一个有用的数据结构，dict的子类，用于创建默认值为零的字典
import functools
#一个重要的功能是提供了 functools.wraps 装饰器，用于更新被装饰函数的元数据，以便更好地保留原函数的信息。
import pickle
#pickle导入导出
import numpy as np
import time

from gym.spaces import discrete
#gym 提供了 gym.spaces 模块来处理不同类型的状态空间和动作空间。
#gym.spaces 模块中的 discrete 类是用于定义离散型空间的一种，它表示一个有限的整数集合，代表了离散的状态或动作空间。
#可以用数来表示离散空间的动作范围

class UnsupportedSpace(Exception):
    pass

# 返回一个包含N个0的列表
def generate_zeros(n):
    return [0] * n


class TabularQAgent(object):
    """
    Agent implementing tabular Q-learning.
    """

    def __init__(self, observation_space, action_space, st = None, **userconfig):
        self.st = st
        # if not isinstance检查A是否是B的类型
        # 这里确保观察到的空间和行为空间都是离散的空间
        if not isinstance(observation_space, discrete.Discrete):
            raise UnsupportedSpace('Observation space {} incompatible with {}. (Only supports Discrete observation spaces.)'.format(observation_space, self))
        if not isinstance(action_space, discrete.Discrete):
            raise UnsupportedSpace('Action space {} incompatible with {}. (Only supports Discrete action spaces.)'.format(action_space, self))
        self.observation_space = observation_space
        self.action_space = action_space
        self.action_n = action_space.n
        self.config = {
            "init_mean": 0.0,      # Initialize Q values with this mean
            "init_std": 0.0,       # Initialize Q values with this standard deviation
            "learning_rate": 0.5,
            "eps": 0.05,            # Epsilon in epsilon greedy policies
            "discount": 0.99,
            "n_iter": 10000}        # Number of iterations
        # 存储一些重要的信息，初始Q表的平均值，方差，学习率，探索率，折扣率，最大探索步数
        self.config.update(userconfig)
        # self.q = defaultdict(lambda: self.config["init_std"] * np.random.randn(self.action_n) + self.config["init_mean"])
        self.q = defaultdict(functools.partial(generate_zeros, n=self.action_n))
        # 创建generate的部分应用版本，将n值具体为某个数

    def act(self, observation, eps=None):
        # 这里在具体行动时，会具有一定的探索概率，防止系统陷入停滞
        if eps is None:
            eps = self.config["eps"]
        # epsilon greedy.
        action = np.argmax(self.q[observation]) if np.random.random() > eps else self.action_space.sample()
        return action

    def learn(self, env, learning_rate=None, eps=None):
        if learning_rate is None:
            learning_rate = self.config["learning_rate"]

        obs = env.reset()
        # env.render(mode='human')

        rAll = 0
        step_count = 0
        # 这里开始训练

        for t in range(self.config["n_iter"]):
            action = self.act(obs, eps)
            obs2, reward, done, _ = env.step(action)
            # env.render(mode='human')

            # Get negative reward every step
            if reward == 0:
                reward = -0.005

            # if agent sucked at same position, punish it
            if obs == obs2:
                reward = -0.01

            # if agent fill to hole then die, punish it
            if done and not reward:
                reward = -1

            future = 0.0
            if not done:
                future = np.max(self.q[obs2])
            # 没有到下一步就会采取贪心策略
            
            self.q[obs][action] = (1 - learning_rate) * self.q[obs][action] + learning_rate * (reward + self.config["discount"] * future)
            # Q-learning算法的更新策略

            obs = obs2

            rAll += reward
            step_count += 1
            # 记录最终的回报和步数

            if done:
                break

        return rAll, step_count

    def test(self, env,show=True):
        # test模块即进行最终测试，控制飞机的模块
        maplocationX=0
        maplocationY=0
        # 这里为了更好的保证飞机的运行，不至于驶出地图外，我们增加了一些限制，
        # 但事实上，去掉这一部分限制，程序也能够正确运行

        self.st.send().takeoff(50)
        time.sleep(3)
        obs = env.reset()
        if show:
            env.render('human')
        action_list=['Left','Down','Right','UP']
        pos=0
        for t in range(self.config["n_iter"]):
            if show:
                env.render('human')

            action = self.act(obs, eps=0)

            obs2, reward, done, _ = env.step(action)
            print("obs2=",obs2)

            dis=obs2-pos
            pos=obs2
            # 这里为了保证在随机模式下也能够正常运行，我们通过人物实际的位置来确定实际执行命令，并且依此给飞机发送命令
            if dis==-8:
                taction=3
            elif dis==8:
                taction=1
            elif dis==-1:
                taction=0
            elif dis==1:
                taction=2
            else:
                taction=-1

            if show:
                env.render('human')
                print(action_list[taction])

                if taction == 3:  # up
                    # if maplocationX==0 and maplocationY-1==3:
                    #     pass
                    # elif maplocationX==1 and maplocationY-1==1:
                    #     pass
                    # elif maplocationX==3 and maplocationY-1==1:
                    #     pass
                    # elif maplocationX==3 and maplocationY-1==2:
                    #     pass

                    if maplocationY > 0:  
                        self.st.send().forward(50)
                        time.sleep(3)
                        print("tag_id==",end="")
                        print(self.st.vision_sensor_info().tag_id)
                        time.sleep(1)
                        maplocationY-=1
                    else:
                        maplocationY=0
                        pass


                elif taction == 1:  # 下
                    # if maplocationX==0 and maplocationY+1==3:
                    #     pass
                    # elif maplocationX==1 and maplocationY+1==1:
                    #     pass
                    # elif maplocationX==3 and maplocationY+1==1:
                    #     pass
                    # elif maplocationX==3 and maplocationY+1==2:
                    #     pass

                    if maplocationY < 7:
                        self.st.send().back(50)
                        time.sleep(3)
                        print("tag_id==",end="")
                        print(self.st.vision_sensor_info().tag_id)
                        time.sleep(1)
                        maplocationY+=1
                    else:
                        maplocationY=3
                        pass


                elif taction == 2:  # 右
                    # if maplocationX+1==0 and maplocationY==3:
                    #     pass
                    # elif maplocationX+1==1 and maplocationY==1:
                    #     pass
                    # elif maplocationX+1==3 and maplocationY==1:
                    #     pass
                    # elif maplocationX+1==3 and maplocationY==2:
                    #     pass

                    if maplocationX < 7:
                        self.st.send().right(55)
                        time.sleep(3)
                        print("tag_id==",end="")
                        print(self.st.vision_sensor_info().tag_id)
                        time.sleep(1)
                        maplocationX+=1
                    else:
                        maplocationX=3
                        pass


                elif taction == 0:  # 左
                    # if maplocationX-1==0 and maplocationY==3:
                    #     pass
                    # elif maplocationX-1==1 and maplocationY==1:
                    #     pass
                    # elif maplocationX-1==3 and maplocationY==1:
                    #     pass
                    # elif maplocationX-1==3 and maplocationY==2:
                    #     pass
                    if maplocationX >0:
                        self.st.send().left(50)
                        time.sleep(3)
                        print("tag_id==",end="")
                        print(self.st.vision_sensor_info().tag_id)
                        time.sleep(1)
                        maplocationX-=1
                    else:
                        maplocationX=0
                        pass
                else:
                    pass

            
            if done:
                self.st.send().land()
                time.sleep(3)
                # break
                if not reward:
                    return 0,t+1
                return 1,t+1

            obs = obs2
        self.st.send().up(30)
        time.sleep(3)
        self.st.send().flip(1, 1)
        time.sleep(3)
        self.st.send().land()
        time.sleep(3)
        return 0,self.config["n_iter"]

    def export(self, file="./pretrained_model/parameter.pkl"):
        with open(file, 'wb') as fd:
            pickle.dump(self, fd)
    # 设置了导出和下载模块，保证了模型的可复现

    @staticmethod
    def load(file="./pretrained_model/parameter.pkl"):
        with open(file, 'rb') as fd:
            instance = pickle.load(fd)
        return instance
    # 下载的算法

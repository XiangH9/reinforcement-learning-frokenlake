import tensorflow.compat.v1 as tf
# 现有tensorflow版本都是2.x版本，这里提供与 TensorFlow 1.x 版本兼容的接口
import numpy as np
import gym
import matplotlib.pyplot as plt

tf.compat.v1.disable_eager_execution()
# 使用 tf.compat.v1.disable_eager_execution() 可以在 TensorFlow 2.x 中显式地禁用 Eager Execution 模式，使得代码更接近 TensorFlow 1.x 中的图计算模式。

#
# from gym.envs.registration import register
# # register(
# #     id='FrozenLakeNotSlippery-v0',
# #     entry_point='gym.envs.toy_text:FrozenLakeEnv',
# #     kwargs={'map_name' : '4x4', 'is_slippery': False},
# #     max_episode_steps=100,
# #     reward_threshold=0.78, # optimum = .8196
# # )


tf.set_random_seed(1)
np.random.seed(1)

class DQN():
    def __init__(self,nstate,naction):
        self.nstate=nstate   # 状态空间的维度
        self.naction=naction  # 动作空间的维度
        self.sess = tf.Session()  # tensorflow会话，用于执行计算图
        self.memcnt=0      # 记录回放缓冲区中当然存储的数据数量
        self.BATCH_SIZE = 64  # 每次从缓冲区中采样量的大小
        self.LR = 0.0012                      # learning rate
        self.EPSILON = 0.92                 # greedy policy
        self.GAMMA = 0.9999                   # reward discount
        self.MEM_CAP = 2000     # 回放缓冲区大小
        self.mem= np.zeros((self.MEM_CAP, self.nstate * 2 + 2))     # initialize memory
        self.updataT=150       # 更新目标网络的频率
        self.built_net()        # 方法用于构建神经网络


    def built_net(self):
        self.s = tf.placeholder(tf.float64, [None,self.nstate])
        self.a = tf.placeholder(tf.int32, [None,])
        self.r = tf.placeholder(tf.float64, [None,])
        self.s_ = tf.placeholder(tf.float64, [None,self.nstate])
        # 状态、动作、奖励、下一状态的占位符，用于输入各个数据

        with tf.variable_scope('q'):                                  # evaluation network
            l_eval = tf.layers.dense(self.s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))
            self.q = tf.layers.dense(l_eval, self.naction, kernel_initializer=tf.random_normal_initializer(0, 0.1))
        # 这里是训练用的网络
        # q变量的作用域，
        # l_eval: 代表评估网络的隐藏层，具有10个神经元，激活函数为 ReLU。
        # self.q: 代表 Q-value 的输出层，其神经元数量为 self.naction，即动作的数量。这是代理根据当前状态估计的 Q-value。

        # 目标网络，有一个 ’q_next‘的作用域
        # l_target: 代表目标网络的隐藏层，具有10个神经元，激活函数为 ReLU。
        # q_next: 代表目标网络的输出层，其神经元数量为 self.naction。目标网络的参数在训练过程中不会更新（trainable=False）。
        with tf.variable_scope('q_next'):                                           # target network, not to train
            l_target = tf.layers.dense(self.s_, 10, tf.nn.relu, trainable=False)
            q_next = tf.layers.dense(l_target, self.naction, trainable=False)

        #计算目标的q-value
        q_target = self.r + self.GAMMA * tf.reduce_max(q_next, axis=1)    #q_next:  shape=(None, naction),
        # 计算当前估计的q-value
        a_index=tf.stack([tf.range(self.BATCH_SIZE,dtype=tf.int32),self.a],axis=1)
        q_eval=tf.gather_nd(params=self.q,indices=a_index)


        loss=tf.losses.mean_squared_error(q_target,q_eval)
        # 损失函数

        self.train=tf.train.AdamOptimizer(self.LR).minimize(loss)
        #  q现实target_net- Q估计，定义优化操作
        self.sess.run(tf.global_variables_initializer())
        # 初始化变量，保证模型可以正确运行

    # greedy策略，但有概率随机选择一个动作（探索）
    def choose_action(self,status):
        fs = np.zeros((1,self.nstate))
        fs[0,status]=1.0  # ONE HOT
        if  np.random.uniform(0.0,1.0)<self.EPSILON:
            action=np.argmax( self.sess.run(self.q,feed_dict={self.s:fs}))
        else:
            action=np.random.randint(0,self.naction)
        return action

# 每次训练之前，先检查是否需要更新目标网络
    def learn(self):
        if(self.memcnt%self.updataT==0):
            t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')
            e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')
            self.sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])
        rand_indexs=np.random.choice(self.MEM_CAP,self.BATCH_SIZE,replace=False)
        temp=self.mem[rand_indexs]
        bs = temp[:,0:self.nstate]#.reshape(self.BATCH_SIZE,NSTATUS)
        ba = temp[:,self.nstate]
        br = temp[:,self.nstate+1]
        bs_ = temp[:,self.nstate+2:]#.reshape(self.BATCH_SIZE,NSTATUS)
        self.sess.run(self.train, feed_dict={self.s:bs,self.a:ba,self.r:br,self.s_:bs_})
# 从回放缓冲区随机选择一批样本，利用这批样本对评估网络进行一次优化操作

    def storeExp(self,s,a,r,s_):
        fs = np.zeros(self.nstate)
        fs[s] = 1.0                       # ONE HOT
        fs_ = np.zeros(self.nstate)
        fs_[s_] = 1.0                          # ONE HOT
        self.mem[self.memcnt%self.MEM_CAP]=np.hstack([fs,a,r,fs_])
        self.memcnt+=1
        # one-hot编码后，存储到回放缓冲区，采用循环队列的模式，确保不会超过回放缓冲区的容量

    def show(self):
        print("show")
        obs = env.reset()
        env.render('human')
        for t in range(10000):
            env.render('human')
            action = dqn.choose_action(obs)
            obs2, reward, done, _ = env.step(action)
            env.render('human')
            if done:
                break
            obs = obs2
    # 展示结果


    def run(self,numsteps):
        cnt_win = 0 # 记录最近50次中完成任务的次数
        winrate_recorder = 0 # 记录最近10回合中成功完成任务的次数
        all_r=0.0  # 记录所有回合的累计奖励
        win_rate=[]
        for i in range(numsteps):
            s=env.reset()
            done=False
            while(not done):
                a=self.choose_action(s)
                s_,r,done,_=env.step(a)
                all_r+=r
                self.storeExp(s,a,r,s_)
                if(self.memcnt>self.MEM_CAP):
                    self.learn()  # 经验池满，则进行一次学习
                    if(done):
                        if(s_==self.nstate-1):
                            cnt_win+=1.0
                            winrate_recorder+=1.0
                            # 检查任务是否完成，更新数据
                s=s_
            if (i % 10 == 0):
                win_rate.append(winrate_recorder / 10)
                winrate_recorder = 0

            if (i % 50 == 0):
                print("period: ",i, ": ")
                if (cnt_win / 50 > 0.4):
                     self.EPSILON += 0.01
                elif (cnt_win / 50 > 0.2):
                      self.EPSILON += 0.005
                elif (cnt_win / 50 > 0.1):
                      self.EPSILON += 0.003
                elif (cnt_win / 50 > 0.05):
                      self.EPSILON += 0.001
                # 更具实际胜率对贪心策略的执行率做出调整，越大越不可能执行随机动作

                print("current accuracy: %.2f %%" %(cnt_win/50.0*100))
                # win_rate.append(cnt_win / 50)
                cnt_win=0  # 重置 cnt_win 为0，为下一个50回合的统计做准备。
                print("Global accuracy : %.2f %%" %(all_r / (i+1)*100))
        print("Global accuracy : ",all_r/numsteps*100,"%")

        x_axis = [i * 10 for i in range(len(win_rate))]
        plt.plot(x_axis, win_rate)
        plt.show()
        # 绘制训练过程中平均胜率的曲线

env = gym.make('FrozenLake-v1',map_name='8x8',is_slippery=True)
#env = gym.make('FrozenLake8x8-v0')
env = env.unwrapped
dqn=DQN(env.observation_space.n,env.action_space.n)
dqn.run(1000)
dqn.show()


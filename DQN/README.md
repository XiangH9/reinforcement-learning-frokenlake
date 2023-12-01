# DQN Algorithm

本文件仅描述了文件架构以及训练和测试的流程，创新点及其他在报告的pdf文件中。请阅读本说明文档后进行测试或训练。

## Abstract

​	冰湖环境是一个4*4的网格。格子包含三种类型：起始格，目标格，安全的冰格和危险的冰洞。目标是让一个agent能够自己从起点到终点，并且不会在中途掉入洞中。在任何时候，agent都可以选择从上，下，左，右四个中选择一个方向进行一步移动。而后会使问题变复杂的是，在冰湖环境中，还有风会偶尔把agent吹得偏离到一个并非它移动目标的格子上。因此，在这种环境下agent就不太可能每次都能表现完美，但是学习如何去避免掉入洞中并达到目标格还是可以做到的。agent每走一步的回报（reward）都是0，除了在达到目标时为1。

​	熟悉使用Gym库；熟悉强化学习代码；在Stochastic 和 Deterministic两种模式下完成仿真模型训练，生成无人机飞行路径；在Stochastic 和 Deterministic模式下，用python控制无人机飞越冰湖，可视化结果，并完成验收。

## Introduction

​	本次运用了DQN算法。DQN算法的核心就是用一个人工神经网络来代替Q表格，即动作价值函数。把Q表格的更新问题变成一个函数拟合问题，相近的状态得到相近的输出动作。主要的模块包括Experience Replay和Fixed Q-targets。



## Requirements

* tensorflow 
* gym == 0.23.1
* numpy
* Matplotplib

## Usages
 完整数据集训练和测试 ，训练完后会展示使用该模型测试的结果。

```shell
python DQN.py
```

接口：

```python
env = gym.make('FrozenLake-v1',is_slippery=False)
env = env.unwrapped
dqn=DQN(env.observation_space.n,env.action_space.n)
dqn.run(1000)
dqn.show()
```

  其中，env为设置棋盘的样式种类，如上为使用gym自带的棋盘布局完成训练。也可以使用自定义的棋盘完成训练和测试。

下方dqn.run()中参数为epoch数目，可以调节epoch大小来调节训练的轮次。

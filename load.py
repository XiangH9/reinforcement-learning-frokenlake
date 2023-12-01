import gym
from tqdm import tqdm

from .tabular_q_agent import TabularQAgent
from .map import map_config

def main():
    env = gym.make('FrozenLake-v1',desc=map_config['desc'],map_name=map_config['map_name'],is_slippery=map_config['is_slippery'])

    tabular_q_agent = TabularQAgent.load()
    
    show=True
    if show:
        tabular_q_agent.test(env,True)
    else:
        test_episodes=100000
        success=0
        total_step=0
        for i in tqdm(range(test_episodes)):
            reward,step=tabular_q_agent.test(env,show=False)
            success+=reward
            total_step+=step
        print(success/test_episodes,total_step/test_episodes)


if __name__ == "__main__":
    main()

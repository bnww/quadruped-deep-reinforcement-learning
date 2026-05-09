from anymal_env import ConfigurableANYmal, smooth_forward_reward_2
from DDPG_Model_Tim import ddpg_learn
import numpy as np

env = ConfigurableANYmal()
env.set_reward_function(smooth_forward_reward_2(0.5))

policy, rewards = ddpg_learn(env,0.99, 1000, 0.1, 5,10)
policy.save("ddpg_model_5")
np.save("ddpg_model_5_rewards", rewards)

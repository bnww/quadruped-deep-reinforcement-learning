from anymal_env import ConfigurableANYmal, smooth_forward_reward_3, simple_standing_reward
from SAC_Model_Tim import sac_learn
import numpy as np

env = ConfigurableANYmal()
env.set_reward_function(smooth_forward_reward_3(0.01))

policy, rewards = sac_learn(env,0.9999, 200000, 0.05, 0.2,32, False, True)
policy.save("sac_model_13")
np.save("sac_model_13_rewards", rewards)

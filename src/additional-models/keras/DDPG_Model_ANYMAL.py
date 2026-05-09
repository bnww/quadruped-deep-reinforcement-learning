from tensorflow import GradientTape, concat, reduce_mean
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import tanh
from collections import namedtuple
from timeit import default_timer as timer
from time import perf_counter_ns
from multiprocessing import Process, Queue
import numpy as np
import os

Transition = namedtuple('Transition', ['state', 'action','reward','new_state'])

class TransitionBatch:
    def __init__(self, transitions):
        self.states, self.actions, self.rewards, self.new_states = tuple(np.array(i) for i in zip(*transitions))

class Memory:
    def __init__(self, batch_size = 10):
        self.items = []
        self.batch_size = batch_size
        
    def put(self, transition):
        self.items.append(transition)
        
    def get_batch(self):
        choices = np.random.choice(len(self.items), self.batch_size)
        return TransitionBatch(np.array(self.items, dtype=object)[choices])

def build_actor_model(env):
    def activation(tensor_in):
        return tanh(tensor_in) * 80

    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(len(env.state[0]) + len(env.state[1]),)))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(env.action_space.shape[0], activation=activation))

    model.compile(optimizer='adam')
    return model

def build_critic_model(env):
    model = Sequential()
    model.add(Dense(1000, activation='relu', input_shape=(len(env.state[0]) + len(env.state[1]) + env.action_space.shape[0],)))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='huber', optimizer='adam')
    return model

def get_flattened_state(observation):
    return np.concatenate((observation['state']['Q'], observation['state']['V'] ))

def action_gaussian_noise(scale, env):
    return [np.random.normal(scale=scale) for _ in range(env.action_space.shape[0])]

def sanitize_action(action):
    return [80 if a > 80 else -80 if a < -80 else a for a in (action)]


def ddpg_learn(env, discount, no_episodes, learn_rate, exploration_value, s_l):
    env.reset()
    memory = Memory(10 * s_l)
    actor_model = build_actor_model(env)
    target_actor_model = build_actor_model(env)
    critic_model = build_critic_model(env)
    target_critic_model = build_critic_model(env)
    rewards = []
    i = 0
    while i < (no_episodes):
        start_time = timer()
        i += 1
        if (i% 100 == 0):
            print("saving backup")
            target_actor_model.save("ddpg_backup_2")
        env.reset()
        state = np.concatenate(env.state)
        done = False
        episode_length = 0
        ep_reward = 0
        while not done and episode_length < 10000:
            episode_length += 1
            isErr = False
            # run step and store results in memory
            if i <= s_l:
                action = env.action_space.sample()
            else:
                action = sanitize_action(actor_model(np.array([state]))[0] + action_gaussian_noise(exploration_value, env))
            try:
                _, reward, done, _ = env.step(action)
                ep_reward += reward
                next_state = np.concatenate(env.state)
            
                memory.put(Transition(state, action, reward, next_state))
                state = next_state
                
            except RuntimeError as err:
                print(err)
                isErr = True
                done = True
                i -= 1
            
            if i % s_l == 0 :
                t = memory.get_batch()

                y = t.rewards + discount * target_critic_model(concat((t.new_states, target_actor_model(t.new_states)),axis=1))

                # Gradient descent step for critic model
                critic_model.train_on_batch(concat((t.states, t.actions), axis=1), y)

                # Gradient ascent step for actor model
                with GradientTape() as g:
                    actions = actor_model(t.states)
                    loss = -reduce_mean(critic_model(concat((t.states,actions), axis=1)),axis=0)
                gradients = g.gradient(loss, actor_model.trainable_variables)
                actor_model.optimizer.apply_gradients(
                    zip(gradients, actor_model.trainable_variables)
                )
                    
                # Perform updates of target models - polyac averaging
                def get_weight_updates(model, target_model):
                    model_weights, target_weights = model.get_weights(), target_model.get_weights()
                    return [learn_rate * weight + (1-learn_rate)* target_weights[i] for i, weight in enumerate(model_weights)]
                
                target_actor_model.set_weights(get_weight_updates(actor_model,target_actor_model))
                target_critic_model.set_weights(get_weight_updates(critic_model,target_critic_model))
        rewards.append(ep_reward)

        end_time = timer()
        time_taken = end_time - start_time
        if i % s_l == 0:
            print("episode", i, "time", time_taken, "estimated time remaining", time_taken * (no_episodes - i)/s_l)
        if not done:
            print("episode reached completion!")

    return target_actor_model, rewards

def parallel_ddpg_learn(env, discount, no_episodes, learn_rate, exploration_value):
    env.reset()
    memory = Memory()
    actor_model = build_actor_model(env)
    target_actor_model = build_actor_model(env)
    critic_model = build_critic_model(env)
    target_critic_model = build_critic_model(env)
    rewards = []
    q = Queue()
    i = 0
    while i < (no_episodes):
        processes = []
        for _ in range(os.cpu_count()):
            i += 1
            if (i% 100 == 0):
                print("saving backup")
                actor_model.save("ddpg_backup")

                processes.append(Process(target=parallel_ddpg_learn_episode, args = (env, actor_model, critic_model, memory, target_actor_model, target_critic_model, discount, exploration_value,q)))
            
            rewards.append(ep_reward)

            
    return target_actor_model, rewards

# def parallel_ddpg_learn_episode(env, actor_model, critic_model, memory, target_actor_model, target_critic_model, discount, exploration_value, q):
#     env = env.copy()
#     env.reset()
#     state = np.concatenate(env.state)
#     done = False
#     episode_length = 0
#     ep_reward = 0
#     while not done and episode_length < 10000:
#         episode_length += 1

#         # run step and store results in memory

#         action = sanitize_action(actor_model(np.array([state]))[0] + action_gaussian_noise(exploration_value, env))
#         try:
#             _, reward, done, _ = env.step(action)
#             ep_reward += reward
#             next_state = np.concatenate(env.state)
        
#             memory.put(Transition(state, action, reward, next_state))
#             state = next_state
            
#         except RuntimeError as err:
#             print(err)
#             done = True
#             i -= 1
        
#         t = memory.get_batch()

#         y = t.rewards + discount * target_critic_model(concat((t.new_states, target_actor_model(t.new_states)),axis=1))

#         # Gradient descent step for critic model
#         critic_model.train_on_batch(concat((t.states, t.actions), axis=1), y)

#         # Gradient ascent step for actor model
#         with GradientTape() as g:
#             actions = actor_model(t.states)
#             loss = -critic_model(concat((t.states,actions), axis=1))
#         gradients = g.gradient(loss, actor_model.trainable_variables)
#         actor_model.optimizer.apply_gradients(
#             zip(gradients, actor_model.trainable_variables)
#         )
            
#         # Perform updates of target models - polyac averaging
#         def get_weight_updates(model, target_model):
#             model_weights, target_weights = model.get_weights(), target_model.get_weights()
#             return [learn_rate * weight + (1-learn_rate)* target_weights[i] for i, weight in enumerate(model_weights)]
        
#         target_actor_model.set_weights(get_weight_updates(actor_model,target_actor_model))
#         target_critic_model.set_weights(get_weight_updates(critic_model,target_critic_model))
#     q.put((actor_model, critic_model,memory,ep_reward))
from tensorflow import clip_by_value
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.activations import tanh
from tensorflow.keras.optimizers import Adam
from collections import namedtuple
import numpy as np
# rng = np.random.Generator(np.random.SFC64())
rng = np.random.default_rng()

class Transition:
    def __init__(self, state, action,reward,new_state):
        self.state = state
        self.action = action
        self.reward = reward
        self.new_state = new_state
        self.priority = 0.01
    
    def to_tuple(self):
        return(self.state, self.action, self.reward, self.new_state)


class TransitionBatch:
    def __init__(self, transitions):
        self.transitions = transitions
        self.states, self.actions, self.rewards, self.new_states = tuple(np.array(i) for i in zip(*(t.to_tuple() for t in transitions)))

    def update_priorities(self,priorities):
        for i in range(len(priorities)):
            self.transitions[i].priority = priorities[i]

class max_size_list:
    def __init__(self, max_size = 100000):
        self.list = []
        self.max_size = max_size
        self.reached = False
        self.index = 0

    def put(self, item):
        if self.reached:
            self.list[self.index] = item
            self.index +=1
            if self.index >= self.max_size:
                self.index = 0
        elif len(self.list) >= self.max_size:
            self.reached = True
            self.put(item)
        else:
            self.list.append(item)


class Memory:
    def __init__(self, batch_size = 5, max_size = 100000, prioritised = True):
        self.items = max_size_list(max_size)
        self.batch_size = batch_size
        self.max_size = max_size
        self.prioritised = prioritised
        
    def put(self, transition):
        self.items.put(transition)
        
    def get_batch(self):
        probs = np.array([ t.priority for t in self.items.list])
        probs = probs / np.sum(probs)
        if self.prioritised:
            choices1 = rng.choice(len(self.items.list), self.batch_size, replace=True, p = probs)
            choices2 = rng.choice(len(self.items.list), self.batch_size)
            choices = np.append(choices1,choices2)
        else:
            choices = rng.choice(len(self.items.list), self.batch_size * 2)
        return TransitionBatch(np.array(self.items.list)[choices])

def build_actor_model(env, compact = False):
    layer_size = 128 if compact else 1024

    model = Sequential()
    model.add(Dense(layer_size, activation='relu', input_shape=(len(env.state[0]) + len(env.state[1]),)))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(env.action_space.shape[0], activation=action_activation))

    model.compile(optimizer=Adam(learning_rate=0.00001))
    return model

def build_critic_model(env, compact = False):
    layer_size = 128 if compact else 1024
    model = Sequential()
    model.add(Dense(layer_size, activation='relu', input_shape=(len(env.state[0]) + len(env.state[1]) + env.action_space.shape[0],)))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(layer_size, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='huber', optimizer=Adam(learning_rate=0.00001))
    return model

def get_flattened_state(observation):
    return np.concatenate((observation['state']['Q'], observation['state']['V'] ))

def action_gaussian_noise(scale, env):
    return [np.random.normal(scale=scale) for _ in range(env.action_space.shape[0])]

def action_activation(tensor_in):
    return clip_by_value(tanh(tensor_in) * 80, -80, 80)
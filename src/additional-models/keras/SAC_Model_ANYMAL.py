from Model_Base_Tim import Memory, Transition, TransitionBatch, build_actor_model, build_critic_model, action_activation
from tensorflow import clip_by_value, GradientTape, concat, reduce_mean, math
from tensorflow.keras import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Input
from tensorflow.keras.activations import tanh
from tensorflow.keras.losses import huber
from tensorflow.keras.optimizers import Adam
from timeit import default_timer as timer
import numpy as np
import tensorflow_probability
Normal = tensorflow_probability.distributions.Normal



def log_std_clip(tensor_in):
    return clip_by_value(tensor_in,-20,2)

class PolicyModel:
    def __init__(self, state_size, action_size, compact):
        layer_size = 128 if compact else 1024
        input_l = Input(shape = (state_size,))
        shared1 = Dense(layer_size, activation="relu", name='policy-s-dense1')(input_l)
        shared2 = Dense(layer_size, activation="relu", name='policy-s-dense2')(shared1)
        shared3 = Dense(layer_size, activation="relu", name='policy-s-dense3')(shared2)
        
        meanOut = Dense(action_size, name='policy-m-dense')(shared3)
        logStdOut = Dense(action_size, activation=log_std_clip, name='policy-l-dense')(shared3)

        self.model = Model(inputs = input_l, outputs = [meanOut, logStdOut])
        self.model.compile(optimizer=Adam(learning_rate=0.00001))

    def get_action(self, state):
        mean, logStd = self.model(np.array([state]))
        mean, logStd = mean[0], logStd[0]

        std = math.exp(logStd)
        action = action_activation(mean + std*np.random.normal(size=std.shape) )

        return action

    def evaluate(self, state, ep = 1e-6):
        mean, logStd = self.model(np.array(state))

        std = math.exp(logStd)
        action_base = mean + std*np.random.normal(size=std.shape)
        action = action_activation(action_base)

        log_prob = Normal(mean, std).log_prob(action_base) - np.log(1 - tanh(action_base) **2 + ep)

        return action, log_prob

    def save(self, filename):
        self.model.save(filename)


def build_policy_model(env, compact = False):
    return PolicyModel(len(env.state[0]) + len(env.state[1]), env.action_space.shape[0], compact)

def build_value_model(env, compact = False):
    layer_size = 128 if compact else 1024
    model = Sequential()
    model.add(Dense(layer_size, activation='relu', input_shape=(len(env.state[0]) + len(env.state[1]),), name='value-dense1'))
    model.add(Dense(layer_size, activation='relu', name='value-dense2'))
    model.add(Dense(layer_size, activation='relu', name='value-dense3'))
    model.add(Dense(1, activation="tanh", name='value-dense4'))

    model.compile(loss='huber', optimizer=Adam(learning_rate=0.00001))
    return model


def sac_learn(env, discount, no_episodes, learn_rate, alpha, s_l, prioritised = True, compact= False):

    def get_weight_updates(model, target_model):
        model_weights, target_weights = model.get_weights(), target_model.get_weights()
        return [learn_rate * weight + (1-learn_rate)* target_weights[i] for i, weight in enumerate(model_weights)]

    env.reset()
    memory = Memory(16 * s_l, prioritised=prioritised)
    value_model = build_value_model(env, compact)
    target_value_model = build_value_model(env, compact)
    critic_model_1 = build_critic_model(env,compact)
    critic_model_2 = build_critic_model(env, compact)
    policy_model = build_policy_model(env, compact)
    rewards = []
    i = 0
    while i < (no_episodes):
        start_time = timer()
        i += 1
        if (i% 100 == 0):
            print("saving backup")
            policy_model.save("sac_backup_R")
        env.reset()
        state = np.concatenate(env.state)
        done = False
        episode_length = 0
        ep_reward = 0
        t_group = []
        while not done and episode_length < 100000:
            episode_length += 1

            # run step and store results in memory
            if i <= s_l:
                action = env.action_space.sample()
            else:
                action = policy_model.get_action(state)
            try:
                _, reward, done, _ = env.step(action)
                ep_reward += reward
                next_state = np.concatenate(env.state)
            
                memory.put(Transition(state, action, reward, next_state))
                state = next_state
                
            except RuntimeError as err:
                print(err)
                done = True
                i -= 1
            
            if episode_length % s_l == 0 :
                # perform batch update
                t = memory.get_batch()

                target_value = target_value_model(t.new_states)
                y = t.rewards + discount * target_value

                # calculate priorities for future batches
                if prioritised:
                    c1 = critic_model_1(concat((t.states, t.actions),axis=1))
                    c2 = critic_model_1(concat((t.states, t.actions),axis=1))

                    loss_model_1 = huber(y, c1)
                    loss_model_2 = huber(y, c2)

                    t.update_priorities(loss_model_1 + loss_model_2)

                

                # update models
                critic_model_1.train_on_batch(concat((t.states, t.actions), axis=1), y)
                critic_model_2.train_on_batch(concat((t.states, t.actions), axis=1), y)

                with GradientTape() as g:
                    new_action, log_prob = policy_model.evaluate(t.states)

                    predicted_new_q = np.min([
                        critic_model_1(concat((t.states, new_action),axis=1)), 
                        critic_model_2(concat((t.states, new_action),axis=1))
                    ], axis=0)

                    loss = -reduce_mean(predicted_new_q - alpha * log_prob)
                gradients = g.gradient(loss, policy_model.model.trainable_variables)
                policy_model.model.optimizer.apply_gradients(
                    zip(gradients, policy_model.model.trainable_variables)
                )

                value_model.train_on_batch(t.states, predicted_new_q - alpha * log_prob)

                target_value_model.set_weights(get_weight_updates(value_model,target_value_model))
        rewards.append(ep_reward)

        end_time = timer()
        time_taken = end_time - start_time
        
        print("episode", i,"reward", ep_reward, "time", time_taken, "estimated time remaining", time_taken * (no_episodes - i))
        if not done:
            print("episode reached completion!")
    return policy_model, rewards

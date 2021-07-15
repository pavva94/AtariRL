import os
import numpy as np
import random
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from collections import deque
from datetime import datetime
import tensorflow as tf

class DQNAgent:
    def __init__(self, n_episode, greedy=False):

        # Initialize atributes
        # Stack four images preprecessed
        self.state_size = (200,160, 4)  # environment.observation_space.shape*4
        self.action_size = 7  # environment.action_space.n
        self.greedy = greedy

        #self.optimizer = Adam(learning_rate=0.000001)
        self.optimizer = RMSprop(learning_rate=0.00025,
                                       decay=0.95,
                                       momentum=0.0,
                                       epsilon=0.00001,
                                       centered=True)
        self.n_episode = n_episode
        self.batch_size = 32

        self.expirience_replay = deque(maxlen=5000)

        # Initialize discount and exploration rate
        self.gamma = 0.95
        self.epsilon_init = 0.1 if self.greedy else 0.5
        self.epsilon_final = 0.01
        self.epsilon = np.logspace(self.epsilon_init, self.epsilon_final, self.n_episode, endpoint=True)
        # esploriation term for UCB and counter for the actions
        # self.ucb_c = 2
        # self.counter_actions = np.ones(self.action_size)  # use ones and not zeros because the fraction in UCB

        # Build networks
        self.q_network = self._build_compile_model()
        self.target_network = self._build_compile_model()
        self.align_target_model()

        self.best_agent_reward = 0

    def store(self, state, action, reward, next_state, terminated):
        self.expirience_replay.append((state, action, reward, next_state, terminated))

    def _build_compile_model(self):
        model = Sequential()
        model.add(Conv2D(16, (8, 8), strides=(4, 4), padding='same', activation='relu', input_shape=self.state_size))
        model.add(Conv2D(32, (4, 4), strides=(2, 2), activation='relu'))
        #model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))

        # initializer = tf.keras.initializers.Constant(10.)
        # model.add(Dense(self.action_size, activation='linear', kernel_initializer=initializer))
        model.add(Dense(self.action_size, activation='linear'))

        # compile the model using traditional Machine Learning losses and optimizers
        model.compile(loss=tf.keras.losses.Huber(), optimizer=self.optimizer, metrics=['accuracy'])
        return model

    def align_target_model(self):
        print("Aligning models..")
        self.target_network.set_weights(self.q_network.get_weights())

    def save_model(self, episode, file_name=None):
        print("Saving weights...")
        if file_name is None:
            file_name = 'Assault_net_weights-' + str(episode) + '-' + str(self.best_agent_reward) + '-'+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.q_network.save_weights(file_name + '.h5')
        return file_name

    def load_model(self, filename):
        # load pre-trained model if exist
        if (os.path.isfile(filename + '.h5')):
            print("Loading previous weights: {}".format(filename))
            self.q_network.load_weights(filename + '.h5')
            self.align_target_model()

    def act(self, state, episode=None):
        if episode:
            # training mode with UCB or epsilon greedy decaying
            # Upper-Confidence-Bound policy
            #ucb_weights = self.ucb_c*np.sqrt(np.log(episode)/self.counter_actions)
            #return np.argmax(self.q_network.predict(state)[0] + ucb_weights)

            # epsilon greedy with epsilon decaying
            if np.random.rand() <= self.epsilon[episode]:
                # print("Epsilon")
                return random.randint(0, self.action_size-1)

        else:
            # if epsilon is None we are in play not training, so fixed epsilon-greedy
            if np.random.rand() <= self.epsilon_final:
                # print("Epsilon")
                return random.randint(0, self.action_size-1)
        # print("POLICY")
        return np.argmax(self.q_network.predict(state)[0])

    def train_on_single(self):
        if len(self.expirience_replay) < self.batch_size:
            return None

        print("Train on single..")
        minibatch = random.sample(self.expirience_replay, self.batch_size)

        # extract SARS tuples
        for state, action, reward, next_state, terminated in minibatch:
            # predict values
            target = self.q_network.predict(np.expand_dims(state, axis=0))

            if terminated:
                # if last state there's no future actions
                target[0][action] = reward
            else:
                # take value for next state
                t = self.target_network.predict(np.expand_dims(next_state, axis=0))
                # update the taget with the max q-value of next state
                # using a greedy policy
                target[0][action] = reward + self.gamma * np.max(t, axis=1)

            self.q_network.fit(np.expand_dims(state, axis=0), target, epochs=1, verbose=0)

    def get_arrays_from_batch(self, batch):
        try:
            states = np.array([x[0] for x in batch])
            actions = np.array([x[1] for x in batch])
            rewards = np.array([x[2] for x in batch])
            next_states = np.array([x[3] for x in batch])
            terminateds = np.array([x[4] for x in batch])
        except:
            states = x[0]
            actions = x[1]
            rewards = x[2]
            next_states = x[3]
            terminateds = x[4]

        return states, actions, rewards, next_states, terminateds

    def train_on_batch(self):
        if len(self.expirience_replay) < self.batch_size:
            return None

        #print("Train on batch..")
        minibatch = random.sample(self.expirience_replay, self.batch_size)
        # get the SARS tuple
        states, actions, rewards, next_states, terminateds = self.get_arrays_from_batch(minibatch)

        # predict values for next states using target net
        next_Q_values = self.target_network.predict(next_states)
        # takes the greedy actions
        max_next_Q_values = np.max(next_Q_values, axis=1)

        # calculate target values for training, not for the last state
        target_Q_values = (rewards + (1 - terminateds) * self.gamma * max_next_Q_values)
        target_Q_values = target_Q_values.reshape(-1, 1)

        mask = tf.one_hot(actions, self.action_size)

        with tf.GradientTape() as tape:
            # takes values predicted for current states
            all_Q_values = self.q_network(states)
            # takes values only for given actions
            Q_values = tf.reduce_sum(all_Q_values * mask, axis=1, keepdims=True)
            #loss = tf.reduce_mean(tf.keras.losses.Huber()(target_Q_values, Q_values))
            loss = tf.reduce_mean(tf.keras.losses.MeanSquaredError()(target_Q_values, Q_values))

        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

        return loss

    def play(self, environment, num_of_episodes, timesteps_per_episode, train=True):
        cont_e = 0
        total_reward_list = []
        total_actions_list = []
        for e in range(0, num_of_episodes):
            print("Episode {} start...".format(e))
            reward_list = []
            actions_list = []
            # Reset the environment
            state = environment.reset()

            state = image_preprocess_observations(state)
            #take the firsts 4 step as init
            states = np.stack([state] * 4, axis = 2)

            # Initialize variables
            episode_reward = 0
            terminated = False
            experience_replay_temp = []

            for timestep in range(timesteps_per_episode):
                # Run Action
                if self.greedy:
                  env_action = self.act(np.expand_dims(states, axis=0))
                else:
                  env_action = self.act(np.expand_dims(states, axis=0), e)

                # +1 on action counter for UCB
                # self.counter_actions[env_action] += 1

                # Take action
                next_state, reward, terminated, info = environment.step(env_action)
                next_state = image_preprocess_observations(next_state)
                next_states = np.append(states[:, :, 1: ], np.expand_dims(next_state, 2), axis = 2)

                self.store(states, env_action, reward, next_states, terminated)

                # train after some timestep, less become too much computational effort
                if cont_e > 100 and train:
                    #print("start timestamp for training: {}".format(str(timestep)))
                    #loss = self.train_on_single()
                    loss = self.train_on_batch()
                    #print("loss: " + str(loss))
                    cont_e = 0

                cont_e += 1
                states = next_states

                # save stats
                reward_list.append(reward)
                actions_list.append(env_action)
                episode_reward += reward
                total_actions_list.append(env_action)

                if terminated:
                    print("The rewards for the episode {} after {} timestep is {}".format(e, timestep, episode_reward))
                    total_reward_list.append(episode_reward)

                    # DQN training at the end of the episode
                    if train:
                        loss = self.train_on_batch()
                        print("Terminated loss: " + str(loss))
                        cont_e = 0

                    print("____________END__________________")
                    break

            if (e + 1) % 20 == 0:
                print("***************ALIGN-MODEL*******************")
                if train:
                    self.align_target_model()
                print("**********************************")

            if (e + 1) % 100 == 0:
                print("***************SAVE-WEIGHTS*******************")
                new_best_agent_reward = self_play(self.best_agent_reward, self, environment)
                if new_best_agent_reward > self.best_agent_reward:
                    self.best_agent_reward = new_best_agent_reward
                    self.save_model(e)
                #print("The score is " + str(new_best_agent_reward))
                print("**********************************")

        return total_reward_list, total_actions_list

import os
import numpy as np
import random
from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import ModelCheckpoint
from datetime import datetime
import tensorflow as tf
if tf.__version__ > "2.4.0":
    import tensorflow_probability as tfp


class Model(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        # model blocks
        self.conv_1 = Conv2D(16, (8, 8), strides=(4,4), padding='same', activation='relu', input_shape=state_size)
        self.conv_2 = Conv2D(32, (4, 4), strides=(2,2), activation='relu')
        self.dense = Dense(256, activation='relu')
        self.dense_v = Dense(128, activation='relu')
        self.dense_a = Dense(128, activation='relu')
        self.out_v = Dense(1, activation=None)
        self.out_a = Dense(self.action_size, activation='softmax')

    def call(self, input_data):
        x = self.conv_1(input_data)
        x = self.conv_2(x)
        x = Flatten()(x)
        x = self.dense(x)

        x_a = self.dense_a(x)
        a = self.out_a(x_a)

        x_v = self.dense_v(x)
        v = self.out_v(x_v)
        return v, a


class ACAgent():
    def __init__(self, gamma = 0.99):
        self.state_size = (200, 160, 4)
        self.action_size = 7
        self.gamma = gamma
        self.optimizer = Adam(learning_rate=1e-6)
        m = Model(self.state_size, self.action_size)
        m.compile(loss=tf.keras.losses.Huber(), optimizer=self.optimizer, metrics=['accuracy'])
        m.build((None, 200, 160, 4))
        self.model = m
        self.batch_size = 32

        self.best_agent_reward = 0

    def save_model(self, episode, file_name=None):
        print("Saving weights...")
        if file_name is None:
            file_name = 'AC_net_weights-' + str(episode) + '-'+ datetime.now().strftime("%Y%m%d-%H%M%S")
        self.model.save_weights(file_name + '.h5')
        return file_name

    def load_model(self, filename):
        # load model if exist
        if (os.path.isfile(filename + '.h5')):
            print("Loading previous weights: {}".format(filename))
            self.model.load_weights(filename + '.h5')

    def act(self, state, e=None):
        _, prob = self.model(state)
        prob = prob.numpy()

        # create a distribution to sample
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)

        # sample an action from that distribution
        action = dist.sample()
        return int(action.numpy()[0])

    def actor_loss_single(self, prob, action, td):
        # create a distribution
        dist = tfp.distributions.Categorical(probs=prob, dtype=tf.float32)
        log_prob = dist.log_prob(action)
        # error is td * ln_prob
        loss = -log_prob*td
        return loss

    def train_on_single(self, state, action, reward, next_state, done):
        # create batches of 1
        state = np.array([state])
        next_state = np.array([next_state])

        with tf.GradientTape() as tape:
            # value and action for the state St
            v, p =  self.model(state, training=True)
            # value and action for the state St+1
            vn, _ = self.model(next_state, training=True)
            # td error
            td = reward + self.gamma * vn * (1 - int(done)) - v

            a_loss = self.actor_loss_single(p, action, td)
            c_loss = td**2  # to reproduce a MSE with only one data
            total_loss = a_loss + 0.4 * c_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return total_loss

    def get_discounted_rewards(self, rewards):
        discounted_rewards = []
        sum_reward = 0
        rewards.reverse()
        for r in rewards:
            sum_reward = r + self.gamma * sum_reward
            discounted_rewards.append(sum_reward)

        return discounted_rewards.reverse()

    def actor_loss_on_batch(self, probs, actions, td):
        probability = []
        log_probability= []
        for pb,a in zip(probs,actions):
            dist = tfp.distributions.Categorical(probs=pb, dtype=tf.float32)
            log_prob = dist.log_prob(a)
            prob = dist.prob(a)
            probability.append(prob)
            log_probability.append(log_prob)

        p_loss= []
        e_loss = []
        td = td.numpy()

        for prob, t, log_prob in zip(probability, td, log_probability):
            # everything done with TF
            t =  tf.constant(t)
            policy_loss = tf.math.multiply(log_prob, t)
            entropy_loss = tf.math.negative(tf.math.multiply(prob, log_prob))
            p_loss.append(policy_loss)
            e_loss.append(entropy_loss)

        # tranform to tensor
        p_loss = tf.stack(p_loss)
        e_loss = tf.stack(e_loss)
        p_loss = tf.reduce_mean(p_loss)
        e_loss = tf.reduce_mean(e_loss)
        loss = -p_loss - 0.0001 * e_loss

        return loss

    def train_on_batch(self, states, actions, discounted_rewards):
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int32)
        discounted_rewards = np.array(discounted_rewards, dtype=np.float32)
        discounted_rewards = tf.reshape(discounted_rewards, (len(discounted_rewards),))
        print(states.shape)

        with tf.GradientTape() as tape:
            v, p = self.model(states, training=True)
            v = tf.reshape(v, (len(v),))
            td = tf.math.subtract(discounted_rewards, v)
            a_loss = self.actor_loss_on_batch(p, actions, td)
            c_loss = 0.5 * tf.keras.losses.mean_squared_error(discounted_rewards, v)
            total_loss = a_loss + 0.4 * c_loss

        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return a_loss, c_loss

    def play(self, environment, num_of_episodes, timesteps_per_episode, train=True):
      cont_e = 0
      total_reward_list = []
      total_actions_list = []
      for e in range(0, num_of_episodes):
          print("Episode {} start...".format(e))
          state_list = []
          reward_list = []
          actions_list = []
          # Reset the environment
          state = environment.reset()

          state = image_preprocess_observations(state)
          #take the firsts 4 step as init
          states = tf.cast(np.stack([state] * 4, axis = 2), tf.float32)

          # Initialize variables
          episode_reward = 0
          terminated = False

          experience_replay_temp = []
          timestep = -1
          #for timestep in range(timesteps_per_episode):
          while True:
              timestep += 1
              # Run Action
              env_action = self.act(np.expand_dims(states, axis=0))

              # Take action
              next_state, reward, terminated, info = environment.step(env_action)
              next_state = image_preprocess_observations(next_state)
              next_states = tf.cast(np.append(states[:, :, 1: ], np.expand_dims(next_state, 2), axis = 2), tf.float32)

              #if train:
                  #loss = self.train_on_single(states, env_action, reward, next_states, terminated)
                  #print("loss: " + str(loss))

              state_list.append(states)
              states = next_states
              reward_list.append(reward)
              actions_list.append(env_action)
              episode_reward += reward

              total_actions_list.append(env_action)

              if terminated:
                  print("The rewards for the episode {} after {} timestep is {}".format(e, timestep, episode_reward))
                  total_reward_list.append(episode_reward)

                  if train:
                    discounted_reward_list = self.get_discounted_rewards(reward_list)
                    loss = self.train_on_batch(state_list, actions_list, discounted_reward_list)

                  print("____________END__________________")
                  break

          if (e + 1) % 50 == 0:
              print("***************SAVE-WEIGHTS*******************")
              self.save_model(e)
              print("**********************************")

      return total_reward_list, total_actions_list

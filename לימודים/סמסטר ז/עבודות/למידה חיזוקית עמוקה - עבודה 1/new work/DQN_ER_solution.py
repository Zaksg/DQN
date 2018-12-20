import random
import gym
import math
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt

score_log = []
loss_log = []


class FitForwardBuilder:
    @staticmethod
    def build_3_layers(alpha, alpha_decay):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))
        return model

    @staticmethod
    def build_5_layers(alpha, alpha_decay):
        model = Sequential()
        model.add(Dense(100, input_dim=4, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(100, activation='relu'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=alpha, decay=alpha_decay))
        return model


class Memory:
    def __init__(self, max_size=100000):
        self._memory = deque(maxlen=max_size)

    def store_transition(self, transition):
        self._memory.append(transition)

    def get_sample(self, sample_size):
        return random.sample(self._memory, min(len(self._memory), sample_size))


class DQNSolverUsingER():
    def __init__(self, max_env_steps=None, gamma=0.95, batch_size=64, update_steps=65, layer_type=3):
      #default hyperparameters
      self.memory = Memory()
      self.cart_pole = gym.make('CartPole-v1')
      self.epsilon = 1.0
      self.epsilon_min = 0.01
      self.epsilon_decay = 0.995
      self.alpha = 0.01
      self.alpha_decay = 0.01
      self.n_episodes = 100000000
      self.steps_target = 475
      #passed hyperparameters
      self.cart_pole._max_episode_steps = max_env_steps
      self.gamma = gamma
      self.batch_size = batch_size
      self.update_steps = update_steps
		  #network structure type
      if (layer_type == 3):
        self.current_model = FitForwardBuilder.build_3_layers(self.alpha, self.alpha_decay)
        self.target_model = FitForwardBuilder.build_3_layers(self.alpha, self.alpha_decay)
      else:
        self.current_model = FitForwardBuilder.build_5_layers(self.alpha, self.alpha_decay)
        self.target_model = FitForwardBuilder.build_5_layers(self.alpha, self.alpha_decay)
      # graphic properties
      plt.rcParams['image.cmap'] = 'RdYlGn'
      plt.rcParams['figure.figsize'] = [15.0, 6.0]
      plt.rcParams['figure.dpi'] = 80
      plt.rcParams['savefig.dpi'] = 30

    def choose_action(self, state, epsilon):
        return self.cart_pole.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(
            self.current_model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        states, target_states = [], []
        minibatch = self.memory.get_sample(batch_size)
        for state, action, reward, next_state, done in minibatch:
            target_state = self.current_model.predict(state)
            target_state[0][action] = reward
            if not done:
                target_state[0][action] += self.gamma * np.max(self.target_model.predict(next_state)[0])

            states.append(state[0])
            target_states.append(target_state[0])

        history = self.current_model.fit(np.array(states), np.array(target_states), batch_size=len(states), verbose=0)
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def update_target_model(self):
        self.target_model.set_weights(self.current_model.get_weights())

    def print_graphs(self):
        plt.plot(range(len(score_log)), score_log[0:(len(score_log))], 'o')
        plt.title("Total reward per episode")
        plt.show()

        plt.plot(range(len(loss_log)), loss_log, 'o')
        plt.title("Total loss per episode")
        plt.show()

        avg_list = []
        for r in range(len(score_log) + 1):
            if r > 99:
                avg_list.append(np.mean(score_log[(r - 100): r]))
            else:
                avg_list.append(np.mean(score_log[(0): r]))

        plt.plot(range(len(avg_list)), avg_list, '-')
        plt.title("AVG reward per 100 episode")
        plt.show()

    def run(self):
        scores = deque(maxlen=100)
        steps = 0
        for episode in range(self.n_episodes):
            state = self.preprocess_state(self.cart_pole.reset())
            done = False
            total_reward = 0
            while not done:
                steps += 1
                if steps % self.update_steps == 0:
                    self.update_target_model()

                action = self.choose_action(state, self.get_epsilon(episode))
                next_state, reward, done, _ = self.cart_pole.step(action)
                next_state = self.preprocess_state(next_state)

                self.memory.store_transition((state, action, reward, next_state, done))

                loss = self.replay(self.batch_size)
                loss_log.append(loss)
                state = next_state
                total_reward += reward

            scores.append(total_reward)
            score_log.append(total_reward)
            mean_score = np.mean(scores)

            if mean_score >= self.steps_target and episode >= 100:
                print('Done after {} episodes'.format(episode))
                return episode - 100
            if episode % 10 == 0:
                print('[Episode {}] - Average reward over last 100 episodes was {}.'.format(episode, mean_score))

            if total_reward > 500:
                print('Episode: {} got score of {} to the current avg of {}'.format(episode, total_reward, mean_score))

            if episode % 1000 == 0 and episode > 0:
                self.print_graphs()

        return episode


if __name__ == '__main__':
    solver = DQNSolverUsingER(max_env_steps=10000, gamma=0.95, batch_size=64, update_steps=64, layer_type=3)
    solver.run()
    solver.print_graphs()
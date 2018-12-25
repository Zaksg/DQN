#PER
import random
import gym
import math
import numpy as np
from collections import deque
import heapq
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import itertools

score_log = []
loss_log = []
tiebreaker = itertools.count()
class DDQNPERSolver():
    def __init__(self, avg_target=475, gamma=1.0, batch_size=64):
        self.learning_rate = 0.01
        self.learning_rate_decay = 0.01
		self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
		self.memory = []
        self.env = gym.make('CartPole-v1')
        self.n_episodes = 100000000
        self.env._max_episode_steps = None
		self.avg_target = avg_target
        self.batch_size = batch_size
        self.gamma = gamma
        
        self.model=self.build_model(self.learning_rate, self.learning_rate_decay)
        self.target_model = self.build_model(self.learning_rate, self.learning_rate_decay)
        self.update_target_model()
        
        #graphic properties
        plt.rcParams['image.cmap'] = 'RdYlGn'
        plt.rcParams['figure.figsize'] = [15.0, 6.0]
        plt.rcParams['figure.dpi'] = 80
        plt.rcParams['savefig.dpi'] = 30

    def build_model(self, learning_rate, learning_rate_decay):
        model = Sequential()
        model.add(Dense(24, input_dim=4, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(48, activation='tanh'))
        model.add(Dense(2, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=learning_rate, decay=learning_rate_decay))
        return model
    
    def remember(self, state, action, reward, next_state, done):
        predicted_q_value = self.model.predict(next_state)[0]
        action_next_best = np.argmax(predicted_q_value)
        predicted_q_value_target = self.target_model.predict(next_state)[0]
        current_score = reward + self.gamma * np.max(self.target_model.predict(next_state)[0])
        td_error = abs(current_score - predicted_q_value)
        current_action = (td_error,state,action,reward, next_state, done)
        
        if len(self.memory) <= 10000000000000: 
            heapq.heappush(self.memory,(next(tiebreaker),current_action))
        else:
            self.memory[0] = current_action
        heapq.heapify(self.memory)

    def choose_action(self, state, epsilon):
        return self.env.action_space.sample() if (np.random.random() <= epsilon) else np.argmax(self.model.predict(state))

    def get_epsilon(self, t):
        return max(self.epsilon_min, min(self.epsilon, 1.0 - math.log10((t + 1) * self.epsilon_decay)))

    def preprocess_state(self, state):
        return np.reshape(state, [1, 4])

    def replay(self, batch_size):
        states, targets = [], []
        current_batch = heapq.nlargest(batch_size, self.memory)
        for _, current_action in current_batch:
            td, state, action, reward, next_state, done = current_action
            predicted_target = self.model.predict(state)
            if done:
              predicted_target[0][action] = reward
            else:
              predicted_target[0][action] = reward + self.gamma * np.max(self.target_model.predict(next_state)[0])
              
            states.append(state[0])
            targets.append(predicted_target[0])
        
        history = self.model.fit(np.array(states), np.array(targets), batch_size=len(states), verbose=0)
        loss = history.history['loss'][0] 
        if self.epsilon > self.epsilon_min:
            self.epsilon = self.epsilon * self.epsilon_decay
        return loss
    
    def update_target_model(self):
      self.target_model.set_weights(self.model.get_weights())
        
    def print_graphs(self):
      plt.plot(range(len(score_log)), score_log[0:(len(score_log))] , 'o')
      plt.title("Total reward per episode")
      plt.show()

      plt.plot(range(len(loss_log)), loss_log , 'o')
      plt.title("Total loss per episode")
      plt.show()

      avg_list = []
      for r in range(len(score_log)+1):
        if r > 99:
          avg_list.append(np.mean(score_log[(r - 100): r]))
        else:
          avg_list.append(np.mean(score_log[(0): r]))

      plt.plot(range(len(avg_list)), avg_list , '-')
      plt.title("AVG reward per 100 episode")
      plt.show()
                
    def solveProblem(self):
        scores = deque(maxlen=100)
        for episode in range(self.n_episodes):
            state = self.preprocess_state(self.env.reset())
            done = False
            total_reward = 0
            while not done and total_reward < 10000:
                action = self.choose_action(state, self.get_epsilon(episode))
                next_state, reward, done, _ = self.env.step(action)
                next_state = self.preprocess_state(next_state)
                self.remember(state, action, reward, next_state, done)
                state = next_state
                total_reward += 1

            scores.append(total_reward)
            score_log.append(total_reward)
            avg_score = np.mean(scores)
            if avg_score >= self.avg_target and episode >= 100:
                print('Solved after: {} episodes with AVG of {}.'.format(episode, avg_score))
                return episode
            if episode % 100 == 0:
                print('AVG score: {} after {} episodes.'.format(avg_score, episode))

            if total_reward > 500:
              print('Episode: {} got score of {} to the current avg of {}'.format(episode, total_reward, avg_score))
            
            self.target_model.set_weights(self.model.get_weights())
            loss = self.replay(self.batch_size)
            loss_log.append(loss)

        return e

if __name__ == '__main__':
    solver = DDQNPERSolver(avg_target=475, gamma=1.0, batch_size=64)
    solver.solveProblem()
    solver.print_graphs()
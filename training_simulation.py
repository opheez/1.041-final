import numpy as np
import random
import timeit
import torch
import torch.optim as optim
from torch.autograd import Variable



class Simulation:
    def __init__(self, Model, ReplayBuffer, gamma, max_steps, num_states, training_epochs, batch_size, learning_rate, init_bikes_per_dock, max_bikes_per_dock, reward_multiplier):
        self.dqn = Model
        self.gamma = gamma
        self.step = 0
        self.max_steps = max_steps
        self.num_states = num_states
        self.num_actions = self.num_states * max_bikes_per_dock * 2
        self.training_epochs = training_epochs
        self.replay_buffer = ReplayBuffer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=learning_rate)
        self.batch_size = batch_size
        self.state = np.full(self.num_states, init_bikes_per_dock)
        self.max_bikes_per_dock = max_bikes_per_dock
        # TODO: init last state of van to be 0
        self.reward_multiplier = reward_multiplier # TODO: separate punishment into empty and full

    def run(self, episode, epsilon):
        """
        Runs an episode of simulation, then starts a training session
        """
        start_time = timeit.default_timer()

        print("\t [INFO] Start simulating the episode")

        # inits
        self.step = 0
        old_undesirable_docks = 0
        old_state = -1
        old_action = -1

        sum_reward = 0
        sum_undesirable_docks = 0
        while self.step < self.max_steps:
            # get current state of all docks
            current_state = self.get_state()

            # calculate reward of previous action: reward is -10 for every dock at 0 or full bikes
            # change in # undesirable docks 
            num_empty = np.count_nonzero(current_state == 0)
            num_full = np.count_nonzero(current_state == self.max_bikes_per_dock)
            current_undesirable_docks = num_empty + num_full
            reward = (old_undesirable_docks - current_undesirable_docks) * self.reward_multiplier

            # saving the data into the memory
            if self.step != 0:
                self.replay_buffer.push(old_state, old_action, reward, current_state)

            # choose the light phase to activate, based on the current state of the intersection
            action = self.choose_action(current_state, epsilon) + 1
            # interpret the action and act
            dock = action//(self.max_bikes_per_dock * 2)
            num_bikes = action % (self.max_bikes_per_dock * 2) - self.max_bikes_per_dock
            self.state[dock] = self.state[dock] + num_bikes

            self.simulate()

            # saving variables for later & accumulate reward
            old_state = current_state
            old_action = action
            old_undesirable_docks = num_full + num_empty

            # saving only the meaningful reward to better see if the agent is behaving correctly
            if reward < 0:
                sum_reward += reward
            sum_undesirable_docks += num_full + num_empty

        avg_reward = sum_reward / self.max_steps
        avg_undesirable_docks = sum_undesirable_docks / self.max_steps
        simulation_time = round(timeit.default_timer() - start_time, 1)

        print("\t [STAT] Average reward:", avg_reward,
              "Average waiting time:", avg_undesirable_docks,
              "- Epsilon:", round(epsilon, 2))

        print("\t [INFO] Training the DQN")
        start_time = timeit.default_timer()
        # training the DQN

        sum_training_loss = 0
        for _ in range(self.training_epochs):
            sum_training_loss += self.compute_td_loss()
        avg_training_loss = sum_training_loss.item() / self.max_steps
        print("\t [STAT] Training Loss :", avg_training_loss)
        training_time = round(timeit.default_timer() - start_time, 1)

        return simulation_time, training_time, avg_reward, avg_undesirable_docks, avg_training_loss

    def simulate(self):
        """
        TODO: simulate transition with external data
        [d1, d2, d3]
        for each station, randomly generate from normal distribution of bikes taken/added to it
        """
        for state in range(self.num_states-1):
            delta = int(np.random.normal(0, self.max_bikes_per_dock/10))
            self.state[state] = self.state[state] + delta


    def choose_action(self, state, epsilon):
        """
        According to epsilon-greedy policy, decide whether to perform exploration or exploitation
        """
        if random.random() < epsilon:
            # random action
            return random.randint(0, self.num_actions - 1)
        else:
            # the best action given the current state
            state = Variable(torch.FloatTensor(state).unsqueeze(0), requires_grad=False)
            q_value = self.dqn.forward(state)

            # apply mask
            mask = np.zeros(self.num_actions)
            for i in range(self.num_states-1): # don't act on the van
                for j in range(self.max_bikes_per_dock): # negative
                    if j + 1 <= min(self.state[i], max(0,self.max_bikes_per_dock - self.state[-1])): # min of bikes at dock and room in van
                        mask[(2 * self.max_bikes_per_dock * i) + j] = 1 
                for j in range(self.max_bikes_per_dock): # positive
                    if j + 1 <= min(max(0,self.max_bikes_per_dock - self.state[i]), self.state[-1]): # min of room at docks and bikes in van
                        mask[(2 * self.max_bikes_per_dock * i) + self.max_bikes_per_dock + j] = 1 
            mask = Variable(torch.BoolTensor(mask).unsqueeze(0), requires_grad=False)

            q_value = q_value * mask

            # print(q_value.size())
            return q_value.max(1)[1].data[0]

    def get_state(self):
        """
        Retrieve the state of the number of bikes at each dock and in the van
        """

        return self.state

    def compute_td_loss(self):

        """
        Compute the temporal difference loss as defined by the Q update
        """

        # sample a batch from the replay buffer
        state, action, reward, next_state = self.replay_buffer.sample(self.batch_size)

        # convert to pytorch variables
        state = Variable(torch.FloatTensor(np.float32(state)), requires_grad=False)
        next_state = Variable(torch.FloatTensor(np.float32(next_state)), requires_grad=False)
        action = Variable(torch.LongTensor(action), requires_grad=False)
        reward = Variable(torch.FloatTensor(reward), requires_grad=False)

        # obtain the q value for the current state by feeding the state to the DQN
        q_values = self.dqn.forward(state)
        # obtain the q value for the next state by feeding the next state to the DQN
        next_q_values = self.dqn.forward(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        # TODO : Calculate the loss
        # set the loss variable to the right expression using the computed quantities above

        loss = (((reward + self.gamma * next_q_value) - q_value) ** 2).mean()

        # next we do the gradient update based on the calculated loss
        self.optimizer.zero_grad()
        loss.backward() # PyTorch will automatically do the gradient update on the defined loss 
        self.optimizer.step()

        return loss

    def save_model(self, path, episode):
        """
        Save the DQN model
        """
        torch.save(self.dqn, path + "/" + str(episode))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class Environment:
    def __init__(self, data, factor_names:list, history_t=90):
        self.data = data
        self.factor_names = factor_names
        self.history_t = history_t

        self.day = 0
        self.done = False
        self.state = [0] + self.data.loc[self.day, ['close'] + self.factor_names].tolist()
        self.state_ = None
        self.reward = 0
        self.return_list = [0]

    def reset(self):
        self.state = [0] + self.data.loc[0, :][['close'] + self.factor_names].tolist()
        self.day = 0
        self.return_list = [0]
        self.reward = 0
        self.done = False
        return self.state[2:]


    def step(self, act):
        self.done = (self.day >= len(self.data) - 1)

        def _buy_and_sell(act):
            if self.state[0] > 0:
                if act == 0:
                    self.state[0] = 0
            else:
                if act == 1:
                    self.state[0] = 1

        def _update_state(future_date = 9):
            state = [self.state[0]] + self.data.loc[self.day+future_date-1, ['close'] + self.factor_names].tolist()
            return state

        _buy_and_sell(act)
        self.day += 1
        self.state_ = _update_state(future_date=5)

        if self.state[0] > 0:
            if act == 0:
                self.reward = 1 - self.state_[1] / self.state[1]
            elif act == 1:
                self.reward = self.state_[1] / self.state[1] - 1
        else:
            if act == 1:
                self.reward = self.state_[1] / self.state[1] - 1
            elif act == 0:
                self.reward = 1 - self.state_[1] / self.state[1]

        close_record = self.state[1]
        self.state = _update_state(future_date=1)  #状态空间不太对只能用过去的数据
        if self.state[0] > 0:
            self.return_list.append(self.state[1] / close_record - 1)
        else:
            self.return_list.append(0)

        if self.done:
            df = pd.DataFrame(self.data.trading_day, columns=['trading_day'])
            df['daily_return'] = self.return_list
            df['cum_ret'] = df['daily_return'].cumsum()

        return self.state[2:], self.reward, self.done, self.state_[2:]



class QNetwork(nn.Module):
    def __init__(self, n_states=10, n_acts=2):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(n_states, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.bn2 = nn.BatchNorm1d(256)
        self.out = nn.Linear(256, n_acts)
        self.out.weight.data.normal_(0, 0.1)
        self.softmax_out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.out(x)
        act_value = self.softmax_out(x)
        return act_value



class DQN:
    def __init__(self, seed=0, memory_size=32, batch_size=16, n_states=10):
        self.eval_net, self.target_net = QNetwork(), QNetwork()
        self.eval_net.train()
        self.target_net.eval()
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=1e-3, betas=(0.9, 0.999))
        self.loss_func = nn.SmoothL1Loss()
        self.loss = 0
        self.epsilon = 0
        self.learn_step_counter = 0

        self.memory = []
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_states = n_states

    def select_action(self, x, epsilon_end=0.05, epsilon_start=0.9, epsilon_decay=500):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        self.epsilon = epsilon_end + (epsilon_start - epsilon_end) * np.exp(-self.learn_step_counter/epsilon_decay)
        if np.random.uniform() < self.epsilon:
            self.eval_net.eval()
            actions_value = self.eval_net.forward(x)
            action = torch.argmax(actions_value, 1).data.numpy()
            action = action[0]
        else:
            action = np.random.randint(2)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        self.memory.append(transition)
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)

    def learn(self):
        self.eval_net.train()
        if self.learn_step_counter % 252 == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(self.memory_size, self.batch_size)
        batch = np.array(self.memory[i] for i in sample_index)

        b_s = torch.FloatTensor(batch[:, :self.n_states])
        b_a = torch.LongTensor(batch[:, self.n_states:self.n_states+1].astype(int))
        b_r = torch.FloatTensor(batch[:, self.n_states+1:self.n_states+2])
        b_s_ = torch.FloatTensor(batch[:, -self.n_states:])

        q_eval = self.eval_net(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + 0.5 * q_next.max(1)[0].view(self.batch_size, 1)
        self.loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()


def plot(env, save_place='DQN.png'):
    plot_df = pd.DataFrame(env.date_memory, columns=['date'])
    plot_df['return'] = env.return_list
    plot_df['close'] = env.data['close']
    plot_df['close'] = plot_df['close'].shift(-1) / plot_df['close'] - 1
    plot_df['return'] = plot_df['return'].shift(-1)
    plot_df.set_index('date', inplace=True)
    plot_df = plot_df.cumsum()
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111)
    ax.plot(plot_df.index, plot_df['close'], label='origin')
    ax.plot(plot_df.index, plot_df['return'], label='strategy')
    ax.plot(plot_df.index, plot_df['return'] - plot_df['close'], label='excess return')
    plt.legend()
    plt.grid(True)
    plt.show()




def train_DQN(env: Environment, seed=0):
    dqn = DQN(seed=seed)
    print('Collecting experience...')
    for i_episode in range(10):
        s = env.reset()
        ep_r = 0
        while True:
            a = dqn.select_action(s)
            s_, r, done, info = env.step(a)
            dqn.store_transition(s, a, r, info)
            ep_r += r
            if len(dqn.memory) > 32:
                dqn.learn()
                if done:
                    print('Ep:', i_episode+1, '| Ep_reward: ', round(ep_r, 2))
                    print('=====================================')
            if done:
                break
            s = s_
        if i_episode % 5 == 4:
            pass
    return dqn


def test_DQN(env_config, test_data:pd.DataFrame, dqn: DQN):
    print('start predicting..')
    env = Environment(test_data, **env_config)
    s = env.reset()
    ep_r = 0
    while True:
        a = dqn.select_action(s)
        s_, r, done, info = env.step(a)
        ep_r += r
        if done:
            break
        s = s_
    print(f'reward: {ep_r}')


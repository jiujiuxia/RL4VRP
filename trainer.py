
import os
import argparse
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from model import DRL4TSP, Encoder
from tasks import vrp_init
from tasks.vrp_init import VehicleRoutingDataset
import copy
from scipy.stats import ttest_rel
from tasks.vrp_init import reward_num

device = torch.device("cuda:0")
print(device)


class StateCritic(nn.Module):
    """Estimates the problem complexity.

    This is a basic module that just looks at the log-probabilities predicted by
    the encoder + decoder, and returns an estimate of complexity
    """

    def __init__(self, static_size, dynamic_size, hidden_size):
        super(StateCritic, self).__init__()

        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder = Encoder(dynamic_size, hidden_size)
        self.project_query = nn.Linear(hidden_size, hidden_size)

        # Define the encoder & decoder models
        self.e = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)
        self.project_d = nn.Conv1d(hidden_size, hidden_size, kernel_size=1)  # conv1d_1

        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        self.W = nn.Parameter(torch.zeros((1, hidden_size),
                                          device=device, requires_grad=True))

        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic):

        static_hidden = self.static_encoder(static)
        dynamic_hidden = self.dynamic_encoder(dynamic[:, 1:2, :])
        batch_size, hidden_size, _ = static_hidden.size()

        last_hh = torch.zeros((dynamic_hidden.size()[0], dynamic_hidden.size()[1]), device=device,requires_grad= True)
        # last_hh = last_hh.squeeze(0)
        for x in range(3):
            last_hh = self.project_query(last_hh)
            last_hh = last_hh.unsqueeze(2).expand_as(static_hidden)  # (batch * dim *point)

            static_hidden = self.e(static_hidden)            #batch*dim*point
            dynamic_hidden = self.project_d(dynamic_hidden)   #batch*dim*point

            # hidden = torch.cat((static_hidden, dynamic_hidden,last_hh), 1)
            hidden = static_hidden + dynamic_hidden + last_hh

            W = self.W.expand(batch_size, 1, hidden_size)
            # print(W,'w')
            attns = torch.bmm(W, torch.tanh(hidden))         #batch*1*dim x batch*dim*point -->batch*1*point

            attns = F.softmax(attns, dim=2)  # (batch * 1 * point)

            attns = attns.squeeze(1)    # batch * point
            # batch * hidden * point  x  batch * point * 1   --> batch * hidden * 1
            last_hh = torch.bmm(static_hidden,attns.unsqueeze(2)).squeeze(2)

        output = F.relu(self.fc1(last_hh))
        output = torch.squeeze(self.fc3(output),1)

        return output

def checkout(old_valid, new_valid ):

    if old_valid == None:
        return new_valid,True
    new_mean = np.mean(new_valid)
    old_mean = np.mean(old_valid)
    note = False
    valid_result = old_valid


    print("candidate mean {}, baseline mean {}, difference {}".format(
        new_mean, old_mean, new_mean - old_mean))
    if new_mean - old_mean < 0:
        # Calc p value
        t, p = ttest_rel(new_valid, old_valid)

        p_val = p / 2  # one-sided
        assert t < 0, "T-statistic should be negative"
        print("p-value: {}".format(p_val))
        if p_val < 0.2:
            print('Update critic')
            note = True
            valid_result = new_valid
    return valid_result, note

def validate_BS(data_loader, actor, reward_fn):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()
    rewards = []

    beam_width = 10
    for batch_idx, batch in enumerate(data_loader):

        static, dynamic,= batch


        static = static.to(device).requires_grad_(False)
        dynamic = dynamic.to(device).requires_grad_(False)

        # Full forward pass through the dataset
        with torch.no_grad():

            tour_indices, _, _ = actor.forward(static, dynamic,is_beam_search= True)

            #print(tour_indices[0])
            reward = reward_fn(static.repeat(beam_width, 1, 1)[:,0:2,:], tour_indices).detach()
            R = np.concatenate(np.split(np.expand_dims(reward.cpu().numpy(), 1), beam_width, axis=0), 1)
            R = np.amin(R, 1, keepdims=False)
            rewards.append(np.mean(R))

    actor.train()

    return rewards

def validate(data_loader, actor, reward_fn):
    """Used to monitor progress on a validation set & optionally plot solution."""
    actor.eval()
    rewards = []

    for batch_idx, batch in enumerate(data_loader):

        static, dynamic,= batch

        static = static.to(device).requires_grad_(False)
        dynamic = dynamic.to(device).requires_grad_(False)

        # Full forward pass through the dataset
        with torch.no_grad():

            tour_indices, _, _ = actor.forward(static, dynamic)

            reward = reward_fn(static[:,0:2,:], tour_indices).detach()

            rewards.append(torch.mean(reward))

    actor.train()

    return rewards

def train(actor, critic, new_critic, task, num_nodes, valid_data, reward_fn,
          render_fn, batch_size, actor_lr, critic_lr, max_grad_norm,
          checkpoint_every, **kwargs):
    """Constructs the main actor & critic networks, and performs all training."""
    import time
    now = '%s' % datetime.datetime.now().time()
    now = now.replace(':', '_')
    save_dir = os.path.join(task, '%d' % num_nodes, now)
    st = time.time()
    final = pd.DataFrame()
    checkpoint_dir = os.path.join(save_dir, 'checkpoints')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if kwargs['optimizer'] == 'sgd':
        optimizer = optim.SGD
    elif kwargs['optimizer'] == 'adam':
        optimizer = optim.Adam
    elif kwargs['optimizer'] == 'adagrad':
        optimizer = optim.Adagrad
    else:
        raise ValueError('Optimizer <%s> not understood'%kwargs['optimizer'])

    actor_optim = optimizer(actor.parameters(), lr=actor_lr)
    critic_optim = optimizer(critic.parameters(), lr=critic_lr)


    valid_loader = DataLoader(valid_data, batch_size, False, num_workers=0)

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9

    max_load = LOAD_DICT[args['num_nodes']]
    actor.train()
    critic.train()

    origin_valid = None
    new_critic.load_state_dict(actor.state_dict())


    for epoch in range(200):
        print('epoch',epoch)

        num_samples = 512000
        train_data = VehicleRoutingDataset(True,
                                           num_samples,
                                           args['batch_size'],
                                           args['num_nodes'],
                                           max_load,
                                           MAX_DEMAND,
                                           city_time=4.6,
                                           max_delta_theta=0.2,
                                           max_service_time=0.2,
                                           )

        train_loader = DataLoader(train_data, batch_size, True, num_workers=0)
        losses, rewards, critic_rewards = [], [], []


        if epoch <= 10:

            for batch_idx, batch in enumerate(train_loader):

                static, dynamic = batch

                static = static.to(device).requires_grad_(False)
                dynamic = dynamic.to(device).requires_grad_(False)

                # Full forward pass through the dataset
                tour_indices, tour_logp, _ = actor(static, dynamic)
                # Sum the log probabilities for each city in the tour
                reward = reward_fn(static[:,0:2,:], tour_indices)

                critic_est = critic(static, dynamic).view(-1)

                advantage = (reward - critic_est)
                actor_loss = torch.mean((advantage.detach() * tour_logp.sum(dim=1)), 0)
                critic_loss = torch.mean(torch.pow(advantage, 2))

                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                critic_optim.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), max_grad_norm)
                critic_optim.step()

                critic_rewards.append(torch.mean(critic_est.detach().data))
                rewards.append(torch.mean(reward.detach().data))
                losses.append(torch.mean(actor_loss.detach().data))

                if (batch_idx + 1) % checkpoint_every == 0:
                    ed = time.time()
                    td = ed - st
                    print('time', td)
                    st = ed

                    mean_loss = np.mean(losses[-checkpoint_every:])
                    mean_reward = np.mean(rewards[-checkpoint_every:])
                    mean_critic = np.mean(critic_rewards[-checkpoint_every:])

                    prefix = 'epoch%d_batch%d_%2.4f' % (epoch, batch_idx, mean_reward)
                    save_path = os.path.join(checkpoint_dir, prefix + '_actor.pt')
                    torch.save(actor.state_dict(), save_path)

                    save_path = os.path.join(checkpoint_dir, prefix + '_critic.pt')
                    torch.save(critic.state_dict(), save_path)

                    #valid_BS = validate_BS(valid_loader, actor, reward_fn)
                    valid = validate(valid_loader, actor, reward_fn)

                    print('%d/%d,Mean epoch loss/reward: %2.4f, %2.4f, %2.4f, %2.4f' % \
                          (batch_idx, len(train_loader), mean_loss, np.mean(valid),
                           mean_reward, mean_critic))
                    each = [batch_idx, np.mean(valid), mean_reward, mean_critic]
                    final = final.append([each])
            origin_valid = valid
            new_critic.load_state_dict(actor.state_dict())

        else:

            for batch_idx, batch in enumerate(train_loader):

                static, dynamic = batch

                static = static.to(device).requires_grad_(False)
                dynamic = dynamic.to(device).requires_grad_(False)

                # Full forward pass through the dataset
                tour_indices, tour_logp, _ = actor(static, dynamic)
                # Sum the log probabilities for each city in the tour
                reward = reward_fn(static[:,0:2,:], tour_indices)

                new_critic.eval()
                # Full forward pass through the dataset
                with torch.no_grad():
                    critic_tour, _, _ = new_critic.forward(static, dynamic)

                    critic_est = reward_fn(static[:,0:2,:], critic_tour).detach()

                advantage = (reward - critic_est)
                actor_loss = torch.mean((advantage.detach() * tour_logp.sum(dim=1)), 0)

                actor_optim.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), max_grad_norm)
                actor_optim.step()

                critic_rewards.append(torch.mean(critic_est.detach().data))
                rewards.append(torch.mean(reward.detach().data))
                losses.append(torch.mean(actor_loss.detach().data))

                if (batch_idx + 1) % checkpoint_every == 0:
                    ed = time.time()
                    td = ed - st
                    print('time', td)
                    st = ed

                    mean_loss = np.mean(losses[-checkpoint_every:])
                    mean_reward = np.mean(rewards[-checkpoint_every:])
                    mean_critic = np.mean(critic_rewards[-checkpoint_every:])

                    prefix = 'epoch%d_batch%d_%2.4f' % (epoch, batch_idx, mean_reward)
                    save_path = os.path.join(checkpoint_dir, prefix + '_actor.pt')
                    torch.save(actor.state_dict(), save_path)

                    # save_path = os.path.join(checkpoint_dir, prefix + '_critic.pt')
                    # torch.save(critic.state_dict(), save_path)

                    # valid_BS = validate_BS(valid_loader, actor, reward_fn)
                    valid = validate(valid_loader, actor, reward_fn)

                    print('%d/%d,Mean epoch loss/reward: %2.4f, %2.4f, %2.4f,  %2.4f' % \
                          (batch_idx, len(train_loader), mean_loss, np.mean(valid),
                           mean_reward, mean_critic))
                    each = [batch_idx, np.mean(valid), mean_reward, mean_critic]
                    final = final.append([each])
            origin_valid, note = checkout(origin_valid,valid)
            if note:
                new_critic.load_state_dict(actor.state_dict())

    final.to_excel('RL+newcritic+20_CVRPTW.xlsx')
    # mark.to_excel('mark_cheng.xlsx')

def train_vrp(args):

    # Determines the maximum amount of load for a vehicle based on num nodes
    LOAD_DICT = {10: 20, 20: 30, 50: 40, 100: 50}
    MAX_DEMAND = 9
    STATIC_SIZE = 5 # (x, y)
    DYNAMIC_SIZE = 1 # (load, demand, time)

    max_load = LOAD_DICT[args['num_nodes']]

    valid_data = VehicleRoutingDataset(False,
                                       args['valid_size'],
                                       args['batch_size'],
                                       args['num_nodes'],
                                       max_load,
                                       MAX_DEMAND,
                                       city_time=4.6,
                                       max_delta_theta=0.2,
                                       max_service_time=0.2,
                                       seed = args['seed'],
                                       is_beam_search = True)

    args['valid_data'] = valid_data
    args['reward_fn'] = vrp_init.reward
    args['render_fn'] = vrp_init.render

    actor = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args['hidden_size'],
                    valid_data.update_dynamic,
                    valid_data.update_mask,
                    args['num_layers'],
                    args['dropout']).to(device)
    new_critic = DRL4TSP(STATIC_SIZE,
                    DYNAMIC_SIZE,
                    args['hidden_size'],
                    valid_data.update_dynamic,
                    valid_data.update_mask,
                    args['num_layers'],
                    args['dropout']).to(device)
    critic = StateCritic(STATIC_SIZE, DYNAMIC_SIZE, args['hidden_size']).to(device)
    actor = actor.to(device)
    new_critic = new_critic.to(device)

    # path = 'vrp/50/12_41_30.143186/checkpoints/epoch197_batch799_18.4053_'
    # # path = 'vrp10_'
    # params = torch.load(path + 'actor.pt', map_location=lambda storage, loc: storage)
    # actor.load_state_dict(params)
    '''
    params = torch.load(path + 'critic.pt', map_location=lambda storage, loc: storage)
    critic.load_state_dict(params)
    '''

    train(actor, critic, new_critic, **args)

if __name__ == '__main__':
    import time

    start = time.time()
    parser = argparse.ArgumentParser(description='Combinatorial Optimization')
    parser.add_argument('--seed', dest='seed', default=113, type=int)
    parser.add_argument('--task', dest='task', default='vrp')
    parser.add_argument('--nodes', dest='num_nodes', default=20, type=int)
    parser.add_argument('--actor_lr', dest='actor_lr', default=1e-4, type=float)
    parser.add_argument('--critic_lr', dest='critic_lr', default=1e-4,
                        type=float)
    parser.add_argument('--max_grad_norm', dest='max_grad_norm', default=2.,
                        type=float)
    parser.add_argument('--checkpoint', dest='checkpoint_every', default=200,
                        type=int)
    parser.add_argument('--batch_size', dest='batch_size', default=512, type=int)
    parser.add_argument('--hidden', dest='hidden_size', default=128, type=int)
    parser.add_argument('--dropout', dest='dropout', default=0.1, type=float)
    parser.add_argument('--layers', dest='num_layers', default=1, type=int)
    parser.add_argument('--optimizer', dest='optimizer', default='adam')
    parser.add_argument('--train_size', dest='train_size', default=128000,
                        type=int)
    parser.add_argument('--valid_size', dest='valid_size', default=1280,
                        type=int)

    args = vars(parser.parse_args())

    if args['task'] == 'vrp':
        train_vrp(args)
    else:
        raise ValueError('Task <%s> not understood'%args['task'])
    end = time.time()
    print('time',end-start)

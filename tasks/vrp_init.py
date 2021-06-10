"""Defines the main task for the VRP.

"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable
import time as systemtime

device = torch.device("cuda:0")


class VehicleRoutingDataset(Dataset):
    def __init__(self, mode, num_samples, batch_size, num_node, max_load=20, max_demand=9,
                 city_time=4.6, max_delta_theta=0.2, max_service_time=0.2, map=1, speed=1,
                 seed=None,is_beam_search = False):
        super(VehicleRoutingDataset, self).__init__()

        if max_load < max_demand:
            raise ValueError(':param max_load: must be > max_demand')

        if seed is None:
            seed = np.random.randint(1234567890)
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.num_node = num_node  ####num_node
        self.batch_size = batch_size
        self.num_samples = num_samples
        self.max_load = max_load
        self.max_demand = max_demand

        self.city_time = city_time
        self.max_delta_theta = max_delta_theta
        self.max_service_time = max_service_time
        self.beam_width = 10
        self.is_beam_search = is_beam_search

        fname = 'vrp' + str(self.num_node) + '.txt'
        if mode == False:
            if os.path.exists(fname):
                print('Loading dataset for {}...'.format(fname))
                data = np.loadtxt(fname, delimiter=' ')
                data = data.reshape(-1, self.num_node + 1, 9)
                data = data.transpose(0, 2, 1)
                self.static = torch.from_numpy(data[:, 0:5, :]).type(torch.FloatTensor).to(device)
                self.dynamic = torch.from_numpy(data[:, 5:9, :]).type(torch.FloatTensor).to(device)
            else:
                '''
                    static:
                    x / y / start_time / due_time / service_time
                    batch*5*point
                '''

                # Depot location will be the first node in each
                locations = torch.rand((self.num_samples, 2, self.num_node + 1))

                d = torch.sqrt((locations[:, 0, :] - locations[:, 0, 0].repeat(self.num_node + 1, 1).t()) ** 2 +
                               (locations[:, 1, :] - locations[:, 1, 0].repeat(self.num_node + 1, 1).t()) ** 2)

                # rand (0.15~0.2)
                delta_time = torch.rand((self.num_samples, 1, self.num_node + 1)) * 0.05 + 0.15
                delta_time[:, :, 0] = self.city_time
                service_time = torch.rand((self.num_samples, 1, self.num_node + 1)) * 0.05 + 0.15
                service_time[:, :, 0] = 0
                scope = (torch.ones(self.num_samples, self.num_node + 1) * (self.city_time) - service_time[:, 0,
                                                                                              :] - delta_time[:, 0,
                                                                                                   :]) / d - 1

                start_time = (torch.rand(self.num_samples, self.num_node + 1) * (scope - 1) + 1) * d
                start_time = torch.unsqueeze(start_time, 1)
                start_time[:, :, 0] = 0
                due_time = start_time + delta_time

                self.static = torch.cat((locations, start_time, due_time, service_time), 1)

                '''
                    dynamic
                    demand / load / time / position
                    batch*4*point
                '''
                demand = torch.randint(1, 9, (self.num_samples, 1, self.num_node + 1))
                demand[:, :, 0] = 0
                load = torch.ones((self.num_samples, 1, self.num_node + 1)) * max_load
                time = torch.zeros((self.num_samples, 1, self.num_node + 1))
                position = torch.zeros((self.num_samples, 1, self.num_node + 1))

                self.dynamic = torch.cat((demand, load, time, position), 1)
                # self.mask = torch.ones((self.batch_size, self.num_node + 1))

                save_data = torch.cat([self.static, self.dynamic], 1).numpy()
                save_data = save_data.transpose(0, 2, 1)
                np.savetxt(fname, save_data.reshape(-1, (self.num_node + 1) * 9))
                self.dynamic = self.dynamic.to(device)
                self.static = self.static.to(device)
        else:
            '''
                static:
                x / y / start_time / due_time / service_time
                batch*5*point
            '''

            # Depot location will be the first node in each
            locations = torch.rand((self.num_samples, 2, self.num_node + 1))

            d = torch.sqrt((locations[:, 0, :] - locations[:, 0, 0].repeat(self.num_node + 1, 1).t()) ** 2 +
                           (locations[:, 1, :] - locations[:, 1, 0].repeat(self.num_node + 1, 1).t()) ** 2)

            # rand (0.15~0.2)
            delta_time = torch.rand((self.num_samples, 1, self.num_node + 1)) * 0.05 + 0.15
            delta_time[:, :, 0] = self.city_time
            service_time = torch.rand((self.num_samples, 1, self.num_node + 1)) * 0.05 + 0.15
            service_time[:, :, 0] = 0
            scope = (torch.ones(self.num_samples, self.num_node + 1) * (self.city_time) - service_time[:, 0,
                                                                                          :] - delta_time[:, 0,
                                                                                               :]) / d - 1

            start_time = (torch.rand(self.num_samples, self.num_node + 1) * (scope - 1) + 1) * d
            start_time = torch.unsqueeze(start_time, 1)
            start_time[:, :, 0] = 0
            due_time = start_time + delta_time

            self.static = torch.cat((locations, start_time, due_time, service_time), 1)

            '''
                dynamic
                demand / load / time / position
                batch*4*point
            '''
            demand = torch.randint(1, 9, (self.num_samples, 1, self.num_node + 1))
            demand[:, :, 0] = 0
            load = torch.ones((self.num_samples, 1, self.num_node + 1)) * max_load
            time = torch.zeros((self.num_samples, 1, self.num_node + 1))
            position = torch.zeros((self.num_samples, 1, self.num_node + 1))

            self.dynamic = torch.cat((demand, load, time, position), 1)
            # self.mask = torch.ones((self.batch_size, self.num_node + 1))
            # print(self.static, self.dynamic)
            self.dynamic = self.dynamic.to(device)
            self.static = self.static.to(device)
            #print('tranin 01', self.static[0,0,:])

    #
    # def generate_data(self):
    #     return  self.static,self.dynamic

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # (static, dynamic, start_loc)
        # return (self.static[idx], self.dynamic[idx], self.static[idx, :, 0:1])
        return (self.static[idx], self.dynamic[idx])

    def update_mask(self, mask, static, dynamic, chosen_idx):
        """
            batch*point
        """
        # Convert floating point to integers for calculations

        batch_size = dynamic.shape[0]
        loads = dynamic.data[:, 1]  # (batch_size, seq_len)
        demands = dynamic.data[:, 0]  # (batch_size, seq_len)

        # If there is no positive demand left, we can end the tour.
        # Note that the first node is the depot, which always has a negative demand
        if demands.eq(0).all():
            return demands * 0.

        # TW
        locations = static[:, :2, :]
        all_position = dynamic[:, 3].clone()

        all_position_onehot = all_position[:, 0].type(torch.LongTensor).reshape(
            [1, batch_size]).squeeze().to(device)
        a = torch.arange(0, batch_size, 1).type(torch.LongTensor).to(device)
        point_x = torch.zeros((batch_size, 1)).to(device)
        point_y = torch.zeros((batch_size, 1)).to(device)

        point_x = locations[:, 0, :].squeeze()
        point_x = point_x[a, all_position_onehot].reshape([batch_size, 1])
        point_y = locations[:, 1, :].squeeze()
        point_y = point_y[a, all_position_onehot].reshape([batch_size, 1])
        # print("sec", systemtime.time()-t2)

        location_x = locations[:, 0].data
        location_y = locations[:, 1].data
        # chosen point to others
        d = torch.sqrt((location_x - point_x) ** 2 + (location_y - point_y) ** 2)

        time = dynamic.data[:, 2]
        # print(d, time)
        due_time = static.data[:, 3]
        # print(time, d, due_time)
        # mask = need to be carried + <loads + TW ok
        new_mask = demands.ne(0) * demands.lt(loads) * (time + d).lt(due_time)
        # print(new_mask[-1])

        # We should avoid traveling to the depot back-to-back
        repeat_home = chosen_idx.ne(0)

        if repeat_home.any():
            new_mask[repeat_home.nonzero(), 0] = 1.
        if (1 - repeat_home).any():
            new_mask[(1 - repeat_home).nonzero(), 0] = 0.

        # ... unless we're waiting for all other samples in a minibatch to finish
        has_no_load = loads[:, 0].eq(0).float()
        has_no_demand = demands[:, 1:].sum(1).eq(0).float()

        combined = (has_no_load + has_no_demand).gt(0)
        if combined.any():
            new_mask[combined.nonzero(), 0] = 1.
            new_mask[combined.nonzero(), 1:] = 0.

        # 2019.1.3 add ant
        early_time = static.data[:, 2]
        new_mask = new_mask.type(torch.FloatTensor).to(device)
        arrive_time = d + time

        # β = 1
        # γ = 1.5
        # θ = 1
        # α = 3
        #
        # d = d + 1e-5
        # η = 1 / d
        # η = η ** β
        # φ = 1 / (due_time - early_time)
        # φ = φ ** γ
        # k = torch.abs(arrive_time - early_time) + torch.abs(arrive_time - due_time)
        # k = 1 / k
        # k = k ** θ
        # w = (loads + demands) / self.max_load
        # w = w ** α
        #
        # ant_mask = η * φ * k * w
        # new_mask = new_mask * ant_mask
        # # print("ant after", new_mask[0])
        # sum = new_mask.sum(1).unsqueeze(1) + 1e-5
        # # print(sum[0])
        # new_mask /= sum

        return torch.tensor(new_mask.float().data, device=device, requires_grad=False)

    def update_dynamic(self, static, dynamic, chosen_idx,beam_parent,is_beam = False):
        """Updates the (load, demand, time, position) dataset values."""
        # print(t_dy)
        # Update the dynamic elements differently for if we visit depot vs. a city

        if is_beam:
            # BatchBeamSeq: [batch_size*beam_width x 1]
            # [0,1,2,3,...,127,0,1,...],
            batchBeamSeq = (torch.range(0,self.batch_size-1).type(torch.LongTensor)).to(device).repeat(self.beam_width).unsqueeze(1)
            # print('batchBeamSeq',batchBeamSeq.shape)
            # batchedBeamIdx:[batch_size*beam_width]
            batchedBeamIdx = batchBeamSeq + self.batch_size * beam_parent.type(torch.cuda.LongTensor)
            dynamic = torch.index_select(dynamic, 0, batchedBeamIdx.squeeze(1))
            # print(static.shape)

        batch_size = dynamic.shape[0]
        visit = chosen_idx.ne(0)
        depot = chosen_idx.eq(0)

        # Clone the dynamic variable so we don't mess up graph
        all_demands = dynamic[:, 0].clone()
        all_loads = dynamic[:, 1].clone()
        all_time = dynamic[:, 2].clone()
        all_position = dynamic[:, 3].clone()
        locations = static[:, :2, :]
        # print(all_position[:,0])
        # calculate d & t
        # point_x = torch.zeros((batch_size, 1)).to(device)
        # point_y = torch.zeros((batch_size, 1)).to(device)
        # for i in range(batch_size):
        #     point_x[i] = locations[i, 0, int(all_position[i][0])]
        #     point_y[i] = locations[i, 1, int(all_position[i][0])]
        all_position_onehot = all_position[:, 0].type(torch.LongTensor).reshape(
            [1, batch_size]).squeeze().to(device)
        #print(all_position_onehot.type())
        a = torch.arange(0, batch_size, 1).type(torch.LongTensor).to(device)

        point_x = locations[:, 0, :].squeeze()
        point_x = point_x[a, all_position_onehot].reshape([batch_size, 1])
        point_y = locations[:, 1, :].squeeze()
        point_y = point_y[a, all_position_onehot].reshape([batch_size, 1])

        location_x = locations[:, 0, :].squeeze()
        location_y = locations[:, 1, :].squeeze()

        # chosen point to others
        d = torch.sqrt((location_x - point_x) ** 2 + (location_y - point_y) ** 2)
        # print('d',d.shape)
        # print('ptr',chosen_idx.shape)
        d = torch.gather(d, 1, chosen_idx.unsqueeze(1))
        d_t = d
        early_time = static[:, 2, :]
        service_time = static[:, 4, :]

        load = torch.gather(all_loads, 1, chosen_idx.unsqueeze(1))
        early_time_g = torch.gather(early_time, 1, chosen_idx.unsqueeze(1))

        demand = torch.gather(all_demands, 1, chosen_idx.unsqueeze(1))
        time = torch.gather(all_time, 1, chosen_idx.unsqueeze(1))
        # demand & load & time & position
        if visit.any():
            new_load = load - demand
            new_start_time = torch.max(early_time_g, d_t + time)
            service_time = torch.gather(service_time, 1, chosen_idx.unsqueeze(1))

            visit_idx = visit.nonzero()
            if visit_idx.size() == torch.Size([1, 1]):
                visit_idx = visit_idx.reshape((-1,))
            else:
                visit_idx = visit_idx.squeeze(1)
            all_loads[visit_idx] = new_load[visit_idx]
            all_demands[visit_idx, chosen_idx[visit_idx]] = 0
            all_time[visit_idx] = (new_start_time + service_time)[visit_idx]
            all_position[visit_idx] = chosen_idx[visit_idx].unsqueeze(1).type(torch.FloatTensor).to(device)

        # print(all_time, all_position)
        # Return to depot to fill vehicle load
        if depot.any():
            depot_idx = depot.nonzero().squeeze()

            all_loads[depot_idx] = self.max_load
            all_time[depot_idx] = 0
            all_position[depot_idx] = 0

        # print(all_loads, all_demands, all_time, all_position)

        tensor = torch.cat(
            (all_demands.unsqueeze(1), all_loads.unsqueeze(1), all_time.unsqueeze(1), all_position.unsqueeze(1)), 1)
        # print(tensor)

        return torch.tensor(tensor.float().data, device=device, requires_grad=False)


def my_func(line):

    while line[-1] == 0:
        line = line[0:-1]
    mask_0 = (line == 0)
    line_new = line[mask_0]
    return line_new.size+1

def reward_num(static, tour_indices):
    '''
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """

    # Convert the indices back into a tour
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1)
    '''
    tour_indices =  tour_indices.cpu().detach().numpy()
    reward = torch.from_numpy(np.apply_along_axis(my_func, 1, tour_indices)).to(device)
    #print(reward)
    return torch.tensor(reward.float().data, device=device, requires_grad=False)


def reward(static, tour_indices):
    """
    Euclidean distance between all cities / nodes given by tour_indices
    """

    # Convert the indices back into a tour
    # print('tour_indices',tour_indices.shape)
    # print()
    idx = tour_indices.unsqueeze(1).expand(-1, static.size(1), -1)
    tour = torch.gather(static.data, 2, idx).permute(0, 2, 1)

    # Ensure we're always returning to the depot - note the extra concat
    # won't add any extra loss, as the euclidean distance between consecutive
    # points is 0
    start = static.data[:, :, 0].unsqueeze(1)
    y = torch.cat((start, tour, start), dim=1)

    # Euclidean distance between each consecutive point
    tour_len = torch.sqrt(torch.sum(torch.pow(y[:, :-1] - y[:, 1:], 2), dim=2))

    return tour_len.sum(1)


def render(static, tour_indices, save_path):
    ...


if __name__ == '__main__':
    print('Cannot be called from main')
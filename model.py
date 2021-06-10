import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import pandas as pd

device = torch.device("cuda:0" )

class Encoder(nn.Module):
    """Encodes the static & dynamic states using 1d Convolution."""

    def __init__(self, input_size, hidden_size):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(input_size, hidden_size, kernel_size=1)

    def forward(self, input):
        output = self.conv(input)
        return output  # (batch, hidden_size, seq_len)



class Attention(nn.Module):
    """Calculates attention over the input nodes given the current state."""

    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.W = nn.Parameter(torch.zeros((1, hidden_size),
                                          device=device, requires_grad=True))
        self.V = nn.Parameter(torch.zeros((1, hidden_size),
                                          device=device, requires_grad=True))

    def forward(self, mark,static_hidden, dynamic_hidden_ld, dynamic_hidden_d, decoder_hidden):

        batch_size, hidden_size, _ = static_hidden.size()

        decoder_hidden = decoder_hidden.unsqueeze(2).expand_as(static_hidden)

        hidden = decoder_hidden + static_hidden + dynamic_hidden_d + dynamic_hidden_ld
        # if mark is not None:
        #     decoder_hidden = np.average(decoder_hidden.squeeze().cpu().detach().numpy())
        #     # print(decoder_hidden)
        #     static_hidden = np.average(static_hidden.cpu().detach().numpy())
        #     dynamic_hidden_d = np.average(dynamic_hidden_d.cpu().detach().numpy())
        #     dynamic_hidden_ld = np.average(dynamic_hidden_ld.cpu().detach().numpy())
        #     each = [decoder_hidden,static_hidden,dynamic_hidden_d,dynamic_hidden_ld]
        #     mark = mark.append([each])
        # print(hidden.shape)

        # Broadcast some dimensions so we can do batch-matrix-multiply
        W = self.W.expand(batch_size, 1,hidden_size )

        attns = torch.squeeze(torch.bmm(W, torch.tanh(hidden)),1)

        attns = attns
        return attns,mark


class Pointer(nn.Module):
    """Calculates the next state given the previous state and input embeddings."""

    def __init__(self, hidden_size, num_layers=1, dropout=0.1):
        super(Pointer, self).__init__()

        self.hidden_size = hidden_size

        # Used to compute a representation of the current decoder output
        self.lstm = torch.nn.LSTMCell(input_size=hidden_size,hidden_size = hidden_size)
        self.lstm = self.lstm.to(device)
        self.encoder_attn = Attention(hidden_size)
        self.encoder_attn = self.encoder_attn.to(device)

        self.project_d = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device) #conv1d_1
        self.project_ld = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device)  # conv1d_1

        self.project_query = nn.Linear(hidden_size, hidden_size).to(device)

        self.project_ref = nn.Conv1d(hidden_size, hidden_size, kernel_size=1).to(device) #conv1d_4

        self.drop_cc = nn.Dropout(p=dropout)
        self.drop_hh = nn.Dropout(p=dropout)

    def forward(self, mark, static_hidden, dynamic_hidden_ld, dynamic_hidden_d, decoder_hidden, last_hh,last_cc):

        last_hh,last_cc = self.lstm(decoder_hidden, (last_hh,last_cc))

        last_hh = self.drop_hh(last_hh)
        last_cc = self.drop_hh(last_cc)

        static_hidden = self.project_ref(static_hidden)
        dynamic_hidden_ld = self.project_ld(dynamic_hidden_ld)
        dynamic_hidden_d =  self.project_d(dynamic_hidden_d)
        last_hh_1 = self.project_query(last_hh)

        # Given a summary of the output, find an  input context
        enc_attn,mark = self.encoder_attn(mark,static_hidden, dynamic_hidden_ld,dynamic_hidden_d, last_hh_1)

        return enc_attn, last_hh,last_cc,mark


class DRL4TSP(nn.Module):

    def __init__(self, static_size, dynamic_size, hidden_size,
                 update_fn=None, mask_fn=None, num_layers=1, dropout=0.):
        super(DRL4TSP, self).__init__()

        if dynamic_size < 1:
            raise ValueError(':param dynamic_size: must be > 0, even if the '
                             'problem has no dynamic elements')

        self.update_fn = update_fn
        self.mask_fn = mask_fn

        # Define the encoder  models
        self.static_encoder = Encoder(static_size, hidden_size)
        self.dynamic_encoder_ld = Encoder(dynamic_size, hidden_size)
        self.dynamic_encoder_d = Encoder(dynamic_size, hidden_size)

        self.pointer = Pointer(hidden_size, num_layers, dropout)

        self.static_encoder = self.static_encoder.to(device)
        self.dynamic_encoder_ld = self.dynamic_encoder_ld.to(device)
        self.dynamic_encoder_d = self.dynamic_encoder_d.to(device)
        self.pointer = self.pointer.to(device)


        for p in self.parameters():
            if len(p.shape) > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, static, dynamic, is_beam_search = False, mark = None,decoder_input=None, last_hh=None):

        batch_size, input_size, sequence_size = static.size()
        hidden_dim = 128

        if is_beam_search:
            beam_width = 10
            dynamic = dynamic.repeat(beam_width, 1, 1)
            static = static.repeat(beam_width, 1, 1)

        else:
            beam_width = 1
        # Always use a mask - if no function is provided, we don't update it
        mask = [0] + [1]*(sequence_size-1)
        # print(mask)
        mask = np.array(mask)
        mask = torch.from_numpy(mask).type(torch.FloatTensor)
        mask = mask.expand(batch_size*beam_width,sequence_size)
        mask = torch.tensor(mask.data).to(device)  #batch*beam x point

        # Structures for holding the output sequences
        tour_idx, tour_logp = [], []

        # Static elements only need to be processed once, and can be used across all 'pointing' iterations.
        # When / if the dynamic elements change, their representations will need to get calculated again.
        static_hidden = self.static_encoder(static)    # batch*beam x hidden_size x point

        load = dynamic[:,0:1,:]
        demand = dynamic[:,1:2,:]
        ld = load-demand
        # dynamic_input = torch.tensor(np.concatenate((ld, demand), axis=1)).type(torch.FloatTensor).to(device)

        dynamic_hidden_ld = self.dynamic_encoder_ld(ld)        # batch*beam x  hidden_size x point
        dynamic_hidden_d = self.dynamic_encoder_d(demand)      # batch*beam x hidden_size x point

        decoder_hidden = static_hidden[:, :, 0]                #batch*beam x hidden_size
        # print(decoder_hidden.shape)
        # initial decoder h
        last_hh = torch.zeros((batch_size*beam_width,dynamic_hidden_d.size()[1]),device=device,requires_grad= True)      # batch*beam x hidden_size
        last_cc = torch.zeros((batch_size*beam_width, dynamic_hidden_d.size()[1]), device=device,requires_grad=True)

        if is_beam_search:
            BatchSequence = torch.range(0,batch_size * beam_width-1).type(torch.LongTensor).to(device).unsqueeze(1)

        max_steps = sequence_size if self.mask_fn is None else 1000
        step = 0
        beam_parent = None
        for _ in range(max_steps):
            # print(step)

            onesall = torch.ones(mask.size()).to(device)
            mask2 = torch.where(mask>0,onesall,torch.zeros(mask.size()).to(device))

            if not mask2.byte().any():

                break

            # print(mask.unsqueeze(1).shape)
            probs, last_hh,last_cc,mark  = self.pointer(mark,static_hidden,
                                          dynamic_hidden_ld,
                                          dynamic_hidden_d,
                                          decoder_hidden, last_hh,last_cc)


            # print("mask2", mask2)
            probs = F.softmax(probs + 10000*mask2, dim=1)

            # When training, sample the next step according to its probability.
            # During testing, we can take the greedy approach and choose highest
            if self.training:
                m = torch.distributions.Categorical(probs)

                # Sometimes an issue with Categorical & sampling on GPU; See:
                # https://github.com/pemami4911/neural-combinatorial-rl-pytorch/issues/5
                ptr = m.sample()
                # while not torch.gather(mask, 1, ptr.data.unsqueeze(1)).byte().all():
                #     ptr = m.sample()
                # print(ptr)
                logp = m.log_prob(ptr)

            else:
                if is_beam_search:
                    if step == 0:
                        # BatchBeamSeq: [batch_size*beam_width x 1]
                        # [0,1,2,3,...,127,0,1,...],
                        batchBeamSeq = (torch.range(0, batch_size - 1).type(torch.LongTensor)).to(device).repeat(
                            beam_width).unsqueeze(1)

                        beam_path = []
                        log_beam_probs = []
                        # in the initial decoder step, we want to choose beam_width different branches
                        # log_beam_prob: [batch_size, sourceL]
                        log_beam_prob = torch.log(torch.split(probs, batch_size, dim=0)[0])
                        # print('log_beam_prob',log_beam_prob.shape)

                    elif step > 0:
                        log_beam_prob = torch.log(probs) + log_beam_probs[-1]
                        # log_beam_prob:[batch_size, beam_width*sourceL]
                        log_beam_prob = torch.cat(torch.split(log_beam_prob, batch_size, dim=0), 1)
                        # print('log_beam_prob after', log_beam_prob.shape)

                    # topk_prob_val,topk_logprob_ind: [batch_size, beam_width]
                    topk_logprob_val, topk_logprob_ind = torch.topk(log_beam_prob, beam_width)

                    # topk_logprob_val , topk_logprob_ind: [batch_size*beam_width x 1]
                    topk_logprob_val = torch.transpose(torch.reshape(
                        torch.transpose(topk_logprob_val,0,1), [1, -1]),0,1)

                    topk_logprob_ind = torch.transpose(torch.reshape(
                        torch.transpose(topk_logprob_ind,0,1), [1, -1]),0,1)
                    # print('topk_logprob_val',topk_logprob_val.shape)
                    # idx,beam_parent: [batch_size*beam_width x 1]
                    ptr = (topk_logprob_ind % sequence_size).type(torch.cuda.LongTensor).squeeze(1) # Which city in route.
                    beam_parent = (topk_logprob_ind // sequence_size).type(torch.cuda.LongTensor)  # Which hypothesis it came from.

                    # batchedBeamIdx:[batch_size*beam_width]
                    # print('batchBeamSeq',batchBeamSeq.shape)
                    # print('batch_size * beam_parent',(batch_size * beam_parent).shape)
                    batchedBeamIdx = batchBeamSeq+ batch_size * beam_parent.type(torch.cuda.LongTensor)
                    prob = torch.index_select(probs, 0, batchedBeamIdx.squeeze(1))
                    # print('prob',prob.shape)
                    logp =topk_logprob_val

                    beam_path.append(beam_parent)
                    log_beam_probs.append(topk_logprob_val)

                else:
                    prob, ptr = torch.max(probs, 1)  # Greedy
                    logp = prob.log()

            # print(ptr.data)
            # print('log', logprob)

            # After visiting a node update the dynamic representation
            if self.update_fn is not None:
                dynamic = self.update_fn(static, dynamic,  ptr.data,beam_parent,is_beam_search)
                load = dynamic[:, 0:1, :]
                demand = dynamic[:, 1:2, :]
                ld = load - demand
                ld = load - demand
                dynamic_hidden_ld = self.dynamic_encoder_ld(ld)
                dynamic_hidden_d = self.dynamic_encoder_d(demand)

                # Since we compute the VRP in minibatches, some tours may have
                # number of stops. We force the vehicles to remain at the depot 
                # in these cases, and logp := 0
                is_done = dynamic[:, 1].sum(1).eq(0).float()
                logp = logp * (1. - is_done)

            # And update the mask so we don't re-visit if we don't need to
            if self.mask_fn is not None:
                mask = self.mask_fn(mask, static, dynamic, ptr.data)

            tour_logp.append(logp.unsqueeze(1))
            tour_idx.append(ptr.data.unsqueeze(1))
            step += 1
            decoder_hidden = torch.gather(static_hidden, 2,
                                         ptr.view(-1, 1, 1).expand(-1, hidden_dim, 1)).detach().squeeze(2)

        if is_beam_search:
            # find paths of the beam search
            tmplst = []
            tmpind = [BatchSequence]
            for k in reversed(range(len(tour_idx))):
                # print('k',k)
                # print('tour_idx[k]',tour_idx[k].shape)
                # print('tmpind',tmpind[-1].shape)
                tmplst = [torch.index_select(tour_idx[k],0, tmpind[-1].squeeze(1))] + tmplst
                # print('tmplst',tmplst[-1].shape)
                tmpind += [torch.index_select(
                    (batchBeamSeq + batch_size*beam_path[k]).type(torch.cuda.LongTensor),0,tmpind[-1].squeeze(1))]
            tour_idx = tmplst
            # print('tmplst', tmplst[-1].shape)
        tour_idx = torch.cat(tour_idx, dim=1)  # (batch_size, seq_len)
        # print('!!!!!!!!!!!!!!!!!!!',tour_idx.shape)
        tour_logp = torch.cat(tour_logp, dim=1)  # (batch_size, seq_len)

        return tour_idx, tour_logp,mark


if __name__ == '__main__':
    raise Exception('Cannot be called from main')

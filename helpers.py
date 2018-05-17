from torch.autograd import Variable
from torch import nn
import torch
import numpy as np
import os
import struct
from bball_data.utils import unnormalize, plot_sequence
import pickle

use_gpu = torch.cuda.is_available()

def estimate_advantages(rewards, values, gamma, tau, use_gpu):
    if use_gpu:
        rewards, values = rewards.cpu(), values.cpu()
    tensor_type = type(rewards)
    returns = tensor_type(rewards.size(0), rewards.size(1), 1)
    deltas = tensor_type(rewards.size(0), rewards.size(1), 1)
    advantages = tensor_type(rewards.size(0), rewards.size(1), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + gamma * prev_return
        deltas[i] = rewards[i] + gamma * prev_value - values[i]
        advantages[i] = deltas[i] + gamma * tau * prev_advantage
        
        prev_return = returns[i]
        prev_value = values[i]
        prev_advantage = advantages[i]

    advantages = (advantages - advantages.mean()) / advantages.std()

    if use_gpu:
        advantages, returns = advantages.cuda(), returns.cuda()
    return advantages, returns

def ones(*shape):
    return torch.ones(*shape).cuda() if use_gpu else torch.ones(*shape)

def zeros(*shape):
    return torch.zeros(*shape).cuda() if use_gpu else torch.zeros(*shape)

def update_dis_and_critic(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, states, actions, l2_reg, i_iter, dis_times, critic_times, use_gpu, update_discrim = True, clip=None):
    if use_gpu:
        exp_states, exp_actions, states, actions = exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()

    """update discriminator"""
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(int(dis_times)):
        if clip is not None:
            for p in discrim_net.parameters():
                p.data.clamp_(-clip, clip)
                
        g_o = discrim_net(Variable(states), Variable(actions))
        e_o = discrim_net(Variable(exp_states), Variable(exp_actions))
        
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        
        if update_discrim:
            optimizer_discrim.zero_grad()
            if discrim_criterion is None:
                discrim_loss = g_o.mean() - e_o.mean()
            else:
                if discrim_net.name == "entire":
                    discrim_loss = discrim_criterion(g_o, Variable(zeros((states.shape[1], 1)))) + \
                        discrim_criterion(e_o, Variable(ones((exp_states.shape[1], 1))))
                else:
                    discrim_loss = discrim_criterion(g_o, Variable(zeros((g_o.shape[0], g_o.shape[1], 1)))) + \
                        discrim_criterion(e_o, Variable(ones((e_o.shape[0], e_o.shape[1], 1))))
            discrim_loss.backward()
            optimizer_discrim.step()
    
    if dis_times > 0:
        return g_o_ave / dis_times, e_o_ave / dis_times

def update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, states_var, actions_var, i_iter, clip_epsilon, use_gpu):
    optimizer_policy.zero_grad()
    g_o = discrim_net(states_var, actions_var)
    if discrim_criterion is None:
        policy_loss = -g_o.mean()
    else:
        if discrim_net.name == "entire":
            policy_loss = discrim_criterion(g_o, Variable(ones((states_var.shape[1], 1))))
        else:
            policy_loss = discrim_criterion(g_o, Variable(ones((g_o.shape[0], g_o.shape[1], 1))))
    
    policy_loss.backward()
    torch.nn.utils.clip_grad_norm(policy_net.parameters(), 10)
    optimizer_policy.step()

# pretrain policy
def pre_train_policy(policy_net, optimizer_policy, expert_data, size, train=True):
    # expert
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    sample_expert_data = expert_data[exp_ind].clone().transpose(0, 1)   ## size * seq_len * 10

    if use_gpu:
        sample_expert_data = sample_expert_data.cuda()
    
    hyperparams = {
        'train' : mid
    }
    
    loss = policy_net(Variable(sample_expert_data), hyperparams)
    
    if train:
        optimizer_policy.zero_grad()
        policy_loss.backward()
        optimizer_policy.step()
    
    return loss.data.cpu().numpy()[0]

# pretrain discriminator
def pre_train_discrim(discrim_net, discrim_criterion, optimizer_discrim, i_iter, exp_states, exp_actions, states, actions, clip=None):
    g_o_ave = 0.0
    e_o_ave = 0.0
    for _ in range(3):

        if clip is not None:
            for p in discrim_net.parameters():
                p.data.clamp_(-clip, clip)

        if use_gpu:
            exp_states, exp_actions, states, actions= exp_states.cuda(), exp_actions.cuda(), states.cuda(), actions.cuda()
        g_o = discrim_net(Variable(states), Variable(actions))
        e_o = discrim_net(Variable(exp_states), Variable(exp_actions))
        
        g_o_ave += g_o.cpu().data.mean()
        e_o_ave += e_o.cpu().data.mean()
        
        optimizer_discrim.zero_grad()
        if discrim_criterion is None:
            discrim_loss = g_o.mean() - e_o.mean()
        else:
            if discrim_net.name == "entire":
                discrim_loss = discrim_criterion(g_o, Variable(zeros((g_o.shape[0], 1)))) + \
                    discrim_criterion(e_o, Variable(ones((e_o.shape[0], 1))))
            else:
                discrim_loss = discrim_criterion(g_o, Variable(zeros((g_o.shape[0], g_o.shape[1], 1)))) + \
                    discrim_criterion(e_o, Variable(ones((e_o.shape[0], e_o.shape[1], 1))))
        discrim_loss.backward()
        optimizer_discrim.step()
    
    if i_iter % 1 == 0:
        with open("pretrain.txt", 'a') as text_file:
            text_file.write("exp: {:.4f}\t mod: {:.4f}\n".format(e_o_ave / 3.0, g_o_ave / 3.0))
        if clip is None:
            print(i_iter, 'exp: ', e_o_ave / 3.0, 'mod: ', g_o_ave / 3.0)
        else:
            print(i_iter, 'exp: ', e_o_ave / 3.0, 'mod: ', g_o_ave / 3.0, 'diff:', g_o_ave / 3.0 - e_o_ave / 3.0)
    
    return g_o_ave / 3.0

def collect_samples(policy_net, discrim_net, expert_data, use_gpu, burn_in, i_iter, size=64, name="sampling", draw=False, wgan=False, sampler='regular'):
    exp_ind = torch.from_numpy(np.random.choice(expert_data.shape[0], size)).long()
    data = expert_data[exp_ind].clone()
    seq_len = data.shape[0]
    
    if use_gpu:
        data = data.cuda()
    data = Variable(data.squeeze().transpose(0, 1))
    # data: seq_length * batch_size * 10

    if sampler == 'regular':
        samples = policy_net.sample(data, burn_in=burn_in)
    elif sampler == 'macro':
        samples = policy_net.sample_macro(data, burn_in=burn_in)
    elif sampler == 'mid':
        samples = policy_net.sample_mid(data, burn_in=burn_in)
    
    states = samples[:-1, :, :].clone()
    actions = samples[1:, :, :].clone()
    exp_states = data[:-1, :, :].clone()
    exp_actions = data[1:, :, :].clone()

    mod_stats = {}
    exp_stats = {}

    if draw:
        #print(samples[:, ])
        mod_stats = draw_data(samples.data, name, i_iter, burn_in)
        #print(mod_stats['ave_length'])
        exp_stats = draw_data(data.data, name + '_expert', i_iter, burn_in)
        #print(exp_stats['ave_length'])
    
    return exp_states.data, exp_actions.data, data.data, states, actions, samples.data, mod_stats, exp_stats

# transfer a 1-d vector into a string
def to_string(x):
    ret = ""
    for i in range(x.shape[0]):
        ret += "{:.3f} ".format(x[i])
    return ret

def ave_player_distance(states):
    # states: numpy (seq_lenth, batch, 10)
    count = 0
    ret = np.zeros(states.shape)
    for i in range(5):
        for j in range(i+1, 5):
            ret[:, :, count] = np.sqrt(np.square(states[:, :, 2 * i] - states[:, :, 2 * j]) + np.square(states[:, :, 2 * i + 1] - states[:, :, 2 * j + 1]))
            count += 1
    return ret

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.linalg.norm(v1) == 0.0 or np.linalg.norm(v2) == 0.0:
        return 0.0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.abs(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) - np.pi / 2)

def ave_rotation(actions):
    length = actions.shape[0]
    ret = np.zeros((length-1, actions.shape[1], 5))
    for i in range(length-1):
        for j in range(actions.shape[1]):
            for k in range(5):
                ret[i, j, k] = angle_between(actions[i, j, (2*k):(2*k+2)], actions[i+1, j, (2*k):(2*k+2)])
    return ret

def draw_data(model_states, name, i_iter, burn_in):
    #print(model_states.max(), model_states.min())
    print("Drawing")
    stats = {}
    model_actions = model_states[1:, :, :] - model_states[:-1, :, :]
        
    val_data = model_states.cpu().numpy()
    val_actions = model_actions.cpu().numpy()

    step_size = np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2]))
    change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
    stats['ave_change_step_size'] = np.mean(change_of_step_size)
    val_seqlength = np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis = 0)
    stats['ave_length'] = np.mean(val_seqlength)  ## when sum along axis 0, axis 1 becomes axis 0
    stats['ave_near_bound'] = np.mean((val_data < (-0.49)) + (val_data > (0.49)))
    stats['ave_out_of_bound'] = np.mean((val_data < -0.51) + (val_data > 0.51))
    
    # more stats added 180425
    stats['ave_player_distance'] = np.mean(ave_player_distance(val_data))
    stats['diff_max_min'] = np.mean(np.max(val_seqlength, axis=1) - np.min(val_seqlength, axis=1))
    stats['ave_angle'] = np.mean(ave_rotation(val_actions))
    
    draw_data = model_states.cpu().numpy()[:, 0, :] 
    draw_data = unnormalize(draw_data)
    colormap = ['b', 'r', 'g', 'm', 'y', 'c']
    plot_sequence(draw_data, macro_goals=None, colormap=colormap[:5], save_name="imgs/{}_{}_offense".format(name, i_iter), burn_in=burn_in)

    return stats

def test_fixed_data(policy_net, exp_state, burn_in, name, i_iter, num_draw=1):
    samples = policy_net.sample(exp_state, burn_in=burn_in)
    
    draw_data(samples.data, name, i_iter, burn_in)
    draw_data(exp_state.data, name + '_expert', i_iter, burn_in)

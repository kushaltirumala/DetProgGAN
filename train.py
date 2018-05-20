import argparse
import math
import os
import pickle
import time
import shutil

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
from torch.autograd import Variable

from bball_data import BBallData
from model import *
import visdom

from bball_data.utils import unnormalize, plot_sequence, animate_sequence

LEVEL = 'mid'

def printlog(line):
    print(line)
    with open(save_path+'log_' + LEVEL + '.txt', 'a') as file:
        file.write(line+'\n')

def hyperparams_str(epoch, hp):
    ret = 'Epoch: {:d}'.format(epoch)

    if hp['train'] is 'macro':
        ret += ' (pretrain macro)'
    if warmup > 0:
        ret += ' | Beta: {:.2f}'.format(hp['beta'])
    if min_eps < 1 or eps_start < n_epochs:
        ret += ' | Epsilon: {:.2f}'.format(hp['eps'])
    if 'GUMBEL' in params['model']:
        ret += ' | Tau: {:.2f}'.format(hp['tau'])

    return ret


def run_epoch(train, hp):
    loader = train_loader if train else test_loader
    losses = []
    test_count = 0

    for batch_idx, data in enumerate(loader):
        if args.cuda:
            data = data.cuda()

        # change (batch, time, x) to (time, batch, x)
        data = Variable(data.squeeze().transpose(0, 1))

        batch_loss = model(data, hp)

        if train:
            optimizer.zero_grad()
            total_loss = batch_loss
            total_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), clip)
            optimizer.step()
        
        losses.append(batch_loss.item())
        
        '''
        if batch_idx % 100 == 0:
            sample_and_draw(batch_idx)
            print(batch_loss.data.cpu().numpy()[0])
        
        test_count += 1
        if test_count > 2:
            break
        '''

    return np.mean(losses)

def sample_and_draw(i_iter, name='pretrain'):
    exp_data = next(iter(test_loader))
    if params['cuda']:
        exp_data = exp_data.cuda()
    
    exp_data = Variable(exp_data.squeeze().transpose(0, 1))
    
    samples = model.sample(exp_data, burn_in=params['burn_in'])

    exp_stats = draw_data(exp_data, "exp", i_iter)
    mod_stats = draw_data(samples, "mod", i_iter)
    
    return exp_stats, mod_stats

def draw_data(model_states, name, i_iter, pretrain=True):
    #print(model_states.max(), model_states.min())
    print("Drawing")
    stats = {}
    if not pretrain:
        model_actions = model_states[1:, :, :] - model_states[:-1, :, :]
            
        val_data = model_states.data.cpu().numpy()
        val_actions = model_actions.data.cpu().numpy()
    
        step_size = np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2]))
        change_of_step_size = np.abs(step_size[1:, :, :] - step_size[:-1, :, :])
        stats['ave_change_step_size'] = np.mean(change_of_step_size)
        stats['ave_length'] = np.mean(np.sum(np.sqrt(np.square(val_actions[:, :, ::2]) + np.square(val_actions[:, :, 1::2])), axis = 0))  ## when sum along axis 0, axis 1 becomes axis 0
        stats['ave_near_bound'] = np.mean((val_data < (-0.49)) + (val_data > (0.49)))
        stats['ave_out_of_bound'] = np.mean((val_data < -0.51) + (val_data > 0.51))

    draw_data = model_states.data.cpu().numpy()[:, 0, :]
    draw_data = unnormalize(draw_data)
    colormap = ['b', 'r', 'g', 'm', 'y', 'c']
    plot_sequence(draw_data, macro_goals=None, colormap=colormap[:5], save_name="imgs/{}_{}_offense".format(name, i_iter), burn_in=params['burn_in'])

    return stats

######################################################################
######################### MAIN STARTS HERE ###########################
######################################################################


parser = argparse.ArgumentParser()
parser.add_argument('-t', '--trial', type=int, required=True)
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--x_dim', type=int, required=True)
parser.add_argument('--y_dim', type=int, required=True)
parser.add_argument('--z_dim', type=int, required=True)
parser.add_argument('--h_dim', type=int, required=True, help='hidden state dimension')
parser.add_argument('--m_dim', type=int, required=True, help='macro-goal dimension')
parser.add_argument('--rnn_dim', type=int, required=True, help='num recurrent cells for next action/state')
parser.add_argument('--rnn_micro_dim', type=int, required=True, help='same as rnn_dim for macro-goal models')
parser.add_argument('--rnn_mid_dim', type=int, required=True, help='mid rnn_dim')
parser.add_argument('--rnn_macro_dim', type=int, required=True, help='num recurrent cells for macro-goals')
parser.add_argument('--n_agents', type=int, required=True)
parser.add_argument('--n_layers', type=int, required=False, default=1, help='num layers in recurrent cells')
parser.add_argument('--subsample', type=int, required=False, default=1, help='subsample sequeneces')
parser.add_argument('--seed', type=int, required=False, default=345, help='PyTorch random seed')
parser.add_argument('--n_epochs', type=int, required=True)
parser.add_argument('--clip', type=int, required=True, help='gradient clipping')
parser.add_argument('--start_lr', type=float, required=True, help='starting learning rate')
parser.add_argument('--min_lr', type=float, required=True, help='minimum learning rate')
parser.add_argument('--batch_size', type=int, required=False, default=64)
parser.add_argument('--save_every', type=int, required=False, default=25, help='periodically save model')
parser.add_argument('--min_eps', type=float, required=False, default=1, help='minimum epsilon for scheduled sampling')
parser.add_argument('--warmup', type=int, required=False, default=0, help='warmup for KL term')
parser.add_argument('--pretrain', type=int, required=False, default=50, help='num epochs to train macro-goal policy')
parser.add_argument('--cuda', action='store_true', default=False, help='use GPU')
parser.add_argument('--cont', action='store_true', default=False, help='continue training a model')
parser.add_argument('--pretrained', action='store_true', default=False, help='load pretrained model')
parser.add_argument('--burn_in', type=int, default=0, required=False, help='burn-in period')
parser.add_argument('--seq_len', type=int, default=0, required=False, help='length of sequence')
args = parser.parse_args()

if not torch.cuda.is_available():
    args.cuda = False

# model parameters
params = {
    'model' : args.model,
    'x_dim' : args.x_dim,
    'y_dim' : args.y_dim,
    'z_dim' : args.z_dim,
    'h_dim' : args.h_dim,
    'm_dim' : args.m_dim,
    'rnn_dim' : args.rnn_dim,
    'rnn_micro_dim' : args.rnn_micro_dim,
    'rnn_mid_dim': args.rnn_mid_dim,
    'rnn_macro_dim' : args.rnn_macro_dim,
    'n_agents' : args.n_agents,
    'n_layers' : args.n_layers,
    'subsample' : args.subsample,
    'seed' : args.seed,
    'cuda' : args.cuda,
    'burn_in' : args.burn_in
}

# hyperparameters
n_epochs = args.n_epochs
clip = args.clip
start_lr = args.start_lr
min_lr = args.min_lr
batch_size = args.batch_size
save_every = args.save_every

# scheduled sampling
min_eps = args.min_eps
eps_start = n_epochs

# anneal KL term in loss
warmup = args.warmup

# multi-stage training
pretrain_time = args.pretrain

# set manual seed
torch.manual_seed(params['seed'])
if args.cuda:
    torch.cuda.manual_seed(params['seed'])

# load model
model = eval(params['model'])(params)
if args.cuda:
    model.cuda()
params['total_params'] = num_trainable_params(model)
print(params)

# create save path and saving parameters
save_path = 'saved/%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'model/')
# pickle.dump(params, open(save_path+'params.p', 'wb'), protocol=2)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    BBallData(train=True, preprocess=True, subsample=params['subsample']), 
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    BBallData(train=False, preprocess=True, subsample=params['subsample']), 
    batch_size=batch_size, shuffle=True, **kwargs)

best_test_loss = 0
lr = start_lr

if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')

vis = visdom.Visdom()
win_pre_policy = None
win_path_length = None
win_out_of_bound = None
win_near_bound = None
win_step_change = None

# continue a previous experiment, but currently have to manually choose model
if args.cont:
    print("loading existing model ...")
    #state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best.pth')
    state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth')
    model.load_state_dict(state_dict)

    hyperparams = {
        'train' : LEVEL
    }
    
    test_loss = run_epoch(train=False, hp=hyperparams)
    printlog('Pretrain Test:\t' + str(test_loss))

for e in range(n_epochs):
    epoch = e+1

    if epoch > pretrain_time:
        exp_stats, mod_stats = sample_and_draw(epoch)
        
        update = 'append' if epoch - pretrain_time > 1 else None
        win_path_length = vis.line(X = np.array([epoch - pretrain_time]), Y = np.column_stack((np.array([exp_stats['ave_length']]), \
            np.array([mod_stats['ave_length']]))), win = win_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
        win_near_bound = vis.line(X = np.array([epoch - pretrain_time]), Y = np.column_stack((np.array([exp_stats['ave_near_bound']]), \
            np.array([mod_stats['ave_near_bound']]))), win = win_near_bound, update = update, opts=dict(legend=['expert', 'model'], title="average near bound rate"))
        win_out_of_bound = vis.line(X = np.array([epoch - pretrain_time]), Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), \
            np.array([mod_stats['ave_out_of_bound']]))), win = win_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
        win_step_change = vis.line(X = np.array([epoch - pretrain_time]), Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), \
            np.array([mod_stats['ave_change_step_size']]))), win = win_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))
    else:
        sample_and_draw(epoch)

    hyperparams = {
        'beta' : 1 if epoch > warmup else epoch/warmup,
        'eps' : 0 if epoch < eps_start else int((epoch-eps_start)/10) + 1,
        'tau' : max(2.5*math.exp(-e/100), 0.1),
        'train' : LEVEL
    }


    if epoch == 50 + pretrain_time or epoch == pretrain_time // 2:
        lr = min_lr
        print(lr)


    # can set a custom learning rate schedule
    # filter removes parameters with requires_grad=False
    # https://github.com/pytorch/pytorch/issues/679
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr)

    printlog(hyperparams_str(epoch, hyperparams))    
    start_time = time.time()

    train_loss = run_epoch(train=True, hp=hyperparams)
    printlog('Train:\t' + str(train_loss))

    test_loss = run_epoch(train=False, hp=hyperparams)
    printlog('Test:\t' + str(test_loss))

    epoch_time = time.time() - start_time
    printlog('Time:\t {:.3f}'.format(epoch_time))

    total_test_loss = test_loss

    # best model on test set
    if best_test_loss == 0 or total_test_loss < best_test_loss:    
        best_test_loss = total_test_loss
        filename = save_path+'model/'+params['model']+'_state_dict_best.pth'

        if epoch <= pretrain_time:
            filename = save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth'

        torch.save(model.state_dict(), filename)
        printlog('Best model at epoch '+str(epoch))

    # periodically save model
    if epoch % save_every == 0:
        filename = save_path+'model/'+params['model']+'_' + LEVEL + '_state_dict_'+str(epoch)+'.pth'
        torch.save(model.state_dict(), filename)
        printlog('Saved model')

    # end of pretrain stage
    if epoch == pretrain_time:
        printlog('END of pretrain')
        best_test_loss = 0
        lr = start_lr

        state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth')
        model.load_state_dict(state_dict)

        test_loss = run_epoch(train=False, hp=hyperparams)
        printlog('Test:\t' + str(test_loss))
    
printlog('Best Test Loss: {:.4f}'.format(best_test_loss))



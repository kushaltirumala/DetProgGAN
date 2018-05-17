import argparse
import os
import sys
import pickle
import time
import numpy as np
import shutil
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bball_data import BBallData
from model import *
from torch.autograd import Variable
from torch import nn

from helpers import *
import visdom

Tensor = torch.DoubleTensor
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser(description='PyTorch GAIL example')
parser.add_argument('--trial', type=int, default=800)
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-8, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.2, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=666, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=256, metavar='N',
                    help='minimal batch size per PPO update (default: 128)')
parser.add_argument('--max-iter-num', type=int, default=50000, metavar='N',
                    help='maximal number of main iterations (default: 2000)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--draw-interval', type=int, default=50, metavar='N',
                    help='interval between drawing and more detailed information (default: 10)')
parser.add_argument('--save-model-interval', type=int, default=50, metavar='N',
                    help="interval between saving model (default: 0, means don't save)")
parser.add_argument('--ppo-epochs', type=int, default=1, metavar='N',
                    help="ppo training epochs (default: 1)")
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help="ppo training batch size (default: 64)")
parser.add_argument('--val-freq', type=int, default=250, metavar='N',
                    help="pretrain validation frequency (default: 250)")
parser.add_argument('--pretrain-policy-iter', type=int, default=0, metavar='N',
                    help="pretrain policy iteration (default: 5000)")
parser.add_argument('--pretrain-disc-iter', type=int, default=0, metavar='N',
                    help="pretrain discriminator iteration (default: 30)")
parser.add_argument('--burn_in', type=int, default=0, required=False, help='burn-in period')

args = parser.parse_args()

use_gpu = True

if not torch.cuda.is_available():
    use_gpu = False

np.random.seed(args.seed)
torch.manual_seed(args.seed)
if use_gpu:
    torch.cuda.manual_seed_all(args.seed)
batch_size = args.batch_size

params = {
    'model' : "PROG_RNN_DET",
    'x_dim' : 2,
    'y_dim' : 10,
    'z_dim' : 16,
    'h_dim' : 200,
    'm_dim' : 90,
    'rnn_dim' : 200,
    'rnn_micro_dim' : 400,
    'rnn_mid_dim' : 300,
    'rnn_macro_dim' : 200,
    'n_agents' : 5,
    'n_layers' : 2,
    'subsample' : 16,
    'seed' : args.seed,
    'cuda' : use_gpu
}

save_path = 'saved/%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'model/')

discrim_net = Discriminator(10, 128, 10, num_layers=2).double()
#discrim_net = Discriminator_entire(10, 128, 10, num_layers=2).double()
policy_net = eval(params['model'])(params).double()
if use_gpu:
    policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
params['total_params'] = num_trainable_params(policy_net)
print(params)

#####################################################
### Load Models
load_models = True
if load_models:
    print("loading existing model ...")
    # train from pretrain
    policy_state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth')
    policy_net.load_state_dict(policy_state_dict, strict=False)
    
    # train from current good
    #policy_state_dict = torch.load(save_path+'model/policy_training_currentbest.pth')
    #policy_net.load_state_dict(policy_state_dict)
    
    # continue training
    
    # policy_state_dict = torch.load(save_path+'model/policy_training.pth')
    # policy_net.load_state_dict(policy_state_dict)
    # discrim_state_dict = torch.load(save_path+'model/discrim_training.pth')
    # discrim_net.load_state_dict(discrim_state_dict)
    
#####################################################

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
test_data = torch.Tensor(pickle.load(open('bball_data/data/Xte_role.p', 'rb'))).transpose(0, 1)[:, ::params['subsample'], :]
train_data = torch.Tensor(pickle.load(open('bball_data/data/Xtr_role.p', 'rb'))).transpose(0, 1)[:, ::params['subsample'], :]
print(test_data.shape, train_data.shape)

'''
train_loader = torch.utils.data.DataLoader(
    BBallData(train=True, preprocess=True, subsample=params['subsample']), 
    batch_size=batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    BBallData(train=False, preprocess=True, subsample=params['subsample']), 
    batch_size=batch_size, shuffle=True, **kwargs)
'''

optimizer_policy = torch.optim.Adam(
    filter(lambda p: p.requires_grad, policy_net.parameters()),
    lr=args.learning_rate * 0.001)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)
discrim_criterion = nn.BCELoss()
if use_gpu:
    discrim_criterion = discrim_criterion.cuda()

# stats
vis = visdom.Visdom()
exp_p = []
win_exp_p = None
mod_p = []
win_mod_p = None
win_pre_policy = None
win_path_length = None
win_out_of_bound = None
win_near_bound = None
win_step_change = None
win_ave_player_dis = None
win_diff_max_min = None
win_ave_angle = None
count = 0
if os.path.exists('imgs'):
    shutil.rmtree('imgs')
if not os.path.exists('imgs'):
    os.makedirs('imgs')
with open("intermediates.txt", "w") as text_file:
    text_file.write("\n")
with open("training.txt", "w") as text_file:
    text_file.write("\n")
with open("action_details.txt", "w") as text_file:
    text_file.write("\n")
with open("pretrain.txt", "w") as text_file:
    text_file.write("\n")
with open("val_stats.txt", "w") as text_file:
    text_file.write("\n")
with open("val_stats_expert.txt", "w") as text_file:
    text_file.write("\n")

fixed_test_data = Variable(test_data[:5].squeeze().transpose(0, 1))
if use_gpu:
    fixed_test_data = fixed_test_data.cuda()

test_fixed_data(policy_net, fixed_test_data, burn_in=args.burn_in, name='pretrained', i_iter=-1, num_draw=1)

# Pretrain Discriminator
for i in range(args.pretrain_disc_iter):
    exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples(policy_net, discrim_net, train_data, use_gpu, args.burn_in, i, name="pretraining", draw=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data
    ret = pre_train_discrim(discrim_net, discrim_criterion, optimizer_discrim, i, exp_states, exp_actions, model_states, model_actions)

    if ret < 0.3:
        break

# Save pretrained model
if args.pretrain_disc_iter > 250:
    torch.save(policy_net.state_dict(), save_path+'model/policy_pretrained.pth')
    torch.save(discrim_net.state_dict(), save_path+'model/discrim_pretrained.pth')
test_fixed_data(policy_net, fixed_test_data, burn_in=args.burn_in, name='pretrained', i_iter=0, num_draw=1)

#policy_net.stop_grad_helper()

# GAN training
update_discrim = True
for i_iter in range(args.max_iter_num):
    ts0 = time.time()
    print("Collecting Data")
    exp_states, exp_actions, exp_seq, model_states_var, model_actions_var, model_seq, mod_stats, exp_stats = \
        collect_samples(policy_net, discrim_net, train_data, use_gpu, 0, i_iter, draw=False)
    model_states = model_states_var.data
    model_actions = model_actions_var.data
    
    if i_iter % args.draw_interval == 0:
        test_fixed_data(policy_net, fixed_test_data, burn_in=args.burn_in, name='fixed_test', i_iter=i_iter, num_draw=1)
        _, _, _, _, _, _, mod_stats, exp_stats = \
            collect_samples(policy_net, discrim_net, test_data, use_gpu, args.burn_in, i_iter, draw=True)
    
        update = 'append' if i_iter > 0 else None
        win_path_length = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['ave_length']]), \
            np.array([mod_stats['ave_length']]))), win = win_path_length, update = update, opts=dict(legend=['expert', 'model'], title="average path length"))
        win_near_bound = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['ave_near_bound']]), \
            np.array([mod_stats['ave_near_bound']]))), win = win_near_bound, update = update, opts=dict(legend=['expert', 'model'], title="average near bound rate"))
        win_out_of_bound = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['ave_out_of_bound']]), \
            np.array([mod_stats['ave_out_of_bound']]))), win = win_out_of_bound, update = update, opts=dict(legend=['expert', 'model'], title="average out of bound rate"))
        win_step_change = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['ave_change_step_size']]), \
            np.array([mod_stats['ave_change_step_size']]))), win = win_step_change, update = update, opts=dict(legend=['expert', 'model'], title="average step size change"))
        win_ave_player_dis = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['ave_player_distance']]), \
            np.array([mod_stats['ave_player_distance']]))), win = win_ave_player_dis, update = update, opts=dict(legend=['expert', 'model'], title="average player distance"))
        win_diff_max_min = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['diff_max_min']]), \
            np.array([mod_stats['diff_max_min']]))), win = win_diff_max_min, update = update, opts=dict(legend=['expert', 'model'], title="average max and min path diff"))
        win_ave_angle = vis.line(X = np.array([i_iter // args.draw_interval]), Y = np.column_stack((np.array([exp_stats['ave_angle']]), \
            np.array([mod_stats['ave_angle']]))), win = win_ave_angle, update = update, opts=dict(legend=['expert', 'model'], title="average rotation angle"))
    
    print("Collecting Data Finished")
    ts1 = time.time()

    t0 = time.time()
    # update discriminator
    mod_p_epoch, exp_p_epoch = update_dis_and_critic(discrim_net, optimizer_discrim, discrim_criterion, exp_states, exp_actions, \
        model_states, model_actions, args.l2_reg, i_iter, dis_times=3.0, critic_times=10.0, use_gpu=use_gpu, update_discrim=update_discrim)
    exp_p.append(exp_p_epoch)
    mod_p.append(mod_p_epoch)
    
    # update policy network using ppo
    if i_iter > 3 and mod_p[-1] < 0.8:
        update_policy(policy_net, optimizer_policy, discrim_net, discrim_criterion, model_states_var, model_actions_var, i_iter, args.clip_epsilon, use_gpu)
    
    t1 = time.time()

    if i_iter % args.log_interval == 0:
        print('{}\tT_sample {:.4f}\tT_update {:.4f}\t\texp_p {:.3f}\tmod_p {:.3f}'.format(
            i_iter, ts1-ts0, t1-t0, exp_p[-1], mod_p[-1]))
        
        update = 'append'
        if win_exp_p is None:
            update = None
        win_exp_p = vis.line(X = np.array([count]), Y = np.column_stack((np.array([exp_p[-1]]), np.array([mod_p[-1]]))), win = win_exp_p, \
                          update = update, opts=dict(legend=['expert_prob', 'model_prob'], title="training curve probs"))
        
        with open("training.txt", "a") as text_file:
            text_file.write('{}\tT_sample {:.4f}\tT_update {:.4f}\texp_p {:.3f}\tmod_p {:.3f}\n'.format(
            i_iter, ts1-ts0, t1-t0, exp_p[-1], mod_p[-1]))
        
        count += 1

    if args.save_model_interval > 0 and (i_iter) % args.save_model_interval == 0:
        torch.save(policy_net.state_dict(), save_path+'model/policy_training.pth')
        torch.save(discrim_net.state_dict(), save_path+'model/discrim_training.pth')

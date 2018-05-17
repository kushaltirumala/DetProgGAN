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
parser.add_argument('--trial', type=int, default=601)
parser.add_argument('--gamma', type=float, default=0.98, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95, metavar='G',
                    help='gae (default: 0.95)')
parser.add_argument('--l2-reg', type=float, default=1e-8, metavar='G',
                    help='l2 regularization regression (default: 1e-3)')
parser.add_argument('--learning-rate', type=float, default=1e-3, metavar='G',
                    help='gae (default: 3e-4)')
parser.add_argument('--clip-epsilon', type=float, default=0.1, metavar='N',
                    help='clipping epsilon for PPO')
parser.add_argument('--num-threads', type=int, default=1, metavar='N',
                    help='number of threads for agent (default: 1)')
parser.add_argument('--seed', type=int, default=666, metavar='N',
                    help='random seed (default: 1)')
parser.add_argument('--min-batch-size', type=int, default=256, metavar='N',
                    help='minimal batch size per PPO update (default: 128)')
parser.add_argument('--max-iter-num', type=int, default=100, metavar='N',
                    help='maximal number of main iterations (default: 2000)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--draw-interval', type=int, default=1, metavar='N',
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
    'subsample' : 2,
    'seed' : args.seed,
    'cuda' : use_gpu
}

save_path = 'saved/%03d/' % args.trial
if not os.path.exists(save_path):
    os.makedirs(save_path)
    os.makedirs(save_path+'model/')

discrim_net = Discriminator(10, 32, 10, num_layers=1).double()
policy_net = eval(params['model'])(params).double()
if use_gpu:
    policy_net, discrim_net = policy_net.cuda(), discrim_net.cuda()
params['total_params'] = num_trainable_params(policy_net)
print(params)

#policy_state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best.pth')
policy_state_dict = torch.load(save_path+'model/'+params['model']+'_state_dict_best_pretrain.pth')
#policy_net.load_state_dict(policy_state_dict)
#policy_state_dict = torch.load(save_path+'model/policy_training_850.pth')
#policy_state_dict = torch.load(save_path+'model/policy_training_currentbest.pth')
policy_net.load_state_dict(policy_state_dict, strict=False)

kwargs = {'num_workers': 1, 'pin_memory': True} if use_gpu else {}
test_data = torch.Tensor(pickle.load(open('bball_data/data/Xte_role.p', 'rb'))).transpose(0, 1)[:, ::params['subsample'], :]
train_data = torch.Tensor(pickle.load(open('bball_data/data/Xtr_role.p', 'rb'))).transpose(0, 1)[:, ::params['subsample'], :]
print(test_data.shape, train_data.shape)

optimizer_policy = torch.optim.Adam(
    filter(lambda p: p.requires_grad, policy_net.parameters()),
    lr=args.learning_rate * 0.001)
optimizer_discrim = torch.optim.Adam(discrim_net.parameters(), lr=args.learning_rate)
discrim_criterion = nn.BCELoss()
if use_gpu:
    discrim_criterion = discrim_criterion.cuda()

# stats
vis = visdom.Visdom()
ave_rewards = []
win_ave_rewards = None
exp_p = []
win_exp_p = None
mod_p = []
win_mod_p = None
value_loss = []
win_value_loss = None
win_pre_policy = None
win_path_length = None
win_out_of_bound = None
win_near_bound = None
win_step_change = None
win_nll_loss = None
win_ave_player_dis = None
win_diff_max_min = None
win_ave_angle = None
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

test_fixed_data(policy_net, fixed_test_data, burn_in=args.burn_in, name='pretrained', i_iter=0, num_draw=1)

# GAN training
update_discrim = True
for i_iter in range(args.max_iter_num):
    print(i_iter)
    _, _, _, _, _, _, mod_stats, exp_stats = \
        collect_samples(policy_net, discrim_net, test_data, use_gpu, args.burn_in, i_iter, draw=True, sampler='mid')

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

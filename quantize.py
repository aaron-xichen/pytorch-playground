import argparse

from utee import misc, quant, selector
import torch
import torch.backends.cudnn as cudnn
cudnn.benchmark =True
from IPython import embed
from collections import OrderedDict
from torch.autograd import Variable

parser = argparse.ArgumentParser(description='PyTorch SVHN Example')
parser.add_argument('--type', default='cifar10', help='mnist|svhn|cifar10|cifar100|stl10')
parser.add_argument('--batch_size', type=int, default=100, help='input batch size for training (default: 64)')
parser.add_argument('--gpu', default=None, help='index of gpus to use')
parser.add_argument('--ngpu', type=int, default=8, help='number of gpus to use')
parser.add_argument('--seed', type=int, default=117, help='random seed (default: 1)')
parser.add_argument('--model_root', default='~/.torch/models/', help='folder to save the model')
parser.add_argument('--data_root', default='/mnt/local0/public_dataset/pytorch/', help='folder to save the model')
parser.add_argument('--logdir', default='log/default', help='folder to save to the log')

parser.add_argument('--input_size', type=int, default=224, help='input size of image')
parser.add_argument('---n_sample', type=int, default=20, help='number of samples to infer the scaling factor')
parser.add_argument('--param_bits', type=int, default=8, help='bit-width for parameters')
parser.add_argument('--fwd_bits', type=int, default=8, help='bit-width for layer output')
parser.add_argument('--overflow_rate', type=float, default=0.0, help='overflow rate')
args = parser.parse_args()

args.gpu = misc.auto_select_gpu(utility_bound=0, num_gpu=args.ngpu, selected_gpus=args.gpu)
args.ngpu = len(args.gpu)
misc.ensure_dir(args.logdir)
args.model_root = misc.expand_user(args.model_root)
args.data_root = misc.expand_user(args.data_root)
args.input_size = 299 if 'inception' in args.type else args.input_size
print("=================FLAGS==================")
for k, v in args.__dict__.items():
    print('{}: {}'.format(k, v))
print("========================================")

assert torch.cuda.is_available(), 'no cuda'
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# load model and dataset fetcher
model_raw, ds_fetcher, is_imagenet = selector.select(args.type, model_root=args.model_root)
args.ngpu = args.ngpu if is_imagenet else 1

# quantize parameters
if args.param_bits < 32:
    state_dict = model_raw.state_dict()
    state_dict_quant = OrderedDict()
    sf_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'running' in k:
            print("Ignoring {}".format(k))
            v_quant = v
        else:
            v_quant, overflow_rate_real, sf = quant.linear_quantize_pytorch(Variable(v), bits=args.param_bits, overflow_rate=args.overflow_rate)
            v_quant = v_quant.data
            sf_dict[k] = (sf, args.param_bits, overflow_rate_real)
        state_dict_quant[k] = v_quant
    model_raw.load_state_dict(state_dict_quant)
    for k, v in sf_dict.items():
        print(k, v)

# quantize forward activation
if args.fwd_bits < 32:
    model_raw = quant.duplicate_model_with_quant(model_raw, bits=args.fwd_bits, overflow_rate=args.overflow_rate, counter=args.n_sample)
    print(model_raw)
    val_ds_tmp = ds_fetcher(10, data_root=args.data_root, train=False, input_size=args.input_size)
    misc.eval_model(model_raw, val_ds_tmp, ngpu=1, n_sample=args.n_sample, is_imagenet=is_imagenet)

# eval model
val_ds = ds_fetcher(args.batch_size, data_root=args.data_root, train=False, input_size=args.input_size)
acc = misc.eval_model(model_raw, val_ds, ngpu=args.ngpu, is_imagenet=is_imagenet)

# print sf
print(model_raw)
print("=> param_bits={}, fwd_bits={}, overflow_rate={}, acc={:.4f}".format(args.param_bits, args.fwd_bits, args.overflow_rate, acc))
embed()

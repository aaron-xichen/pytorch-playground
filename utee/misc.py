import cv2
import os
import shutil
import pickle as pkl
import time
import numpy as np
import hashlib

from IPython import embed

class Logger(object):
    def __init__(self):
        self._logger = None

    def init(self, logdir, name='log'):
        if self._logger is None:
            import logging
            if not os.path.exists(logdir):
                os.makedirs(logdir)
            log_file = os.path.join(logdir, name)
            if os.path.exists(log_file):
                os.remove(log_file)
            self._logger = logging.getLogger()
            self._logger.setLevel('INFO')
            fh = logging.FileHandler(log_file)
            ch = logging.StreamHandler()
            self._logger.addHandler(fh)
            self._logger.addHandler(ch)

    def info(self, str_info):
        self.init('/tmp', 'tmp.log')
        self._logger.info(str_info)
logger = Logger()

print = logger.info
def ensure_dir(path, erase=False):
    if os.path.exists(path) and erase:
        print("Removing old folder {}".format(path))
        shutil.rmtree(path)
    if not os.path.exists(path):
        print("Creating folder {}".format(path))
        os.makedirs(path)

def load_pickle(path):
    begin_st = time.time()
    with open(path, 'rb') as f:
        print("Loading pickle object from {}".format(path))
        v = pkl.load(f)
    print("=> Done ({:.4f} s)".format(time.time() - begin_st))
    return v

def dump_pickle(obj, path):
    with open(path, 'wb') as f:
        print("Dumping pickle object to {}".format(path))
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)

def auto_select_gpu(mem_bound=500, utility_bound=0, gpus=(0, 1, 2, 3, 4, 5, 6, 7), num_gpu=1, selected_gpus=None):
    import sys
    import os
    import subprocess
    import re
    import time
    import numpy as np
    if 'CUDA_VISIBLE_DEVCIES' in os.environ:
        sys.exit(0)
    if selected_gpus is None:
        mem_trace = []
        utility_trace = []
        for i in range(5): # sample 5 times
            info = subprocess.check_output('nvidia-smi', shell=True).decode('utf-8')
            mem = [int(s[:-5]) for s in re.compile('\d+MiB\s/').findall(info)]
            utility = [int(re.compile('\d+').findall(s)[0]) for s in re.compile('\d+%\s+Default').findall(info)]
            mem_trace.append(mem)
            utility_trace.append(utility)
            time.sleep(0.1)
        mem = np.mean(mem_trace, axis=0)
        utility = np.mean(utility_trace, axis=0)
        assert(len(mem) == len(utility))
        nGPU = len(utility)
        ideal_gpus = [i for i in range(nGPU) if mem[i] <= mem_bound and utility[i] <= utility_bound and i in gpus]

        if len(ideal_gpus) < num_gpu:
            print("No sufficient resource, available: {}, require {} gpu".format(ideal_gpus, num_gpu))
            sys.exit(0)
        else:
            selected_gpus = list(map(str, ideal_gpus[:num_gpu]))
    else:
        selected_gpus = selected_gpus.split(',')

    print("Setting GPU: {}".format(selected_gpus))
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(selected_gpus)
    return selected_gpus

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def model_snapshot(model, new_file, old_file=None, verbose=False):
    from collections import OrderedDict
    import torch
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    if old_file and os.path.exists(expand_user(old_file)):
        if verbose:
            print("Removing old model {}".format(expand_user(old_file)))
        os.remove(expand_user(old_file))
    if verbose:
        print("Saving model to {}".format(expand_user(new_file)))

    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        if v.is_cuda:
            v = v.cpu()
        state_dict[k] = v
    torch.save(state_dict, expand_user(new_file))


def load_lmdb(lmdb_file, n_records=None):
    import lmdb
    import numpy as np
    lmdb_file = expand_user(lmdb_file)
    if os.path.exists(lmdb_file):
        data = []
        env = lmdb.open(lmdb_file, readonly=True, max_readers=512)
        with env.begin() as txn:
            cursor = txn.cursor()
            begin_st = time.time()
            print("Loading lmdb file {} into memory".format(lmdb_file))
            for key, value in cursor:
                _, target, _ = key.decode('ascii').split(':')
                target = int(target)
                img = cv2.imdecode(np.fromstring(value, np.uint8), cv2.IMREAD_COLOR)
                data.append((img, target))
                if n_records is not None and len(data) >= n_records:
                    break
        env.close()
        print("=> Done ({:.4f} s)".format(time.time() - begin_st))
        return data
    else:
        print("Not found lmdb file".format(lmdb_file))

def str2img(str_b):
    return cv2.imdecode(np.fromstring(str_b, np.uint8), cv2.IMREAD_COLOR)

def img2str(img):
    return cv2.imencode('.jpg', img)[1].tostring()

def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()

def eval_model(model, ds, n_sample=None, ngpu=1, is_imagenet=False):
    import tqdm
    import torch
    from torch import nn
    from torch.autograd import Variable

    class ModelWrapper(nn.Module):
        def __init__(self, model):
            super(ModelWrapper, self).__init__()
            self.model = model
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        def forward(self, input):
            input.data.div_(255.)
            input.data[:, 0, :, :].sub_(self.mean[0]).div_(self.std[0])
            input.data[:, 1, :, :].sub_(self.mean[1]).div_(self.std[1])
            input.data[:, 2, :, :].sub_(self.mean[2]).div_(self.std[2])
            return self.model(input)

    correct1, correct5 = 0, 0
    n_passed = 0
    if is_imagenet:
        model = ModelWrapper(model)
    model = model.eval()
    model = torch.nn.DataParallel(model, device_ids=range(ngpu)).cuda()

    n_sample = len(ds) if n_sample is None else n_sample
    for idx, (data, target) in enumerate(tqdm.tqdm(ds, total=n_sample)):
        n_passed += len(data)
        data =  Variable(torch.FloatTensor(data)).cuda()
        indx_target = torch.LongTensor(target)
        output = model(data)
        bs = output.size(0)
        idx_pred = output.data.sort(1, descending=True)[1]

        idx_gt1 = indx_target.expand(1, bs).transpose_(0, 1)
        idx_gt5 = idx_gt1.expand(bs, 5)

        correct1 += idx_pred[:, :1].cpu().eq(idx_gt1).sum()
        correct5 += idx_pred[:, :5].cpu().eq(idx_gt5).sum()

        if idx >= n_sample - 1:
            break

    acc1 = correct1 * 1.0 / n_passed
    acc5 = correct5 * 1.0 / n_passed
    return acc1, acc5

def load_state_dict(model, model_urls, model_root):
    from torch.utils import model_zoo
    from torch import nn
    import re
    from collections import OrderedDict
    own_state_old = model.state_dict()
    own_state = OrderedDict() # remove all 'group' string
    for k, v in own_state_old.items():
        k = re.sub('group\d+\.', '', k)
        own_state[k] = v

    state_dict = model_zoo.load_url(model_urls, model_root)

    for name, param in state_dict.items():
        if name not in own_state:
            print(own_state.keys())
            raise KeyError('unexpected key "{}" in state_dict'
                           .format(name))
        if isinstance(param, nn.Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)

    missing = set(own_state.keys()) - set(state_dict.keys())
    no_use = set(state_dict.keys()) - set(own_state.keys())
    if len(no_use) > 0:
        raise KeyError('some keys are not used: "{}"'.format(no_use))


import torch
from utils import *
import argparse
import os
from models.wrappers import STR_Net, MTR_Net
from config import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

parser = argparse.ArgumentParser()

# environment configurations
parser.add_argument('--storage_root', default='C:\\Users\\dimme\\Cold\\MTL_Refine', type=str, help='path to storage (logs, checkpoints, etc.)')
parser.add_argument('--load_checkpoint', default=None, type=str, help='path to load checkpoint model')
parser.add_argument('--save_epoch', default=10, type=int, help='number of epochs to train before checkpointing')
parser.add_argument('--seed', default=0, type=int, help='random seed for pytorch')
parser.add_argument('--cuda', default='0', type=str, choices=['0','1','2', '3','4','5','6','7'], help='gpu index')
parser.add_argument('--name', default='test', type=str, help='name of the run')
parser.add_argument('--model_file', required=True, type=str, help='path to model config file')
parser.add_argument('--augmentation', default=False, type=bool, help='apply augmentations')
parser.add_argument('--grad_scaling', default=False, type=bool, help='apply gradient scaline')

def set_seeds(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Random seed set as {seed}")

if __name__ == "__main__":
    args = parser.parse_args()

    opt = create_config(vars(args), args.model_file)
    print(f'config: {opt}')

    set_seeds(opt.seed)

    device = torch.device(f"cuda:{opt.cuda}" if torch.cuda.is_available() else "cpu")
    print(f'device: {device}')

    rf_model = get_model(opt)
    criterion = get_criterion(opt)
    if opt.setup == 'single_task':
        model = STR_Net(opt.TASKS.NAMES, rf_model, criterion).to(device)
    else:    
        model = MTR_Net(opt.TASKS.NAMES, rf_model, criterion).to(device)

    if opt.load_checkpoint is not None:
        print(f'Loading checkpoint: {opt.checkpoint}')
        state_dict = torch.load(opt.load_checkpoint)
        model.load_state_dict(state_dict)
        print('Checkpoint loaded successfully...')

    model.set_optimizers(opt)
    model.set_schedulers(opt)

    train_set = get_train_dataset(opt)
    test_set = get_test_dataset(opt)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=opt.trBatch,
        pin_memory=True,
        drop_last=True,
        num_workers=opt.nworkers,
        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=opt.valBatch,
        pin_memory=True,
        drop_last=False,
        num_workers=opt.nworkers,
        shuffle=False)

    print('starting training...')
    multitask_trainer(train_loader,
                        test_loader,
                        model,
                        device,
                        opt)
    print('done...')

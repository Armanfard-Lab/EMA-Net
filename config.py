import torch
import yaml
from easydict import EasyDict as edict

def create_config(args, model):
    cfg = edict()
    
    for k,v in args.items():
        cfg[k] = v

    with open(model, 'r') as file:
        model_config = yaml.safe_load(file)
    
    for k,v in model_config.items():
        cfg[k] = v
    
    # Parse the task dictionary separately
    cfg.TASKS, extra_args = parse_task_dictionary(cfg['train_db_name'], cfg['task_dictionary'])

    for k, v in extra_args.items():
        cfg[k] = v
    
    # Other arguments   
    if cfg['train_db_name'] == 'nyuv2':
        cfg.TRAIN = edict()
        cfg.TEST = edict()
        cfg.TRAIN.SCALE = (288,384)
        cfg.TEST.SCALE = (288, 384)
    
    elif cfg['train_db_name'] == 'cityscapes':
        cfg.TRAIN = edict()
        cfg.TEST = edict()
        cfg.TRAIN.SCALE = (128,256)
        cfg.TEST.SCALE = (128, 256)

    else:
        raise NotImplementedError
    
    from configs.mypath import db_paths, PROJECT_ROOT_DIR
    cfg['db_paths'] = db_paths
    cfg['PROJECT_ROOT_DIR'] = PROJECT_ROOT_DIR
    
    return cfg

def parse_task_dictionary(db_name, task_dictionary):
    """ 
        Return a dictionary with task information. 
        Additionally we return a dict with key, values to be added to the main dictionary
    """

    task_cfg = edict()
    other_args = dict()
    task_cfg.NAMES = []
    task_cfg.NUM_OUTPUT = {}

    if 'include_semseg' in task_dictionary.keys() and task_dictionary['include_semseg']:
        tmp = 'semseg'
        task_cfg.NAMES.append('semseg')
        if db_name == 'nyuv2':
            task_cfg.NUM_OUTPUT[tmp] = 13
        elif db_name == 'cityscapes':
            task_cfg.NUM_OUTPUT[tmp] = 19
        else:
            raise NotImplementedError

    if 'include_depth' in task_dictionary.keys() and task_dictionary['include_depth']:
        tmp = 'depth'
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 1
        # Set effective depth evaluation range. Refer to:
        # https://github.com/sjsu-smart-lab/Self-supervised-Monocular-Trained-Depth-Estimation-using-Self-attention-and-Discrete-Disparity-Volum/blob/3c6f46ab03cfd424b677dfeb0c4a45d6269415a9/evaluate_city_depth.py#L55
        task_cfg.depth_max = 80.0
        task_cfg.depth_min = 0.

    if 'include_normals' in task_dictionary.keys() and task_dictionary['include_normals']:
        # Surface Normals 
        tmp = 'normals'
        assert(db_name in ['PASCALContext', 'NYUD', 'nyuv2', 'cityscapes', 'pascal'])
        task_cfg.NAMES.append(tmp)
        task_cfg.NUM_OUTPUT[tmp] = 3

    return task_cfg, other_args


def get_backbone(opt):
    if opt.backbone == 'resnet34':
        from models.resnet import resnet34
        backbone = resnet34(opt['backbone_kwargs']['pretrained'])
        backbone_channels = 512
    
    elif opt.backbone == 'resnet50':
        from models.resnet import resnet50
        backbone = resnet50(opt['backbone_kwargs']['pretrained'])
        backbone_channels = 2048

    elif opt.backbone == 'hrnet_w18':
        from models.hrnet import hrnet_w18
        backbone = hrnet_w18(opt['backbone_kwargs']['pretrained'])
        backbone_channels = [18, 36, 72, 144]

    else:
        raise NotImplementedError(f'{opt.backbone} not implimented')

    if 'fuse_hrnet' in opt['backbone_kwargs'] and opt['backbone_kwargs']['fuse_hrnet']: # Fuse the multi-scale HRNet features
        from models.hrnet import HighResolutionFuse
        backbone = torch.nn.Sequential(backbone, HighResolutionFuse(backbone_channels, 256))
        backbone_channels = sum(backbone_channels)

    return backbone, backbone_channels

def get_backbone_dims(opt):
    if opt.backbone == 'resnet34':
        if opt.train_db_name == 'NYUD':
            dims = [14,18]
        else:
            dims = [16,16]
    elif opt.backbone == 'hrnet_w18':
        dims = [opt.TRAIN.SCALE[0]//4, opt.TRAIN.SCALE[1]//4]
    else:
        raise NotImplementedError(f'{opt.backbone} not implimented')

    return dims

def get_head(opt, backbone_channels, task):
    """ Return the decoder head """

    if opt['head'] == 'deeplab':
        from models.aspp import DeepLabHead
        return DeepLabHead(backbone_channels, opt.TASKS.NUM_OUTPUT[task])

    elif opt['head'] == 'hrnet':
        from models.hrnet import HighResolutionHead
        return HighResolutionHead(backbone_channels, opt.TASKS.NUM_OUTPUT[task])

    else:
        raise NotImplementedError

def get_model(opt):

    if opt.setup == 'single_task':
        
        backbones = {}
        for task in opt.TASKS.NAMES:
            backbone, backbone_channels = get_backbone(opt)
            backbones[task] = backbone
        backbones = torch.nn.ModuleDict(backbones)
        
        heads = torch.nn.ModuleDict({task: get_head(opt, backbone_channels, task) for task in opt.TASKS.NAMES})

        from models.baseline import STL
        model = STL(opt, backbones, backbone_channels, heads)

    
    elif opt.setup == 'multi_task':
        
        backbone, backbone_channels = get_backbone(opt)

        if opt.model == 'hps':
            from models.baseline import HPS
            heads = torch.nn.ModuleDict({task: get_head(opt, backbone_channels, task) for task in opt.TASKS.NAMES})
            model = HPS(opt, backbone, backbone_channels, heads)

        elif opt.model == 'padnet':
            from models.padnet import PADNet
            model = PADNet(opt, backbone, backbone_channels)
        
        elif opt.model == 'papnet':
            from models.papnet import PAPNet
            model = PAPNet(opt, backbone, backbone_channels)
        
        elif opt.model == 'emanet':
            from models.emanet import EMANet
            backbone_dims = get_backbone_dims(opt)
            model = EMANet(opt, backbone, backbone_channels, backbone_dims)
        
        elif opt.model == 'm_emanet':
            from models.m_emanet import MEMANet
            backbone_dims = get_backbone_dims(opt)
            heads = torch.nn.ModuleDict({task: get_head(opt, backbone_channels, task) for task in opt.TASKS.NAMES})
            model = MEMANet(opt, backbone, backbone_channels, backbone_dims, heads)

        elif opt.model == 'mti_net':
            from models.mti_net import MTINet
            heads = torch.nn.ModuleDict({task: get_head(opt, backbone_channels, task) for task in opt.TASKS.NAMES})
            model = MTINet(opt, backbone, backbone_channels, heads)
        
        else:
            raise NotImplementedError(f'{opt.model} not implimeneted')
    
    else:
        raise NotImplementedError(f'Unknown setup: {opt.setup}')
    
    return model

""" 
    Loss functions 
"""

def get_criterion(opt):
    if 'loss_weighting' in opt['loss_kwargs']:
        loss_weighting = get_loss_weighting(opt)

    if opt['loss_kwargs']['loss_scheme'] == 'stl':
        from losses.loss_schemes import SingleTaskLoss
        return SingleTaskLoss(opt.TASKS.NAMES)

    elif opt['loss_kwargs']['loss_scheme'] == 'hps':
        from losses.loss_schemes import MultiTaskLoss
        return MultiTaskLoss(opt.TASKS.NAMES, loss_weighting)

    elif opt['loss_kwargs']['loss_scheme'] == 'padnet':
        from losses.loss_schemes import PADNetLoss
        return PADNetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, loss_weighting)
    
    elif opt['loss_kwargs']['loss_scheme'] == 'papnet':
        from losses.loss_schemes import PAPNetLoss
        return PAPNetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, loss_weighting)

    elif opt['loss_kwargs']['loss_scheme'] == 'emanet':
        from losses.loss_schemes import EMANetLoss
        return EMANetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, loss_weighting)
    
    elif opt['loss_kwargs']['loss_scheme'] == 'mti_net':
        from losses.loss_schemes import MTINetLoss
        return MTINetLoss(opt.TASKS.NAMES, opt.TASKS.NAMES, loss_weighting)

    else:
        raise NotImplementedError(f'{opt.loss_scheme} not implimented')
    
def get_loss_weighting(opt):
    if opt['loss_kwargs']['loss_weighting'] == 'scalarization':
        from losses.loss_weights import Scalarization
        return Scalarization(opt)
    
    elif opt['loss_kwargs']['loss_weighting'] == 'uncertainty':
        from losses.loss_weights import Uncertainty
        return Uncertainty(opt)

    else:
        raise NotImplementedError
    
def get_train_dataset(opt, transforms=None):
    """ Return the train dataset """

    db_name = opt['train_db_name']
    print('Preparing train dataset for db: {}'.format(db_name))
    
    if db_name == 'nyuv2':
        from data.create_dataset import NYUv2
        database = NYUv2(opt.db_paths['nyuv2'],
                         do_semseg='semseg' in opt.TASKS.NAMES,
                         do_depth='depth' in opt.TASKS.NAMES,
                         do_normal='normals' in opt.TASKS.NAMES,
                         train=True,
                         augmentation=opt.augmentation)
    
    elif db_name == 'cityscapes':
        if opt.augmentation:
            print('Applying data augmentation...')
        from data.create_dataset import CityScapes
        database = CityScapes(opt.db_paths['cityscapes'],
                         do_semseg='semseg' in opt.TASKS.NAMES,
                         do_depth='depth' in opt.TASKS.NAMES,
                         train=True,
                         augmentation=opt.augmentation)
    else:
        raise NotImplemented("train_db_name: Choose among (nyuv2, cityscapes)")
    
    return database

def get_test_dataset(opt, transforms=None):
    """ Return the test dataset """

    db_name = opt['val_db_name']
    print('Preparing test dataset for db: {}'.format(db_name))

    if db_name == 'nyuv2':
        from data.create_dataset import NYUv2
        database = NYUv2(opt.db_paths['nyuv2'],
                         do_semseg='semseg' in opt.TASKS.NAMES,
                         do_depth='depth' in opt.TASKS.NAMES,
                         do_normal='normals' in opt.TASKS.NAMES,
                         train=False,
                         augmentation=opt.augmentation)
    
    elif db_name == 'cityscapes':
        from data.create_dataset import CityScapes
        database = CityScapes(opt.db_paths['cityscapes'],
                         do_semseg='semseg' in opt.TASKS.NAMES,
                         do_depth='depth' in opt.TASKS.NAMES,
                         train=False,
                         augmentation=opt.augmentation)
        
    else:
        raise NotImplemented("test_db_name: Choose among (nyuv2, cityscapes)")

    return database
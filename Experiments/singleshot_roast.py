import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from FakeRoast import FakeRoastUtil_v2
from FakeRoast.FakeRoast import *
import pdb

def get_parameters(model):
    s = 0
    for p in model.parameters():
        if p.requires_grad:
            s+= p.numel()
            print(p.shape, p.numel())

    return s

def analyse(model):
    names = []
    pnames = []
    norms = []
    for name, module in model.named_modules():
        if name.endswith('WHelper'):
            continue

        if isinstance(module, FakeRoastConv2d) or isinstance(module, FakeRoastLinear):
            for pname, param in module.named_parameters():
                  if not param.requires_grad:
                    continue
                  if pname in ['WHelper.weight']:
                      norm = torch.norm(module.scale * module.WHelper.wt_comp_to_orig(param)).item()
                  else:
                      norm = torch.norm(param).item()
                  names.append(name)
                  pnames.append(pname)
                  norms.append(norm)

        else:
            for pname, param in module.named_parameters(recurse=False):
                  if not param.requires_grad:
                      continue
                  norm = torch.norm(param).item()
                  names.append(name)
                  pnames.append(pname)
                  norms.append(norm)

    return pd.DataFrame({"name" : names, "pname" : pnames, "norm": norms})

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                       args.pretrained)


    possible_params = get_parameters(model)

    sparsity = 10**(-float(args.compression))
    # roast if needed
    roaster = None
    if args.roast_mapper == "random":
        mapper_args = None
    else:
        mapper_args = { "mapper": args.roast_mapper, "hasher" : "uhash", "block_k" : 16, "block_n" : 16, "block": 8, "seed" : 1011}
    if args.use_global_roast:
        print("GLOBAL ROAST")
        roaster = FakeRoastUtil_v2.ModelRoasterGradScaler(model, True, sparsity, verbose=FakeRoastUtil_v2.NONE, 
                                                module_limit_size=args.module_limit_size, init_std=args.roast_init_std,
                                                scaler_mode=args.roast_scaler_mode, mapper_args=mapper_args)
        model = roaster.process()
        
    elif args.use_local_roast:
        print("LOCAL ROAST")
        roaster = FakeRoastUtil_v2.ModelRoaster(model, False, sparsity, verbose=FakeRoastUtil_v2.NONE,
                                                module_limit_size=args.module_limit_size, mapper_args=mapper_args)

        model = roaster.process()
    roasted_parameters = 0
    if roaster is not None:
        assert(possible_params == roaster.original_total_params)
        roasted_parameters = roaster.original_roastable_params
        

    print(model, flush=True)
    df = analyse(model)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    norm_df = df[df.pname != "roast_array"]
    print("full model norm", np.linalg.norm(norm_df.norm))

    if args.analyse_model:
        bef_df = analyse(model)
        model.load_state_dict(torch.load(args.analyse_model))
        aft_df = analyse(model)
        norm_df = bef_df.merge(aft_df, on=["name", "pname"], suffixes=('_bef', '_aft'))
        print(tabulate(norm_df, headers='keys', tablefmt='psql'))
        norm_df = norm_df[norm_df.pname != "roast_array"]
        print("full model norm", np.linalg.norm(norm_df.norm_bef), np.linalg.norm(norm_df.norm_aft))
        return

    total_params = get_parameters(model)
    if args.use_global_roast:
        print((total_params == int(sparsity*roasted_parameters) + (possible_params - roasted_parameters)))
    if args.use_local_roast:
        print(abs(total_params - int(sparsity*roasted_parameters) - (possible_params - roasted_parameters)) < 50)
    # parameter storage
    compression_result = pd.DataFrame({ "original" : possible_params,
                           "unroasted" : possible_params - roasted_parameters,
                           "roasted" : roasted_parameters,
                           "final" : total_params
                         }, index=[0])
    print(compression_result, flush=True)

    model = model.to(device)
    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)

    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    best_model_dict, post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                  test_loader, device, args.post_epochs, args.verbose, use_roast_scaler=args.use_roast_grad_scaler) 

    ## Display Results ##
    frames = [post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Post-Roast', 'Final'])
    print("Train results:\n", train_result)
    print("Parameter Compression: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        compression_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(best_model_dict,"{}/best_model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

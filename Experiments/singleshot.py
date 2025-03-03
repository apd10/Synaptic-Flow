import numpy as np
import pandas as pd
from tabulate import tabulate
import torch
import torch.nn as nn
from Utils import load
from Utils import generator
from Utils import metrics
from train import *
from prune import *
from Layers import layers

def analyse(model):
    names = []
    pnames = []
    norms = []
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            print(name, pname)
            if 'weight_mask' in dir(module) and pname == "weight":
                norm = torch.norm(param * module.weight_mask).item()
            else:
                norm = torch.norm(param).item()
            print(name, pname, param.shape, norm)
            names.append(name)
            pnames.append(pname)
            norms.append(norm)

    return pd.DataFrame({"name" : names, "pname" : pnames, "norm": norms})

def sparse_to_full(model, add_noise_to_zeros=-1):
    for name, module in model.named_modules():
        if type(module) in [layers.Linear]:
            module.weight.data[:,:] = module.weight * module.weight_mask 
            if add_noise_to_zeros > 0.0:
                module.weight.data[:,:] += torch.randn_like(module.weight) * (1 - module.weight_mask) * add_noise_to_zeros
            module.weight_mask.data[:,:] = torch.ones_like(module.weight_mask)
            if module.bias is not None:
                module.bias.data[:] = module.bias * module.bias_mask
                module.bias_mask.data[:] = torch.ones_like(module.bias_mask)
        if type(module) in [layers.Conv2d]:
            module.weight.data[:,:,:,:] = module.weight * module.weight_mask  
            if add_noise_to_zeros > 0.0:
                module.weight.data[:,:,:,:] += torch.randn_like(module.weight) * (1 - module.weight_mask) * add_noise_to_zeros
            module.weight_mask.data[:,:,:,:] = torch.ones_like(module.weight_mask)
            if module.bias is not None:
                module.bias.data[:] = module.bias * module.bias_mask
                module.bias_mask.data[:] = torch.ones_like(module.bias_mask)
    return model

def run(args):
    ## Random Seed and Device ##
    torch.manual_seed(args.seed)
    device = load.device(args.gpu)

    ## Data ##
    print('Loading {} dataset.'.format(args.dataset))
    input_shape, num_classes = load.dimension(args.dataset) 
    prune_loader = load.dataloader(args.dataset, args.prune_batch_size, True, args.workers, args.prune_dataset_ratio * num_classes)
    train_loader = load.dataloader(args.dataset, args.train_batch_size, True, args.workers)
    test_loader = load.dataloader(args.dataset, args.test_batch_size, False, args.workers)

    ## Model, Loss, Optimizer ##
    print('Creating {}-{} model.'.format(args.model_class, args.model))
    model = load.model(args.model, args.model_class)(input_shape, 
                                                     num_classes, 
                                                     args.dense_classifier, 
                                                     args.pretrained).to(device)
    print(model)
    print(flush=True)
    if args.analyse_model:
        bef_df = analyse(model)
        model.load_state_dict(torch.load(args.analyse_model))
        aft_df = analyse(model)
        norm_df = bef_df.merge(aft_df, on=["name", "pname"], suffixes=('_bef', '_aft'))
        print(tabulate(norm_df, headers='keys', tablefmt='psql'))
        print("full model norm (before, after)", np.linalg.norm(norm_df.norm_bef), np.linalg.norm(norm_df.norm_aft))
        return

    if args.sparse_full_fine_tune:
        model.load_state_dict(torch.load(args.sparse_full_fine_tune))
        model = sparse_to_full(model, args.add_noise_to_zeros)


    loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(args.optimizer)
    optimizer = opt_class(generator.parameters(model), lr=args.lr, weight_decay=args.weight_decay, **opt_kwargs)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_drops, gamma=args.lr_drop_rate)


    ## Pre-Train ##
    print('Pre-Train for {} epochs.'.format(args.pre_epochs))
    pre_model, pre_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                 test_loader, device, args.pre_epochs, args.verbose)

    ## Prune ##
    print('Pruning with {} for {} epochs.'.format(args.pruner, args.prune_epochs))
    pruner = load.pruner(args.pruner)(generator.masked_parameters(model, args.prune_bias, args.prune_batchnorm, args.prune_residual))
    sparsity = 10**(-float(args.compression))
    prune_loop(model, loss, pruner, prune_loader, device, sparsity, 
               args.compression_schedule, args.mask_scope, args.prune_epochs, args.reinitialize, args.prune_train_mode, args.shuffle, args.invert)

    # prune result#
    prune_result = metrics.summary(model, 
                                   pruner.scores,
                                   metrics.flop(model, input_shape, device),
                                   lambda p: generator.prunable(p, args.prune_batchnorm, args.prune_residual))
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    unpruned = prune_result[~prune_result.prunable]['size'].sum()
    pruned = prune_result[prune_result.prunable]['size'].sum()
    print( "Total", total_params, "Unpruned:", unpruned,  "Prunable:", pruned,  "After pruning:", possible_params)
    
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

    ## Post-Train ##
    print('Post-Training for {} epochs.'.format(args.post_epochs))
    best_model, post_result = train_eval_loop(model, loss, optimizer, scheduler, train_loader, 
                                  test_loader, device, args.post_epochs, args.verbose) 

    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

    ## Save Results and Model ##
    if args.save:
        print('Saving results.')
        pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
        post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
        prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
        torch.save(model.state_dict(),"{}/model.pt".format(args.result_dir))
        torch.save(best_model,"{}/best_model.pt".format(args.result_dir))
        torch.save(optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
        torch.save(scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))



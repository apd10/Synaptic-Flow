import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from Utils.manipulators import *
from FakeRoast.FakeRoastUtil_v2 import RoastGradScaler
from FakeRoast.FakeRoast import FakeRoast

def analyse_grads(model):
    print(" ================ analysing grad norms ===============")
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if not param.requires_grad:
                continue
            if isinstance(module, FakeRoast):
                grad = module.grad_comp_to_orig(param.grad)
                norm = torch.norm(grad).item()
            else:
                grad = param.grad
                norm = torch.norm(grad).item()
            print(name, pname, grad.shape, norm)

def train(model, loss, optimizer, dataloader, device, epoch, verbose, log_interval=10, use_roast_scaler=False):
    model.train()
    total = 0
    for batch_idx, (data, target) in enumerate(dataloader):
        if verbose:
            print(batch_idx, "train loop")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        train_loss = loss(output, target)
        total += train_loss.item() * data.size(0)
        train_loss.backward()
        if verbose & (batch_idx  < 1):
            grads = []
            for p in model.parameters():
                if p.requires_grad:
                    grads.append(torch.norm(p.grad).item() if (p.grad is not None) else 0)
            print("grad: ", grads, flush=True)

        if verbose:
            analyse_grads(model)
        if use_roast_scaler:
            RoastGradScaler().scale_step(model)
        if verbose:
            analyse_grads(model)
        optimizer.step()
        if verbose & (batch_idx % log_interval == 0):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(dataloader.dataset),
                100. * batch_idx / len(dataloader), train_loss.item()))
    return total / len(dataloader.dataset)

def eval(model, loss, dataloader, device, verbose, tag=""):
    model.eval()
    total = 0
    correct1 = 0
    correct5 = 0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total += loss(output, target).item() * data.size(0)
            _, pred = output.topk(5, dim=1)
            correct = pred.eq(target.view(-1, 1).expand_as(pred))
            correct1 += correct[:,:1].sum().item()
            correct5 += correct[:,:5].sum().item()
    average_loss = total / len(dataloader.dataset)
    accuracy1 = 100. * correct1 / len(dataloader.dataset)
    accuracy5 = 100. * correct5 / len(dataloader.dataset)
    if verbose:
        print('(',tag,')','Evaluation: Average loss: {:.4f}, Top 1 Accuracy: {}/{} ({:.2f}%)'.format(
            average_loss, correct1, len(dataloader.dataset), accuracy1))
    return average_loss, accuracy1, accuracy5

def train_eval_loop(model, loss, optimizer, scheduler, train_loader, test_loader, device, epochs, verbose, do_validation=False, use_roast_scaler=False):
    # split train dataloader into validation
    validation_loader = None
    if do_validation:
        train_loader, validation_loader = split_dataloader(train_loader, 0.95, 101)
        print("TRAIN", len(train_loader.dataset))
        print("VAL", len(validation_loader.dataset))
    
    val_loss, val_accuracy1, val_accuracy5 = -1,-1,-1
    if do_validation:
        val_loss, val_accuracy1, val_accuracy5 = eval(model, loss, validation_loader, device, verbose, 'val')
    test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose, 'test')

    columns = ['train_loss',  'val_loss', 'val1_accuracy', 'val5_accuracy', 'test_loss', 'top1_accuracy', 'top5_accuracy']
    rows = [[np.nan, val_loss, val_accuracy1, val_accuracy5, test_loss, accuracy1, accuracy5]]
    prev_lr = scheduler.get_last_lr()
    best_val_acc = 0
    model_state_dict = None
    for epoch in range(epochs):
        lr = scheduler.get_last_lr()
        if lr != prev_lr:
            if verbose:
                print("epoch:", epoch, "LR change", prev_lr, "-->", lr, flush=True)
            prev_lr = lr
            
        train_loss = train(model, loss, optimizer, train_loader, device, epoch, verbose, use_roast_scaler=use_roast_scaler)
        if do_validation:
            val_loss, val_accuracy1, val_accuracy5 = eval(model, loss, validation_loader, device, verbose, 'val')
        test_loss, accuracy1, accuracy5 = eval(model, loss, test_loader, device, verbose, 'test')
        row = [train_loss, val_loss, val_accuracy1, val_accuracy5, test_loss, accuracy1, accuracy5]
        scheduler.step()
        rows.append(row)
        print("test  ep {:3d} it {:3d} loss {:.3f} acc {:.3f}%".format(epoch+1, 0, test_loss, accuracy1))

        if val_accuracy1 > best_val_acc:
            best_val_acc = val_accuracy1
            model_state_dict = copy.deepcopy(model.state_dict())
            if verbose:
                print("model update @acc ", best_val_acc)
                print(columns)
                print(row)
    return model_state_dict, pd.DataFrame(rows, columns=columns)




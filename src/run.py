#
# Copyright 2023- IBM Inc. All rights reserved
# SPDX-License-Identifier: Apache2.0
#
import torch
import torch.optim as optim
from torch import nn
from sklearn.metrics import accuracy_score

from src.sparsify import get_pruning_rate, prune_weights
from src.penalty import count_terminal_conditions
from src.util import load_checkpoint, save_checkpoint, compute_metrics
from src.best_model_strategy import is_best_model


def validate(val_loader, model, criterion):
    model.eval()
    out, y_all = [], []

    with torch.no_grad():
        for batch, (x, y) in enumerate(val_loader):
            output = model(x)
            out.append(output.detach())
            y_all.append(y)

    output = torch.cat(out)
    output = output >= 0.5
    y = torch.cat(y_all)
    loss = criterion(output, y)
    acc = accuracy_score(y, output)
    return acc, loss.item()


def train(model, train_loader, val_loader, epochs, pruning=False, verbose=False, path_checkpoint='checkpoint.pt',
          keep_loss=[]):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(params=model.parameters(), lr=0.1, amsgrad=False)

    nbr_batches = train_loader.batch_sampler.num_batches
    iteration = 0
    if pruning:
        tot_iterations = (epochs - pruning.start) * nbr_batches

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            iteration = iteration + 1

            x, y = data

            if pruning:
                if epoch >= pruning.start:
                    if epoch == pruning.start and i == 0:
                        iteration = 1
                    if iteration % pruning.freq == 0:
                        pruning_rate = get_pruning_rate(pruning.rate, iteration, tot_iterations)
                        prune_weights(model, pruning_rate)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward pass
            outputs = model(x)
            # compute the loss
            penalty = count_terminal_conditions(model)
            loss = criterion(outputs, y) + 1e-5 * penalty
            # backward pass
            loss.backward()
            # optimize the model
            optimizer.step()

            running_loss += loss.item()

        training_avg_loss = running_loss / (i + 1)
        # Validate
        val_acc, val_loss = validate(val_loader, model, criterion)
        keep_loss.append(val_loss)

        # Keep best model
        if epoch == 0:
            is_best = True
            best_metrics = (val_acc, val_loss, penalty)
        else:
            is_best = is_best_model((val_acc, val_loss, penalty), best_metrics)

        if is_best:
            best_metrics = (val_acc, val_loss, penalty)
            save_checkpoint({
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'best_acc': val_acc,
                'best_loss': val_loss,
                'best_penalty': penalty,
                'optimizer': optimizer.state_dict()
            }, path=path_checkpoint)

        if verbose:
            print(
                f'[{epoch + 1}] train loss: {training_avg_loss:.9f}, val loss: {val_loss:.9f} | best val acc: {best_metrics[0]:.2f}, best penalty: {best_metrics[2]:.2f}',
                end='\r')

    print()
    load_checkpoint(model, optimizer, verbose=verbose, path=path_checkpoint)


def test(model, test_loader):
    labels = [0.0, 1.0]
    out, y_all = [], []
    with torch.no_grad():
        model.eval()
        for data in test_loader:
            x, y = data
            output = model(x)
            out.append(output.detach())
            y_all.append(y.squeeze(-1))
    y_pred = torch.cat(out).numpy().round()
    y_test = torch.cat(y_all).numpy()
    res = compute_metrics(y_test, y_pred, labels)
    print(
        f'network test accuracy: {res["acc"] * 100:.1f}%, balanced: {res["bal_acc"] * 100:.1f}% (TP:{res["tp"]}, FN:{res["fn"]}, FP:{res["fp"]}, TN:{res["tn"]})')


def test_rule(rule, test_loader):
    labels = [0.0, 1.0]
    x_test = list(map(lambda data: data[0].numpy(), test_loader.dataset))
    y_test = list(map(lambda data: data[1].item(), test_loader.dataset))

    y_pred, rules_triggered = rule.batch_apply_rule(x_test)
    y_pred = list(map(float, y_pred))
    res = compute_metrics(y_test, y_pred, labels)
    print(
        f'rule test accuracy: {res["acc"] * 100:.1f}%, balanced: {res["bal_acc"] * 100:.1f}% (TP:{res["tp"]}, FN:{res["fn"]}, FP:{res["fp"]}, TN:{res["tn"]})')

    rule_metrics = dict()
    for rule in set(rules_triggered):
        ind = [index for index, element in enumerate(rules_triggered) if element == rule]
        pred = [y_pred[instance] for instance in ind]
        true = [y_test[instance] for instance in ind]
        rule_metrics[str(rule)] = compute_metrics(true, pred, labels)
    occ_rules = dict((str(x), rules_triggered.count(x)) for x in set(rules_triggered))
    for rule, value in rule_metrics.items():
        print(
            f'  Expr {rule:<5} - triggered {occ_rules[str(rule)]:<4} - acc: {value["acc"] * 100:.1f}% (TP:{value["tp"]}, FN:{value["fn"]}, FP:{value["fp"]}, TN:{value["tn"]})')

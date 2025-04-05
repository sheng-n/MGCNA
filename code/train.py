import copy
import time
import torch
import numpy as np
import torch.nn as nn
from layer import MGCNA
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve, average_precision_score, f1_score, auc


def train_model(dataset, train_loader, test_loader, args):

    model = MGCNA(feature=args.dimensions, hidden1 = args.hidden1 , hidden2 = args.hidden2, decoder1 = args.decoder1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    m = torch.nn.Sigmoid()
    loss_node = torch.nn.BCELoss()
    if args.cuda:
        model.to("cuda")


    # Train model
    t_total = time.time()
    print('Start Training...')

    for epoch in range(args.epochs):

        t = time.time()
        print('-------- Epoch ' + str(epoch + 1) + ' --------')
        y_pred_train = []
        y_label_train = []

        for i, (label, inp) in enumerate(train_loader):

            if args.cuda:
                label = label.cuda()

            model.train()
            optimizer.zero_grad()
            output = model(dataset, inp)

            log = torch.squeeze(m(output))
            loss_train = loss_node(log, label.float())
            loss_train.backward()
            optimizer.step()

            # all_loss += loss_train
            label_ids = label.to('cpu').numpy()
            y_label_train = y_label_train + label_ids.flatten().tolist()
            y_pred_train = y_pred_train + log.flatten().tolist()

            if i % 100 == 0:  #
                print('epoch: ' + str(epoch + 1) + '/ iteration: ' + str(i + 1) + '/ loss_train: ' + str(
                    loss_train.cpu().detach().numpy()))

        roc_train = roc_auc_score(y_label_train, y_pred_train)  #
        print('epoch: {:04d}'.format(epoch + 1),
                  # 'epoch_loss: {:.4f}'.format(average_loss.item()),
                  'loss_train: {:.4f}'.format(loss_train.item()),
                  'auroc_train: {:.4f}'.format(roc_train),
                  'time: {:.4f}s'.format(time.time() - t))

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Testing
    auroc_test, prc_test, f1_test, loss_test = test(model, test_loader, dataset, args)
    print('loss_test: {:.4f}'.format(loss_test.item()), 'auroc_test: {:.4f}'.format(auroc_test),
          'auprc_test: {:.4f}'.format(prc_test), 'f1_test: {:.4f}'.format(f1_test))


def test(model, loader, dataset, args):
    m = torch.nn.Sigmoid()
    loss_node = torch.nn.BCELoss()

    model.eval()
    y_pred = []
    y_label = []

    with torch.no_grad():
        for i, (label, inp) in enumerate(loader):
            if args.cuda:
                label = label.cuda()

            output = model(dataset, inp)

            log = torch.squeeze(m(output))
            loss = loss_node(log, label.float())

            label_ids = label.to('cpu').numpy()
            y_label = y_label + label_ids.flatten().tolist()
            y_pred = y_pred + log.flatten().tolist()
            outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])

    # AUC
    fpr, tpr, auc_thresholds = roc_curve(y_label, y_pred)
    auc_score = auc(fpr, tpr)

    # AUCPR
    precision, recall, pr_threshods = precision_recall_curve(y_label, y_pred)
    aupr_score = auc(recall, precision)
    print('auroc_test: {:.4f}'.format(auc_score),'auprc_test: {:.4f}'.format(aupr_score))

    return auc_score, aupr_score, f1_score(y_label, outputs), loss


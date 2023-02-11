import os
import shutil
import warnings
from datetime import datetime

import numpy as np
import torch
from torch import optim

import Nets
import config
import dataset
import utils
import wandb
from train import Trainer

wandb.init(project="Ensemble", entity=" ")

warnings.filterwarnings("ignore")


def main():
    args = config.parse_args()
    wandb.config.update(args)

    if torch.cuda.is_available():
        device = args.device
    else:
        device = 'cpu'

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    now_time = datetime.now()
    now_time_str = str(now_time.month) + str(now_time.day) + str(now_time.hour) + str(now_time.minute)

    model_path = os.path.join(
        args.save,
        args.cell_line +
        '_' + now_time_str + '.pt')

    train_dataset = dataset.LDAMLncAtlasDataset(
        args.train_CNRCI, args.train_dataset)
    dev_dataset = dataset.LDAMLncAtlasDataset(args.dev_CNRCI, args.dev_dataset)

    model = Nets.en_model(args)

    if device != 'cpu':
        model.cuda(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay)

    best_fscore = 0

    model.initialize()

    max_m = args.max_m
    s = args.s

    cls_num_list = train_dataset.get_cls_num_list()
    for epoch in range(args.epochs):
        idx = epoch // 99
        betas = [0, 0.9999]
        effective_num = 1.0 - np.power(betas[idx], cls_num_list)
        per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
        per_cls_weights = per_cls_weights / \
                          np.sum(per_cls_weights) * len(cls_num_list)
        if device != 'cpu':
            per_cls_weights = torch.FloatTensor(per_cls_weights).cuda(device)
        else:
            per_cls_weights = torch.FloatTensor(per_cls_weights)

        criterion = Nets.LDAMLoss(
            cls_num_list=cls_num_list,
            max_m=0.5,
            s=30,
            weight=per_cls_weights,
            device=args.device).cuda(device)

        trainer = Trainer(args, model, criterion, optimizer, device)

        _ = trainer.train(train_dataset)

        train_loss, train_preds, train_labels = trainer.test(train_dataset)

        train_preds_CNRCI = train_preds
        train_labels_CNRCI = train_labels

        train_accuracy, train_precision, train_recall, train_roc_auc, train_report, train_support, train_report_dict = \
            utils.evaluate_metrics_sklearn(
                train_preds_CNRCI.numpy(),
                train_labels_CNRCI.numpy())

        dev_loss, dev_preds, dev_labels = trainer.test(dev_dataset)

        dev_preds_CNRCI = dev_preds
        dev_labels_CNRCI = dev_labels

        dev_accuracy, dev_precision, dev_recall, dev_roc_auc, dev_report, dev_support, dev_report_dict = \
            utils.evaluate_metrics_sklearn(
                dev_preds_CNRCI.numpy(), dev_labels_CNRCI.numpy())


        fscore = (dev_report_dict['1']['f1-score'] +
                  dev_report_dict['0']['f1-score']) / 2

        wandb.log({"dev_AUC": dev_roc_auc, "dev_ACC": dev_accuracy,"dev_fscore": fscore,
                   "dev_class0_recall": dev_report_dict['0']['recall'],
                   "dev_class0_precision": dev_report_dict['0']['precision'],
                   "dev_class0_f1score": dev_report_dict['0']['f1-score'],
                   "dev_class1_recall": dev_report_dict['1']['recall'],
                   "dev_class1_precision": dev_report_dict['1']['precision'],
                   "dev_class1_f1score": dev_report_dict['1']['f1-score'],
                   "epoch": epoch + 1})

        roc_auc, bacc, auprc, mcc, f1 = utils.evaluate_metrics_sklearn_new(
            dev_preds_CNRCI.numpy(), dev_labels_CNRCI.numpy())

        wandb.log({"roc_auc": roc_auc, "bacc": bacc, "auprc": auprc, "mcc": mcc, "f1": f1, "epoch": epoch + 1})

        if best_fscore < fscore:
            best_fscore = fscore
            wandb.run.summary["best_roc_auc"] = roc_auc
            wandb.run.summary["best_bacc"] = bacc
            wandb.run.summary["best_auprc"] = auprc
            wandb.run.summary["best_mcc"] = mcc
            wandb.run.summary["best_f1"] = f1
            wandb.run.summary["max_m"] = max_m
            wandb.run.summary["s"] = s

            torch.save(model.state_dict(), model_path)


if __name__ == "__main__":
    main()

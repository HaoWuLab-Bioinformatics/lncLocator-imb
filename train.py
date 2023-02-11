import logging
import numpy
import torch
from torch.utils.data import DataLoader
from torchsampler import ImbalancedDatasetSampler
import utils


class Trainer(object):
    def __init__(self, args, model, criterion, optimizer, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.epoch = 0
        logging.info("Trainer informations:")
        logging.info("Model information:\n{}".format(model))
        logging.info("Optimizer information:\n{}".format(optimizer))
        logging.info("Device information:\n{}".format(device))
        model_paras = ""
        paras = model.named_parameters()
        for name, para in paras:
            if para.requires_grad:
                model_paras += "{}:{}\n".format(name, para.size())
        logging.info("Parameters information:\n{}".format(model_paras))

    def train(self, dataset):
        logging.info("Start training...")
        self.model.train()
        self.optimizer.zero_grad()
        total_loss = 0.0

        if self.args.ImbalancedDatasetSampler == True:
            sampler = ImbalancedDatasetSampler(dataset)

            dataloader = DataLoader(dataset,
                                    self.args.batchsize,
                                    collate_fn=utils.collate_fn,
                                    sampler=sampler)
        else:
            dataloader = DataLoader(dataset,
                                    self.args.batchsize,
                                    shuffle=True,
                                    collate_fn=utils.collate_fn)

        num = 0
        for idx, (code, feature, label) in enumerate(dataloader):
            CNRCI_label = torch.tensor(label).to(self.device)

            self.optimizer.zero_grad()

            input_code = [utils.seq_to_tensor(
                i,
                k=self.args.k,
                stride=self.args.stride) for i in code]

            feature = torch.tensor(
                numpy.array(
                    feature,
                    dtype=float)).to(
                torch.float32)

            feature = feature.to(self.device)

            CNRCI_pred = self.model(input_code, feature)

            class_label = (CNRCI_label > 0).to(torch.int64)

            loss = self.criterion(CNRCI_pred, class_label)
            # loss = self.criterion(CNRCI_pred[:, 1].to(torch.float), class_label.to(torch.float))

            total_loss += loss.item()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5)

            self.optimizer.step()

            num = idx
        self.epoch += 1
        return total_loss / num

    def test(self, dataset):

        self.model.eval()

        batchsize = self.args.test_batchsize

        with torch.no_grad():
            total_loss = 0

            CNRCI_predictions = \
                torch.zeros(len(dataset), dtype=torch.float, device='cpu')
            CNRCI_ground_truths = \
                torch.zeros(len(dataset), dtype=torch.float, device='cpu')

            dataloader = DataLoader(dataset,
                                    self.args.test_batchsize,
                                    shuffle=False,
                                    collate_fn=utils.collate_fn)

            # Start test
            for idx, (code, feature, label) in enumerate(dataloader):
                # Move the labels to gpu
                CNRCI_label = torch.tensor(label).to(self.device)

                input_code = [utils.seq_to_tensor(
                    i,
                    k=self.args.k,
                    stride=self.args.stride) for i in code]

                # Record the labels
                batchsize_real = CNRCI_label.shape[0]

                class_label = (CNRCI_label > 0).to(torch.int64)

                feature = torch.tensor(
                    numpy.array(
                        feature,
                        dtype=float)).to(
                    torch.float32)

                feature = feature.to(self.device)

                # Get the predictions from model
                CNRCI_pred = self.model(input_code, feature)

                loss = self.criterion(CNRCI_pred, class_label)
                # loss = self.criterion(CNRCI_pred[:, 1].to(torch.float), class_label.to(torch.float))

                # Record the total loss
                total_loss += loss.item() * CNRCI_pred.shape[0]

                CNRCI_pred = CNRCI_pred[:, 1]

                # Record the predictions
                CNRCI_predictions[idx * batchsize:
                                  idx * batchsize + batchsize_real] = \
                    CNRCI_pred.to('cpu').squeeze()

                CNRCI_ground_truths[idx * batchsize:
                                    idx * batchsize + batchsize_real] = \
                    class_label.to('cpu').squeeze()

        return total_loss / \
            len(dataset), CNRCI_predictions, CNRCI_ground_truths
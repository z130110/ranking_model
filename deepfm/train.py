import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import roc_auc_score
import os
import psutil
from time import time
import datetime
from logging import getLogger
from importlib import import_module
import json

logger = getLogger("my logger")
#torch.set_printoptions(precision = 3, sci_mode = False)

class trainer(object):
    def __init__(self, model, arg_parser, train_batch, dev_batch, torch_device):
        self.model = model
        self.arg_parser = arg_parser
        self.lr = arg_parser.lr
        self.lr_decay = arg_parser.lr_decay
        self.lr_patience = arg_parser.lr_patience
        self.init_method = arg_parser.init_method
        self.torch_device = torch_device
        self.train_batch = train_batch
        self.dev_batch = dev_batch
        self.iter_patience = 0
        self.epoch_patience = 0
        self.arg_parser.EarlyStopping = arg_parser.EarlyStopping
        self.json_plot = {"train_epoch_loss":[], "train_epoch_auc":[], "dev_loss":[], "dev_auc":[]}
        self.monitor_dev = arg_parser.lr_monitor # "dev_auc" # options: "dev_auc", "auc_loss", "scheduler"
        self.EarlyStopping_triggered = False
        if not arg_parser.from_pretrained:
            logger.info("New model builded, save initilized checkpoint to the path: " + self.arg_parser.checkpoint_save)
            logger.info("save initilized model to the path: " + self.arg_parser.model_save)
            self.init_weights_()
            checkpoint = {"model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": self.build_optimizer().state_dict(),
                           "model_param_dict": self.arg_parser.model_param,
                           "epoch": 0,
                           "dev_loss": 1,
                           "dev_auc": 0.0}
            torch.save(checkpoint, self.arg_parser.checkpoint_save)
            torch.save(self.model, self.arg_parser.model_save)


    def init_weights_(self):
        init_list = ["uniform_", "normal_", "xavier_uniform_", "xavier_normal_", "kaiming_uniform_", \
                    "kaiming_normal_", "orthogonal_"]
        for name, weight in self.model.named_parameters():
            if "bn" not in name :
                #print(name,weight.shape)
                if "weight" in name:
                    getattr(nn.init,self.init_method)(weight)
                elif "bias" in name:
                    getattr(nn.init,"constant_")(weight, 0.0)
                else:
                    pass


    def build_optimizer(self):
        model_optimizer = getattr(torch.optim, self.arg_parser.optimizer)(self.model.parameters(), lr = self.lr)
        if self.arg_parser.from_pretrained:
            logger.info("load optimizer from pretrained checkpoint.")
            model_optimizer.load_state_dict(self.arg_parser.checkpoint_loaded["optimizer_state_dict"])
        return model_optimizer


    def train_(self, optimizer):
        batch_size = self.arg_parser.batch_size_train
        epochs = self.arg_parser.num_epochs
        self.model.train()
        logger.info("start to train")
        logger.info(f"number of train batch iters: {len(self.train_batch)}")
        # Adadelta, Adagrad, Adam, AdamW, SGD, SparseAdam, Adamax, ASGD, RMSprop, LBFGS, Rprop
        best_dev_loss = float("inf")
        best_dev_auc = float("-inf")
        best_dev_accuracy = 0
        train_loss_info = {}        # collection loss data to draw the loss curve
        train_loss_info["num_epochs"] = epochs
        train_loss_info["batch_size"] = batch_size
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = "min", factor = self.lr_decay, patience = self.lr_patience, verbose = True, min_lr = 0.000001)
        for epoch in range(epochs):
            last_dev_loss = float("inf") if epoch == 0 else dev_loss
            last_dev_auc = 0 if epoch == 0 else dev_auc
            epoch_train_loss_total = 0;
            num_batch_iter = len(self.train_batch)
            score_list = []
            label_list = []

            if self.arg_parser.shuffle_before_epoch == True:
               self.train_batch.shuffle_data()
            else:
               pass

            for i, minibatch in enumerate(self.train_batch):
                train_t = time()
                input_ = minibatch[0].to(self.torch_device)
                label_ = minibatch[1].to(self.torch_device)
                output_ = self.model(input_)
                self.model.zero_grad()
                loss = F.binary_cross_entropy(output_, label_.float())
                loss.backward()
                optimizer.step()
                scores = output_.detach().cpu().numpy().flatten().tolist()
                score_list += scores
                labels = label_.cpu().numpy().flatten().tolist()
                label_list += labels
                epoch_train_loss_total += loss.cpu().item()

            train_loss = round((epoch_train_loss_total / num_batch_iter), 7)
            train_auc_all = round(roc_auc_score(label_list, score_list), 7)
            dev_loss, dev_auc = self.evaluation(self.model, self.dev_batch)
            logger.info(f"{epoch}th epoch finished, train loss: {train_loss}, train auc: {train_auc_all}, val loss: {dev_loss}, val auc: {dev_auc}")
            self.json_plot["train_epoch_loss"].append(train_loss)
            self.json_plot["train_epoch_auc"].append(train_auc_all)
            self.json_plot["dev_loss"].append(dev_loss)
            self.json_plot["dev_auc"].append(dev_auc)

            checkpoint = {"model_state_dict": self.model.state_dict(),
                           "optimizer_state_dict": optimizer.state_dict(),
                           "model_param_dict": self.arg_parser.model_param,
                           "epoch":epoch,
                           "dev_loss": best_dev_loss,
                           "dev_auc": dev_auc}

            if self.arg_parser.lr_reduce:
                if self.monitor_dev == "dev_loss":
                    scheduler.step(dev_loss)
                    if dev_loss < best_dev_loss:
                        best_dev_loss = dev_loss
                        torch.save(checkpoint, self.arg_parser.checkpoint_save)
                        torch.save(self.model, self.arg_parser.model_save)
                    else:
                        self.iter_patience += 1
                elif self.monitor_dev == "dev_auc":
                    scheduler.step(dev_auc)
                    if dev_auc > best_dev_auc:
                        best_dev_auc = dev_auc
                        torch.save(checkpoint, self.arg_parser.checkpoint_save)
                        torch.save(self.model, self.arg_parser.model_save)
                    else:
                        self.iter_patience += 1
                else:
                    torch.save(checkpoint, self.arg_parser.checkpoint_save)
                    torch.save(self.model, self.arg_parser.model_save)


            self.EarlyStopping(dev_loss, last_dev_loss, dev_auc, last_dev_auc)
            if self.EarlyStopping_triggered == True:
                logger.info("=" * 70)
                logger.info(f"Early Stopping triggered after {epoch + 1} epoches, calculating test accuracy...")
                with open(self.arg_parser.result_dir + "/json_plot.json", "w") as json_f:
                    json.dump(self.json_plot, json_f)
                return None

        logger.info("Training fnished, full epoch, calculating test accuracy...")
        with open(self.arg_parser.result_dir + "/json_plot.json", "w") as json_f:
            json.dump(self.json_plot, json_f)

    def EarlyStopping(self, cur_dev_loss, last_dev_loss, cur_dev_auc, last_dev_auc):
        if self.monitor_dev == "dev_loss":
            self.epoch_patience += 1 if cur_dev_loss >= last_dev_loss else 0
        if self.monitor_dev == "dev_auc":
            self.epoch_patience += 1 if cur_dev_auc <= last_dev_auc else 0
        if self.epoch_patience == self.arg_parser.num_patience and self.arg_parser.EarlyStopping == True:
            self.EarlyStopping_triggered = True
            # if not self.save_best_model:
            #     torch.save(self.model.state_dict(), self.arg_parser.checkpoint_save)
            #     torch.save(self.model, self.arg_parser.model_save)


    def evaluation(self, model, batches, test = False):
        model.eval()
        loss = 0
        #label_all = np.array([], dtype = int)
        label_all = []
        socre_list = []
        #predict_all = np.array([], dtype = int)
        #score_list = []
        with torch.no_grad():
            for i, patch in enumerate(batches):
                input_ = patch[0].to(self.torch_device)
                label_ = patch[1].to(self.torch_device)
                output_ = model(input_)
                loss += F.binary_cross_entropy(output_, label_.float()).cpu()
                scores = output_.cpu().numpy().flatten().tolist()
                #label_all = np.append(label_all, label_.cpu().numpy())
                label_all += label_.cpu().numpy().flatten().tolist()
                socre_list += scores
                #predict_all = np.append(predict_all, predict.numpy())
        loss = loss / len(batches)
        loss = round(loss.item(), 7)
        auc = round(roc_auc_score(label_all, socre_list), 7)
        return loss, auc


    def evaluate_test(self, test_batch):
        model_checkpoint = torch.load(self.arg_parser.checkpoint_save, map_location = self.torch_device)
        model = self.model
        model.load_state_dict(model_checkpoint["model_state_dict"])
        test_loss, test_auc = self.evaluation(model, test_batch, test = True)
        logger.info(f"test loss:{test_loss}, test auc: {test_auc} .")

        logger.info(f"Saving onnx model...")
        Arsenal_model = ArsenalInput().to(self.torch_device)
        model_onnx = nn.Sequential(Arsenal_model, model).to(self.torch_device)
        model_onnx.eval()

        for minibatch in test_batch:
            onnx_input = minibatch[0].to(self.torch_device)
            if self.arg_parser.model == "mlp":
                onnx_input = onnx_input.float()

            torch.onnx.export(model_onnx, {"input": onnx_input}, self.arg_parser.result_dir + "/onnx_model.onnx", \
                               verbose = True, input_names = ["input"], output_names = ["score"], \
                                 dynamic_axes = {"input":{0: "batch_size"}, "score":{0: "batch_size"}})
            break


class ArsenalInput(nn.Module):
    def forward(self, x):
        x = x['input']
        return x

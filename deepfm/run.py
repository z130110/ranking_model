import torch
import argparse
from torch import nn
from importlib import import_module
from train import trainer
from logger import create_logger
from loss_plot import plot_curve
import numpy as np
import datetime
import os
import ast
import json
from torch.utils.data import DataLoader
from preprocess import processor, build_batch_iter
import psutil
import gc


parser = argparse.ArgumentParser(description = "ranking model deepfm v25")
parser.add_argument("--model", type = str, default = "DeepFM", help = "select a model")
parser.add_argument("--num_class", type = int, default = 2)
parser.add_argument("--valid_split", type = float, default = 0.1, help = "split validation set")
parser.add_argument("--num_dnn_layer", type = int, default = 4)
parser.add_argument("--hidden_size", type = int, default = 128, help = "hidden dimension when using RNN cells")
parser.add_argument("--dropout_rate", type = float, default = 0.5)
parser.add_argument("--embed_dim", type = int, default = 8)
parser.add_argument("--device", type = str, default = "GPU", help = "GPU or CPU")
parser.add_argument("--batch_size_train", type = int, default = 500000, help = "batch size of training")
parser.add_argument("--batch_size_dev", type = int, default = 100000, help = "batch size of validation")
parser.add_argument("--num_epochs", type = int, default = 500, help = "batch size")
parser.add_argument("--num_patience", type = int, default = 50, help = "number of times to trigger the early stopping")
parser.add_argument("--lr", type = float, default = 0.001, help = "learning rate")
parser.add_argument("--lr_decay", type = float, default = 0.8, help = "learning rate")
parser.add_argument("--lr_patience", type = float, default = 5, help = "shrinking the lr if no update")
parser.add_argument("--lr_monitor", type = str, default = "dev_auc", help = "learning rate monitoring indicator")
parser.add_argument("--optimizer", type = str, default = "Adam", help = "Optimizer's name, must be method of torch.optim")
parser.add_argument("--init_method", type = str, default = "xavier_uniform_", help = "Method which initilize the net'weights ")
parser.add_argument("--random_seed", type = int, default = 0, help = "torch's random seed")
parser.add_argument("--from_pretrained", type = ast.literal_eval, default = False, help = "if use pretrained model, give the path to the argument")
parser.add_argument("--transfer_learning", type = ast.literal_eval, default = False, help = "transfer learn new data every day or not.")
parser.add_argument("--use_bn", type = ast.literal_eval, default = True, help = "use 1d batch normalization or not")
parser.add_argument("--use_dnn", type = ast.literal_eval, default = True, help = "use dnn or not")
parser.add_argument("--use_fm", type = ast.literal_eval, default = True, help = "use fm or not")
parser.add_argument("--fm_dense_cross", type = ast.literal_eval, default = False, help = " dense part into FM or not")
parser.add_argument("--lr_reduce", type = ast.literal_eval, default = True, help = "EarlyStopping triggered or not")
parser.add_argument("--EarlyStopping", type = ast.literal_eval, default = True, help = "EarlyStopping triggered or not")
parser.add_argument("--test_run", type = ast.literal_eval, default = False, help = "Run test mode")
parser.add_argument("--shuffle_before_epoch", type = ast.literal_eval, default = True, help = "Shuffle data before each epoch")
args = parser.parse_args()


if __name__ == "__main__":
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    np.set_printoptions(suppress=True)
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #torch_device = torch.device('cpu')
    args.torch_device = torch_device
    torch.set_printoptions(precision = 5, threshold =2000, edgeitems = 5, linewidth = 120, sci_mode = False)
    date_dir = datetime.datetime.now().strftime("%Y_%m%d_%H%M%S/")
    result_dir = "train_result/" + date_dir
    args.result_dir = result_dir
    if os.path.exists(result_dir) == False:
        os.makedirs(result_dir)
    args.checkpoint_save = result_dir + "model_checkpoint.pkl"   # torch model's path
    args.model_save = result_dir + "torch_model.pth"
    log_save_path = result_dir + "train_log.log"
    logger_ = create_logger(log_save_path)
    logger_.info('\n'.join('%s: %s' % (k, str(v)) for k, v in dict(vars(args)).items()))
    logger_.info("Task PID code: " + str(os.getpid()))
    logger_.info("torch model and log file saved directory:")
    logger_.info(os.getcwd() + "/train_result/" + date_dir)

    process_module = processor(args)
    if args.test_run == True:
        train_input_np, train_labels_np = process_module.test_run_train()
        dev_input_np, dev_labels_np = process_module.test_run_dev()
    else:
        train_input_np, train_labels_np = process_module.get_train_data()
        dev_input_np, dev_labels_np = process_module.get_dev_data()


    args.input_dim = train_input_np.shape[1]
    args.sparse_col_index_id = process_module.sparse_col_index_id
    args.sparse_col_index_nonid = process_module.sparse_col_index_nonid
    args.dense_col_index = process_module.dense_col_index

    args.seq_vocab_size = args.vocab_size_list[0]
    args.detail_seq_col_index = process_module.detail_seq_col_index
    args.addf_seq_col_index = process_module.addf_seq_col_index
    args.detail_seq_len_index = process_module.detail_seq_len_index
    args.addf_seq_len_index = process_module.addf_seq_len_index

    logger_.info("Sparse ID column indices for model: " + str(args.sparse_col_index_id))
    logger_.info("Sparse Non-ID column indices for model: " + str(args.sparse_col_index_nonid))
    logger_.info("Dense column indices for model: " + str(args.dense_col_index))

    logger_.info("max index of each ID type categorical feature: " + str(args.vocab_size_list))
    logger_.info("max index of sequence features: " + str(args.vocab_size_list[0]))
    logger_.info("Building batch iters...")

    train_iter = build_batch_iter(train_input_np, train_labels_np, args.batch_size_train, device = torch_device)
    dev_iter = build_batch_iter(dev_input_np, dev_labels_np, args.batch_size_dev, device = torch_device)

    if args.from_pretrained:
        checkpoint_path = "train_result/best_result/model_checkpoint.pkl"
        logger_.info("Load model's checkpoint from pretrained.")
        logger_.info("chekpoint path :" + checkpoint_path)
        args.checkpoint_loaded = torch.load(checkpoint_path)
        args.model_param = args.checkpoint_loaded["model_param_dict"]
        model = import_module("models." + args.model).model_net(args.model_param).to(torch_device)
        model.load_state_dict(args.checkpoint_loaded["model_state_dict"])

    else:
        logger_.info("Building model's network.")
        args.model_param = {
                       "input_dim":args.input_dim, \
                       "embed_dim":args.embed_dim, \
                       "vocab_size_list": args.vocab_size_list, \
                       "seq_vocab_size": args.seq_vocab_size,\
                       "sparse_col_index_id": args.sparse_col_index_id, \
                       "sparse_col_index_nonid": args.sparse_col_index_nonid, \
                       "dense_col_index": args.dense_col_index, \
                       "hidden_size": args.hidden_size, \
                       "num_dnn_layer": args.num_dnn_layer, \
                       "dropout_rate": args.dropout_rate, \
                       "use_fm": args.use_fm, \
                       "use_dnn": args.use_dnn, \
                       "use_bn": args.use_bn, \
                       "fm_dense_cross": args.fm_dense_cross, \
                       "detail_seq_col_index": args.detail_seq_col_index, \
                       "addf_seq_col_index": args.addf_seq_col_index, \
                       "detail_seq_len_index": args.detail_seq_len_index, \
                       "addf_seq_len_index": args.addf_seq_len_index, \
                       }

        json.dump(args.model_param, open(args.result_dir + "/model_param.json", "w"))
        model = import_module("models." + args.model).model_net(args.model_param).to(torch_device)

    trainer_ins = trainer(model, args, train_iter, dev_iter, torch_device)
    optimizer = trainer_ins.build_optimizer()
    trainer_ins.train_(optimizer)
    current_meomory_used = psutil.Process(os.getpid()).memory_info().rss / (2 ** 30)
    logger_.info("Current meomory used before delete train and dev: " + str(round(current_meomory_used, 6)) + " GB")
    del train_input_np, train_labels_np, train_iter, dev_input_np, dev_labels_np, dev_iter, trainer_ins.train_batch, trainer_ins.dev_batch,
    gc.collect()
    current_meomory_used = psutil.Process(os.getpid()).memory_info().rss / (2 ** 30)
    logger_.info("Current meomory after delete train and dev: " + str(round(current_meomory_used, 6)) + " GB")
    if args.test_run == True:
        test_input_np, test_labels_np =  process_module.test_run_test()
    else:
        test_input_np, test_labels_np =  process_module.get_test_data()
    test_iter = build_batch_iter(test_input_np, test_labels_np, args.batch_size_dev, device = torch_device)
    trainer_ins.evaluate_test(test_iter)
    loss_dict = trainer_ins.json_plot
    plot_save_path = result_dir + "/train_curve.jpg"
    plot_curve(json_data = loss_dict, save_name = plot_save_path).plot_loss()
    logger_.info("Final torch model and log file saved directory:")
    logger_.info(os.getcwd() + "/train_result/" + date_dir)

# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from tqdm import tqdm
import itertools
import os
import json
import pickle
from logging import getLogger
import datetime
import gc
import ast
from importlib import import_module
import glob
import multiprocessing as mp

logger = getLogger("my logger")
class processor(object):
    def __init__(self, arg_parser):
        self.arg_parser = arg_parser
        self.col_names_train_v22 = [
                            "label",
                            "boss_id",
                            "job_id",
                            "geek_id",
                            "exp_id",
                            "deal_type",
                            "geek_position",
                            "geek_city",
                            "geek_workyears",
                            "geek_degree",
                            "geek_major",
                            "el",
                            "eh",
                            "geek_paddf_rate_7d",
                            "geek_success_times_7d",
                            "boss_position",
                            "boss_city",
                            "job_workyears",
                            "job_degree",
                            "boss_title_type",
                            "jl",
                            "jh",
                            "salary_type",
                            "low_salary",
                            "high_salary",
                            "boss_min_chat_tdiff",
                            "job_min_active_tdiff",
                            "job_paddf_rate_7d",
                            "job_success_times_7d",
                            "boss_paddf_success_times_2d",
                            "boss_paddf_success_rate_2d",
                            "boss_paddf_pchat_rate_2d",
                            "boss_paddf_rate_2d",
                            "job_pas_addf_num_24h",
                            "job_paddf_success_times_2d",
                            "job_paddf_success_times_7d",
                            "job_paddf_rate_14d",
                            "job_success_times_2d",
                            "job_psuccess_times_7d",
                            "job_paddfchat_times_7d",
                            "job_type"]

        self.col_names_train_v3 = [
                    "label",
                    "geek_school",
                    "job_company",
                    "boss_id",
                    "job_id",
                    "geek_id",
                    "exp_id",
                    "deal_type",
                    "geek_position",
                    "geek_city",
                    "geek_workyears",
                    "geek_degree",
                    "geek_major",
                    "el",
                    "eh",
                    "geek_paddf_rate_7d",
                    "geek_success_times_7d",
                    "geek_view_boss_14d",
                    "geek_det_boss_14d",
                    "boss_position",
                    "boss_city",
                    "job_workyears",
                    "job_degree",
                    "boss_title_type",
                    "jl",
                    "jh",
                    "boss_min_chat_tdiff",
                    "job_min_active_tdiff",
                    "job_paddf_rate_7d",
                    "job_success_times_7d",
                    "boss_paddf_success_times_2d",
                    "boss_paddf_success_rate_2d",
                    "boss_paddf_pchat_rate_2d",
                    "boss_paddf_rate_2d",
                    "job_pas_addf_num_24h",
                    "job_paddf_success_times_2d",
                    "job_paddf_success_times_7d",
                    "job_paddf_rate_14d",
                    "job_success_times_2d",
                    "job_psuccess_times_7d",
                    "job_paddfchat_times_7d",
                    "salary_type",
                    "low_salary",
                    "high_salary",
                    "job_type",
                    "detail_position",
                    "addf_position"]

        self.col_names_test_v3 = [
                            "label",
                            "geek_school",
                            "job_company",
                            "boss_id",
                            "job_id",
                            "geek_id",
                            "exp_id",
                            "sessionid",
                            "geek_position",
                            "geek_city",
                            "geek_workyears",
                            "geek_degree",
                            "geek_major",
                            "el",
                            "eh",
                            "geek_paddf_rate_7d",
                            "geek_success_times_7d",
                            "geek_view_boss_14d",
                            "geek_det_boss_14d",
                            "boss_position",
                            "boss_city",
                            "job_workyears",
                            "job_degree",
                            "boss_title_type",
                            "jl",
                            "jh",
                            "boss_min_chat_tdiff",
                            "job_min_active_tdiff",
                            "job_paddf_rate_7d",
                            "job_success_times_7d",
                            "boss_paddf_success_times_2d",
                            "boss_paddf_success_rate_2d",
                            "boss_paddf_pchat_rate_2d",
                            "boss_paddf_rate_2d",
                            "job_pas_addf_num_24h",
                            "job_paddf_success_times_2d",
                            "job_paddf_success_times_7d",
                            "job_paddf_rate_14d",
                            "job_success_times_2d",
                            "job_psuccess_times_7d",
                            "job_paddfchat_times_7d",
                            "salary_type",
                            "low_salary",
                            "high_salary",
                            "job_type",
                            "detail_position",
                            "addf_position"]

        self.col_names_train_v32 = [
                                    "label",
                                    "geek_school",
                                    "job_company",
                                    "boss_id",
                                    "job_id",
                                    "geek_id",
                                    "exp_id",
                                    "sessionid",
                                    "geek_position",
                                    "geek_city",
                                    "geek_workyears",
                                    "geek_degree",
                                    "geek_major",
                                    "el",
                                    "eh",
                                    "geek_paddf_rate_7d",
                                    "geek_success_times_7d",
                                    "geek_view_boss_14d",
                                    "geek_det_boss_14d",
                                    "boss_position",
                                    "boss_city",
                                    "job_workyears",
                                    "job_degree",
                                    "boss_title_type",
                                    "jl",
                                    "jh",
                                    "boss_min_chat_tdiff",
                                    "job_min_active_tdiff",
                                    "job_paddf_rate_7d",
                                    "job_success_times_7d",
                                    "boss_paddf_success_times_2d",
                                    "boss_paddf_success_rate_2d",
                                    "boss_paddf_pchat_rate_2d",
                                    "boss_paddf_rate_2d",
                                    "job_pas_addf_num_24h",
                                    "job_paddf_success_times_2d",
                                    "job_paddf_success_times_7d",
                                    "job_paddf_rate_14d",
                                    "job_success_times_2d",
                                    "job_psuccess_times_7d",
                                    "job_paddfchat_times_7d",
                                    "g2b_company_gof",
                                    "g2b_degree_gof",
                                    "g2b_position_similarity",
                                    "g2b_salary_gof",
                                    "g2b_scale_gof",
                                    "g2b_skill_match",
                                    "g2b_stage_gof",
                                    "g2b_title_type_gof",
                                    "g2b_w2v_int_orig_gof",
                                    "g2b_w2v_int_pref_gof",
                                    "g2b_work_distance",
                                    "g2b_workyears_gof",
                                    "salary_type",
                                    "low_salary",
                                    "high_salary",
                                    "job_type",
                                    "detail_position",
                                    "addf_position",
                                    "rank"
                                    "ds"]

        self.col_names_test_v32 = [
                                    "label",
                                    "geek_school",
                                    "job_company",
                                    "boss_id",
                                    "job_id",
                                    "geek_id",
                                    "exp_id",
                                    "sessionid",
                                    "geek_position",
                                    "geek_city",
                                    "geek_workyears",
                                    "geek_degree",
                                    "geek_major",
                                    "el",
                                    "eh",
                                    "geek_paddf_rate_7d",
                                    "geek_success_times_7d",
                                    "geek_view_boss_14d",
                                    "geek_det_boss_14d",
                                    "boss_position",
                                    "boss_city",
                                    "job_workyears",
                                    "job_degree",
                                    "boss_title_type",
                                    "jl",
                                    "jh",
                                    "boss_min_chat_tdiff",
                                    "job_min_active_tdiff",
                                    "job_paddf_rate_7d",
                                    "job_success_times_7d",
                                    "boss_paddf_success_times_2d",
                                    "boss_paddf_success_rate_2d",
                                    "boss_paddf_pchat_rate_2d",
                                    "boss_paddf_rate_2d",
                                    "job_pas_addf_num_24h",
                                    "job_paddf_success_times_2d",
                                    "job_paddf_success_times_7d",
                                    "job_paddf_rate_14d",
                                    "job_success_times_2d",
                                    "job_psuccess_times_7d",
                                    "job_paddfchat_times_7d",
                                    "g2b_company_gof",
                                    "g2b_degree_gof",
                                    "g2b_position_similarity",
                                    "g2b_salary_gof",
                                    "g2b_scale_gof",
                                    "g2b_skill_match",
                                    "g2b_stage_gof",
                                    "g2b_title_type_gof",
                                    "g2b_w2v_int_orig_gof",
                                    "g2b_w2v_int_pref_gof",
                                    "g2b_work_distance",
                                    "g2b_workyears_gof",
                                    "salary_type",
                                    "low_salary",
                                    "high_salary",
                                    "job_type",
                                    "detail_position",
                                    "addf_position",
                                    "rank"
                                    "ds"]

        self.label = ["label"]
        self.feat_remove = ["boss_id", "job_id", "geek_id", "exp_id", "sessionid"]

        ### ====================== used features =============================
        self.geek_company = ["geek_school"]
        self.job_company = ["job_company"]
        self.geek_id = [
                        "geek_position",
                        "geek_city",    # geek_position, geek_city
                        "geek_major"]
        self.job_id = [
                        "boss_position",
                        "boss_city"]  # boss_position, boss_city
        self.geek_semi_dense = [
                        "geek_workyears",
                        "geek_degree"]
        self.job_semi_dense = [
                        "job_workyears",
                        "job_degree",
                        "boss_title_type",]
        self.geek_dense = [
                        "el",
                        "eh",
                        "geek_paddf_rate_7d",
                        "geek_success_times_7d"]
        self.job_dense = [
                        "jl",
                        "jh",
                        "boss_min_chat_tdiff",
                        "job_min_active_tdiff",
                        "job_paddf_rate_7d",
                        "job_success_times_7d",
                        "boss_paddf_success_times_2d",
                        "boss_paddf_success_rate_2d",
                        "boss_paddf_pchat_rate_2d",
                        "boss_paddf_rate_2d",
                        "job_pas_addf_num_24h",
                        "job_paddf_success_times_2d",
                        "job_paddf_success_times_7d",
                        "job_paddf_rate_14d",
                        "job_success_times_2d",
                        "job_psuccess_times_7d",
                        "job_paddfchat_times_7d",
                        "job_type"]
        self.geek_sequence = [
                        "detail_position",
                        "addf_position"]
        self.k8s_col = self.label + self.geek_id + self.job_id + self.geek_semi_dense + self.job_semi_dense + self.geek_dense + self.job_dense + self.geek_sequence
        ### ========================================================================
        self.feat_id = self.geek_id + self.job_id
        self.feat_semi_dense = self.geek_semi_dense + self.job_semi_dense
        self.feat_dense = self.geek_dense + self.job_dense
        self.feat_sequence = self.geek_sequence
        self.feat_model = self.feat_id + self.feat_semi_dense + self.feat_dense
        dense_col_end = len(self.feat_id) + len(self.feat_semi_dense) + len(self.feat_dense)

        self.sparse_col_index_id = list(range(len(self.feat_id)))
        self.sparse_col_index_nonid =  list(range(len(self.feat_id), len(self.feat_id + self.feat_semi_dense)))
        self.dense_col_index = list(range(len(self.feat_id + self.feat_semi_dense), dense_col_end))

        ### ---------------------------- sequence parameters ------------------------------------------------
        self.seq_padding = 10
        self.detail_seq_col_index = list(range(dense_col_end, dense_col_end + self.seq_padding))
        self.addf_seq_col_index = list(range(dense_col_end + self.seq_padding, dense_col_end + self.seq_padding * 2))
        self.detail_seq_len_index = dense_col_end + self.seq_padding * len(self.feat_sequence)
        self.addf_seq_len_index = self.detail_seq_len_index + 1

        position_map_load = json.load(open("mapping_dict_sparse/position_map_with_virtual.json"))
        self.position_map = self.load_json2_mapping(position_map_load)

        logger.info("All used features in-order:")
        feat_order_dict = {str(i+1): feat for i, feat in enumerate(self.feat_model)}
        logger.info("\n".join('%s: %s' % (k, v) for k, v in feat_order_dict.items()))

        self.train_data_dir = "/data1/xuwen/dataset/qm_geek_rank_success_train_v32/"
        self.dev_test_data_dir = "/data1/xuwen/dataset/qm_geek_rank_success_test_v32/" + "2020-10-20/"
        self.col_names_train_df = self.col_names_train_v32
        if not self.arg_parser.test_run:
            logger.info("Train data dir:" + self.train_data_dir)
            logger.info("Test data dir:" + self.dev_test_data_dir)
            self.assert_dev_test_npy()


    def get_train_days(self):
        today = datetime.date.today()
        day_end = datetime.date(2020,10,20)  # temp date
        day_since = datetime.date(2020,6,30)
        day_gap = (day_end - day_since).days
        all_train_days = [day_end - datetime.timedelta(days = i) for i in reversed(range(0, day_gap+1))]
        all_train_days = list(map(str, all_train_days))
        return all_train_days


    def transfer_learning_days(self):
        today = datetime.date.today()
        day_end = today - datetime.timedelta(days = 2)  # 获取前天的日期
        day_since = today - datetime.timedelta(days = 7)
        day_gap = (day_end - day_since).days
        all_days = [day_end - datetime.timedelta(days = i) for i in reversed(range(0, day_gap+1))]
        transfer_train_days = list(map(str, all_days))
        transfer_dev_test_day = str(day_end)
        return transfer_train_days, transfer_dev_test_day


    def get_train_data(self):
        # if self.arg_parser.transfer_learning and self.arg_parser.from_pretrained:
        #     train_days = self.transfer_learning_days()
        # else:
        #     train_days = self.get_train_days()
        #
        # logger.info("Days for training set")
        # logger.info(train_days)
        # df_train = []
        # for date_ in train_days:
        #     day_train_file = self.train_data_dir + date_ + "/000000_0"
        #     day_train_df = pd.read_csv(day_train_file, names = self.col_names_train_df, index_col = None, header = None, sep = "\001", na_values='\\N', low_memory=False)
        #     day_train_df = day_train_df[self.feat_model + self.feat_sequence + self.label]
        #     day_train_df.fillna(0)
        #     day_train_df.replace(np.inf, 0, inplace=True)
        #     day_train_df.replace(np.nan, 0, inplace=True)
        #     df_train.append(day_train_df)
        #
        # df_train = pd.concat(df_train, axis = 0)

        df_train = pd.read_csv(self.train_data_dir + "0801_1018_v32_train.csv", names = self.k8s_col, index_col = None, sep = "\001", na_values='\\N', low_memory=False)
        #df_train = pd.read_csv("/data1/xuwen/dataset/qm_geek_rank_success_train_v32/0801_1018_v32_train_run_test.csv")
        df_train.fillna(0)
        df_train.replace(np.inf, 0, inplace=True)
        df_train.replace(np.nan, 0, inplace=True)

        df_train_cost = round(df_train.memory_usage(index=True).sum()/(1024**3), 5)
        logger.info(f" Train df memory used: {df_train_cost} GB.")
        train_np = self.np_converter(df_train[self.feat_model + self.label].values)
        train_input_np, train_labels_np = train_np[:,:-1], train_np[:,-1:]
        ### ---------------------------------------numpy sequence ---------------------------------------------
        pool = mp.Pool()
        train_pos_sequences = df_train[self.feat_sequence].values
        structured_seq_detail = np.array(list(pool.map(self.pos_seq_converter, train_pos_sequences[:, 0])))
        structured_seq_addf = np.array(list(pool.map(self.pos_seq_converter, train_pos_sequences[:, 1])))
        seq_len_detail = structured_seq_detail[:,-1:]
        seq_len_addf = structured_seq_addf[:,-1:]
        train_seq_all = np.concatenate([structured_seq_detail[:,:10], structured_seq_addf[:,:10], seq_len_detail, seq_len_addf], axis = 1)
        train_input_np = np.concatenate([train_input_np, train_seq_all], axis = 1)
        return train_input_np, train_labels_np


    def get_dev_data(self):
        dev_path_npy = os.path.join(self.dev_test_data_dir, "deepfm_v32/np_dev.npy")
        dev_np = np.load(dev_path_npy, allow_pickle=True).astype(float)
        dev_input_np, dev_label_np = dev_np[:, :-1], dev_np[:, -1:]
        return dev_input_np, dev_label_np


    def get_test_data(self):
        test_path_npy = os.path.join(self.dev_test_data_dir, "deepfm_v32/np_test.npy")
        test_np = np.load(test_path_npy, allow_pickle=True).astype(float)
        test_input_np = test_np[:, :-1]
        test_label_np = test_np[:, -1:]
        return test_input_np, test_label_np


    def test_run_train(self):
        run_test_df = pd.read_csv("dataset/10_08_train_v3", names = self.col_names_train_v3, index_col = None,  header = None, sep = "\t", na_values='\\N', low_memory=False)
        #run_test_df = pd.read_csv("dataset/000000_0", names = self.col_names_train_v22, index_col = None,  header = None, sep = "\t", na_values='\\N')
        run_test_df[self.feat_model + self.feat_sequence + self.label].fillna(0)
        run_test_df.replace(np.inf, 0, inplace=True)
        run_test_df.replace(np.nan, 0, inplace=True)
        df_train, df_dev = self.train_dev_split(run_test_df)
        train_np = self.np_converter(df_train[self.feat_model + self.label].values)
        train_input_np = train_np[:, :-1]
        train_label_np = train_np[:, -1:]
        ### ---------------------------------------numpy sequence ---------------------------------------------
        pool = mp.Pool()
        train_pos_sequences = df_train[self.feat_sequence].values
        structured_seq_detail = np.array(list(pool.map(self.pos_seq_converter, train_pos_sequences[:, 0])))
        structured_seq_addf = np.array(list(pool.map(self.pos_seq_converter, train_pos_sequences[:, 1])))
        seq_len_detail = structured_seq_detail[:,-1:]
        seq_len_addf = structured_seq_addf[:,-1:]
        train_seq_all = np.concatenate([structured_seq_detail[:,:10], structured_seq_addf[:,:10], seq_len_detail, seq_len_addf], axis = 1)
        train_input_np = np.concatenate([train_input_np, train_seq_all], axis = 1)
        return train_input_np, train_label_np


    def test_run_dev(self):
        run_test_df = pd.read_csv("dataset/10_08_train_v3", names = self.col_names_train_v3, index_col = None,  header = None, sep = "\t", na_values='\\N', low_memory=False)
        #run_test_df = pd.read_csv("dataset/000000_0", names = self.col_names_train_v22, index_col = None,  header = None, sep = "\t", na_values='\\N')
        run_test_df[self.feat_model + self.label].fillna(0)
        run_test_df.replace(np.inf, 0, inplace=True)
        run_test_df.replace(np.nan, 0, inplace=True)
        df_train, df_dev = self.train_dev_split(run_test_df)
        dev_np = self.np_converter(df_dev[self.feat_model + self.label].values)
        dev_input_np = dev_np[:, :-1]
        dev_label_np = dev_np[:, -1:]
        ### ---------------------------------------numpy sequence ---------------------------------------------
        pool = mp.Pool()
        dev_pos_sequences = df_dev[self.feat_sequence].values
        structured_seq_detail = np.array(list(pool.map(self.pos_seq_converter, dev_pos_sequences[:, 0])))
        structured_seq_addf = np.array(list(pool.map(self.pos_seq_converter, dev_pos_sequences[:, 1])))
        seq_len_detail = structured_seq_detail[:,-1:]
        seq_len_addf = structured_seq_addf[:,-1:]
        dev_seq_all = np.concatenate([structured_seq_detail[:,:10], structured_seq_addf[:,:10], seq_len_detail, seq_len_addf], axis = 1)
        dev_input_np = np.concatenate([dev_input_np, dev_seq_all], axis = 1)
        return dev_input_np, dev_label_np


    def test_run_test(self):
        run_test_df = pd.read_csv("dataset/10_08_train_v3", names = self.col_names_train_v3, index_col = None,  header = None, sep = "\t", na_values='\\N', low_memory=False)
        run_test_df[self.feat_model + self.label].fillna(0)
        run_test_df.replace(np.inf, 0, inplace=True)
        run_test_df.replace(np.nan, 0, inplace=True)
        test_np = self.np_converter(run_test_df[self.feat_model + self.label].values)
        test_input_np = test_np[:, :-1]
        test_label_np = test_np[:, -1:]
        ### ---------------------------------------numpy sequence ---------------------------------------------
        pool = mp.Pool()
        test_pos_sequences = run_test_df[self.feat_sequence].values
        structured_seq_detail = np.array(list(pool.map(self.pos_seq_converter, test_pos_sequences[:, 0])))
        structured_seq_addf = np.array(list(pool.map(self.pos_seq_converter, test_pos_sequences[:, 1])))
        seq_len_detail = structured_seq_detail[:,-1:]
        seq_len_addf = structured_seq_addf[:,-1:]
        test_seq_all = np.concatenate([structured_seq_detail[:,:10], structured_seq_addf[:,:10], seq_len_detail, seq_len_addf], axis = 1)
        test_input_np = np.concatenate([test_input_np, test_seq_all], axis = 1)
        return test_input_np, test_label_np


    def assert_dev_test_npy(self):
        if not os.path.exists(self.dev_test_data_dir + "deepfm_v32/np_dev.npy") or not os.path.exists(self.dev_test_data_dir + "deepfm_v32/np_test.npy"):
            self.save_npy_dev_test()


    def save_npy_dev_test(self):
        # partition_files = sorted(glob.glob(self.dev_test_data_dir + "0000*"))
        # df_dev_test = []
        # for partition in partition_files:
        #     day_dev_test_df = pd.read_csv(partition, names = self.col_names_test_v3, index_col = None, header = None, sep = "\001", na_values='\\N', low_memory=False)
        #     day_dev_test_df = day_dev_test_df[self.feat_model + self.feat_sequence + self.label]
        #     day_dev_test_df.fillna(0)
        #     day_dev_test_df.replace(np.inf, 0, inplace=True)
        #     day_dev_test_df.replace(np.nan, 0, inplace=True)
        #     df_dev_test.append(day_dev_test_df)

        # df_dev_test = pd.concat(df_dev_test, axis = 0)
        df_dev_test = pd.read_csv(self.dev_test_data_dir + "000000_0_sql_feat_only", names = self.k8s_col, index_col = None, sep = "\001", na_values='\\N', low_memory=False)
        df_dev_test.fillna(0)
        df_dev_test.replace(np.inf, 0, inplace=True)
        df_dev_test.replace(np.nan, 0, inplace=True)
        df_dev_test_cost = round(df_dev_test.memory_usage(index=True).sum()/(1024**3), 5)
        logger.info(f" Dev and test df memory used: {df_dev_test_cost} GB.")
        dev_test_np = self.np_converter(df_dev_test[self.feat_model + self.feat_sequence + self.label].values)
        del df_dev_test
        gc.collect()

        np_dev_test_positive = dev_test_np[dev_test_np[:, -1].astype(int) == 1] # label feat1, feat2, ....
        np_dev_test_negative = dev_test_np[dev_test_np[:, -1].astype(int) == 0]
        del dev_test_np

        num_all_positive = np_dev_test_positive.shape[0]
        num_all_negative = np_dev_test_negative.shape[0]

        sample_indices_all_positive = list(range(0, num_all_positive))
        sample_indices_all_negative = list(range(0, num_all_negative))
        np.random.shuffle(sample_indices_all_positive)
        np.random.shuffle(sample_indices_all_negative)

        dev_split_rate = 0.2
        dev_positive_split = int(num_all_positive * dev_split_rate)
        dev_negative_split = int(num_all_negative * dev_split_rate)

        dev_indices_positive = sample_indices_all_positive[:dev_positive_split]
        test_indices_positive = sample_indices_all_positive[dev_positive_split:]
        dev_indices_negative = sample_indices_all_negative[:dev_negative_split]
        test_indices_negative = sample_indices_all_negative[dev_negative_split:]

        np_dev_positive = np_dev_test_positive[dev_indices_positive]
        np_test_positive = np_dev_test_positive[test_indices_positive]

        np_dev_negative = np_dev_test_negative[dev_indices_negative]
        np_test_negative = np_dev_test_negative[test_indices_negative]

        np_dev_concat_pos_nag = np.concatenate([np_dev_positive, np_dev_negative], axis = 0)
        np_test_concat_pos_nag = np.concatenate([np_test_positive, np_test_negative], axis = 0)

        dev_shuffle = list(range(0, np_dev_concat_pos_nag.shape[0]))
        np.random.shuffle(dev_shuffle)
        np_dev_concat_pos_nag = np_dev_concat_pos_nag[dev_shuffle, :]

        test_shuffle = list(range(0, np_test_concat_pos_nag.shape[0]))
        np.random.shuffle(test_shuffle)
        np_test_concat_pos_nag = np_test_concat_pos_nag[test_shuffle, :]

        pool = mp.Pool()

        dev_detail_str = np_dev_concat_pos_nag[:, -3]
        dev_addf_str = np_dev_concat_pos_nag[:, -2]
        dev_structured_seq_detail = np.array(list(pool.map(self.pos_seq_converter, dev_detail_str)))
        dev_structured_seq_addf = np.array(list(pool.map(self.pos_seq_converter, dev_addf_str)))
        # logger.info(dev_structured_seq_detail)
        # logger.info(dev_structured_seq_addf)
        # logger.info(dev_structured_seq_detail.shape)
        # logger.info(dev_structured_seq_addf.shape)
        dev_seq_len_detail = dev_structured_seq_detail[:,-1:]
        dev_seq_len_addf = dev_structured_seq_addf[:,-1:]
        dev_seq_all = np.concatenate([dev_structured_seq_detail[:,:10], dev_structured_seq_addf[:,:10], dev_seq_len_detail, dev_seq_len_addf], axis = 1)
        dev_input_non_seq = np_dev_concat_pos_nag[:, :-3]
        dev_label = np_dev_concat_pos_nag[:, -1:]
        np_dev_all = np.concatenate([dev_input_non_seq, dev_seq_all, dev_label], axis = 1)

        test_detail_str = np_test_concat_pos_nag[:, -3]
        test_addf_str = np_test_concat_pos_nag[:, -2]
        test_structured_seq_detail = np.array(list(pool.map(self.pos_seq_converter, test_detail_str)))
        test_structured_seq_addf = np.array(list(pool.map(self.pos_seq_converter, test_addf_str)))
        test_seq_len_detail = test_structured_seq_detail[:,-1:]
        test_seq_len_addf = test_structured_seq_addf[:,-1:]
        test_seq_all = np.concatenate([test_structured_seq_detail[:,:10], test_structured_seq_addf[:,:10], test_seq_len_detail, test_seq_len_addf], axis = 1)
        test_input_non_seq = np_test_concat_pos_nag[:, :-3]
        test_label = np_test_concat_pos_nag[:, -1:]
        np_test_all = np.concatenate([test_input_non_seq, test_seq_all, test_label], axis = 1)

        np.save(self.dev_test_data_dir + "deepfm_v32/np_dev.npy", np_dev_all)
        np.save(self.dev_test_data_dir + "deepfm_v32/np_test.npy", np_test_all)


    def np_converter(self, input_np):
        #input_np = df_data[self.feat_model + self.label].values
        converter_id_dict_list, converter_nonid_dict_list = self.create_mapping_list()
        self.arg_parser.vocab_size_list = [len(sparse_dict.keys())+1 for sparse_dict in converter_id_dict_list]

        for convert_col, convert_dict in zip(self.sparse_col_index_id, converter_id_dict_list):
            input_np[:, convert_col] = self.sparse_col_convert(input_np[:, convert_col], convert_dict)
        for convert_col, convert_dict in zip(self.sparse_col_index_nonid, converter_nonid_dict_list):
            input_np[:, convert_col] = self.sparse_col_convert(input_np[:, convert_col], convert_dict)
        return input_np


    def create_mapping_list(self):
        position_map_load = json.load(open("mapping_dict_sparse/position_map_with_virtual.json"))
        position_map = self.load_json2_mapping(position_map_load)

        city_map_load = json.load(open("mapping_dict_sparse/city_map.json"))
        city_map = self.load_json2_mapping(city_map_load)

        major_map_load = json.load(open("mapping_dict_sparse/major_map.json"))
        major_map = self.load_json2_mapping(major_map_load)

        geek_wordyear_load = json.load(open("mapping_dict_dense/geek_workyear.json"))
        geek_wordyear = self.load_json2_mapping(geek_wordyear_load)

        geek_degree_load = json.load(open("mapping_dict_dense/degree.json"))
        geek_degree = self.load_json2_mapping(geek_degree_load)

        job_workyear_load = json.load(open("mapping_dict_dense/job_workyear.json"))
        job_workyear = self.load_json2_mapping(job_workyear_load)

        job_degree_load = json.load(open("mapping_dict_dense/degree.json"))
        job_degree = self.load_json2_mapping(job_degree_load)

        boss_title_type_load = json.load(open("mapping_dict_dense/boss_title_type.json"))
        boss_title_type = self.load_json2_mapping(boss_title_type_load)

        converter_id_dict_list = [position_map, city_map, major_map, position_map, city_map]
        converter_nonid_dict_list = [geek_wordyear, geek_degree, job_workyear, job_degree, boss_title_type]
        return  converter_id_dict_list, converter_nonid_dict_list


    def load_json2_mapping(self,loaded_json):
        mapping = {}
        for key_, value_ in loaded_json.items():
            mapping[int(key_)] = int(value_)
        return mapping


    def sparse_col_convert(self, sparse_col, sparse_dict):
        sparse_col = sparse_col.astype("int")
        sparse_col = np.array([sparse_dict.get(v) if sparse_dict.get(v)!= None else 0 for v in sparse_col ])
        return sparse_col


    def pos_seq_converter(self, seq):
        if type(seq) == str:
            seq_int = ast.literal_eval(seq)
            if type(seq_int) == int:
                seq_len = 1
                seq_int = self.position_map.get(seq_int) if self.position_map.get(seq_int)!= None else 0
                seq_int = [seq_int] + [0] * 9

            elif len(seq_int) > 1 and len(seq_int) < 10:
                seq_len = len(seq_int)
                seq_int = [self.position_map.get(pos_id) if self.position_map.get(pos_id)!= None else 0 for pos_id in seq_int]
                seq_int = seq_int + (10 - len(seq_int)) * [0]
            elif len(seq_int) == 10:
                seq_len = 10
                seq_int = [self.position_map.get(pos_id) if self.position_map.get(pos_id)!= None else 0 for pos_id in seq_int]
                seq_int = list(seq_int)
            return seq_int + [seq_len]
        else:
            return [0] * 10 + [1]


    def train_dev_split(self, df, valid_split = 0.1):
        num_row = df.shape[0]
        dev_start = int(num_row * ( 1 - valid_split * 2))  # test_split = 0.1
        #test_start = int(num_row * ( 1 - valid_split ))
        train_df = df.iloc[:dev_start]
        dev_df = df.iloc[dev_start:]
        #test_df = df.iloc[test_start:]
        return train_df, dev_df


class build_batch_iter(object):
    def __init__(self,np_input, np_label, batch_size = 2000, device = "cpu"):
        self.np_input = np_input
        self.np_label = np_label
        self.batch_size = batch_size
        self.num_samples = len(np_input)
        self.num_minibatch = self.num_samples // self.batch_size
        self.residue = self.num_samples % self.num_minibatch
        self.batch_index = 0
        self.device = device

    def shuffle_data(self):
        # 使正样本和负样本均匀分布。先将负样本shuffle，之后取前k个和正样本数量相同的负样本，再和正样本进行concate。
        # 最后shuffle concate后的正负样本数据集
        sample_indices = list(range(0, self.np_label.shape[0]))
        np.random.shuffle(sample_indices)
        self.np_input = self.np_input[sample_indices, :]
        self.np_label = self.np_label[sample_indices, :]

    def to_tensor_input(self, np_input):
        input_tensor = torch.DoubleTensor(np_input).to(self.device)
        return input_tensor

    def to_tensor_label(self, np_label):
        label_tensor = torch.LongTensor(np_label).to(self.device)
        return label_tensor

    def __next__(self):
        if self.batch_index < self.num_minibatch:
            batch_input = self.np_input[self.batch_size * self.batch_index : self.batch_size * (self.batch_index + 1)]
            batch_label = self.np_label[self.batch_size * self.batch_index : self.batch_size * (self.batch_index + 1)]
            self.batch_index += 1
            batch_input = self.to_tensor_input(batch_input)
            batch_label = self.to_tensor_label(batch_label)
            return batch_input, batch_label
        elif self.residue != 0 and self.batch_index == self.num_minibatch:
            batch_input = self.np_input[self.batch_size * self.batch_index:]
            batch_label = self.np_label[self.batch_size * self.batch_index:]
            self.batch_index += 1
            batch_input = self.to_tensor_input(batch_input)
            batch_label = self.to_tensor_label(batch_label)
            return batch_input, batch_label
        else:
            self.batch_index = 0
            raise StopIteration

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_minibatch if self.residue == 0 else self.num_minibatch + 1

    def __getitem__(self, position):
        if position < self.num_minibatch:
            batch_input = self.np_input[self.batch_size * position : self.batch_size * (position + 1)]
            batch_label = self.np_label[self.batch_size * position : self.batch_size * (position + 1)]
            batch_input = self.to_tensor_input(batch_input)
            batch_label = self.to_tensor_label(batch_label)
            return batch_input,  batch_label
        elif self.residue != 0 and position == self.num_minibatch:
            batch_input = self.to_tensor_input(self.np_input[self.batch_size * position : ])
            batch_label = self.to_tensor_label(self.np_label[self.batch_size * position : ])
            return batch_input, batch_label
        else:
            raise IndexError(": Index out of range")

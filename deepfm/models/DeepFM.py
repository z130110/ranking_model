import torch
from torch import nn

class model_net(nn.Module):
    def __init__(self, param_dict):
        super(model_net, self).__init__()
        '''
        ### Model's forward shape should be: [batch size, number features], type: tensor.float ###
        ### output shape: [batch size, 1], which are probabilities for each batch sample ###
        param_dict: a dictionary which contains all requiring parameters
        all input parameters with corresponding examples:
            "embed_dim": 8  ### type: int. Embedding dimension, not vocabulary size.
            "vocab_size_list": [150, 323]  ### type: list. Vocaburary sizes for each categorical features.
            "sparse_col_index": [0, 1, 2]  ### type: list. Column indices of categorical features.
            "dense_col_index": [2, 3, 4, 5] ### type: list. Column indices of numerical features.
            "hidden_size": 128  ### type: int. Dimension of each hidden layers.
            "num_dnn_layer": 4 ### type: int. Number of hidden layers.
            "dropout_rate": 0.5  ### type: float. dropout rate.
            "use_fm": True  ### type: bool. Use fm or not, if not, them this network is a normal DNN.
            "use_dnn": True ### type: bool. Use DNN or not. if not, them this network is a pure FM.
            "use_bn": True  ### type: bool. Use bn or not,
            "fm_dense_cross": False  ###  type: bool. Whether add numerical features to FM or not, default is not.
            "torch_device": ""  torch.device("cuda")  ### type: torch.device.
            "sparse_dict_list": [ {100:0, 2000:1}, {},{},....]   ### type: list. A list of categorical mapping dictionary
        '''
        self.param_dict = param_dict
        self.embedding_cols =nn.ModuleList([nn.Embedding(vocab_size, self.param_dict["embed_dim"]) for vocab_size in param_dict["vocab_size_list"]])
        self.embedding_seq_detail = nn.Embedding(self.param_dict["seq_vocab_size"], self.param_dict["embed_dim"], padding_idx=0, scale_grad_by_freq=True)
        self.embedding_seq_addf = nn.Embedding(self.param_dict["seq_vocab_size"], self.param_dict["embed_dim"], padding_idx=0, scale_grad_by_freq=True)
        self.embed_FM_1d_sparse_col = nn.ModuleList([nn.Embedding(vocab_size, 1, padding_idx=0, scale_grad_by_freq=True) for vocab_size in param_dict["vocab_size_list"]])
        self.embed_FM_1d_detail = nn.Embedding(self.param_dict["seq_vocab_size"], 1, padding_idx=0, scale_grad_by_freq=True)
        self.embed_FM_1d_addf = nn.Embedding(self.param_dict["seq_vocab_size"], 1, padding_idx=0, scale_grad_by_freq=True)

        self.fully_connect = nn.Linear(param_dict["hidden_size"], 1)
        self.use_fm = param_dict["use_fm"]
        self.use_dnn = param_dict["use_dnn"]
        self.fm_dense_cross = param_dict["fm_dense_cross"]
        #self.num_categorical_feat = len(param_dict["vocab_size_list"])
        self.num_categorical_feat = len(param_dict["vocab_size_list"]) + len([param_dict["detail_seq_len_index"], param_dict["addf_seq_len_index"]])
        self.num_dense_feat =  len(param_dict["sparse_col_index_nonid"]) + len(param_dict["dense_col_index"])
        self.fm_linear = FM_linear(num_dense_feats = self.num_dense_feat)
        self.fm_bn = nn.BatchNorm1d(1)

        self.dnn_input_dim = self.param_dict["embed_dim"] * self.num_categorical_feat + self.num_dense_feat

        self.dnn = DNN(param_dict["num_dnn_layer"], input_dim = self.dnn_input_dim, hidden_dim = param_dict["hidden_size"], \
                    dropout_rate = self.param_dict["dropout_rate"], use_bn = self.param_dict["use_bn"])

        self.dnn_linear = nn.Linear(param_dict["hidden_size"] // (2 ** (param_dict["num_dnn_layer"] - 1)), 1, bias=True)

        #self.dnn_linear = nn.Linear(param_dict["hidden_size"], 1, bias=True)#.to(param_dict["torch_device"])

        #self.weight = nn.Parameter(torch.Tensor(len(param_dict["sparse_col_index_nonid"]) + len(param_dict["dense_col_index"]), 1))
        #.to(param_dict["torch_device"])
        #self.sparse_dict_list =  param_dict["sparse_dict_list"]
        #print(self.sparse_dict_list)
        #self.dnn_out_bias = nn.Parameter(torch.zeros((1,)))
        # init_std = 0.0001
        # torch.nn.init.normal_(self.weight, mean = 0, std = init_std)


    def FM(self, input_embeddings):
        fm_input = input_embeddings
        square_of_sum = torch.pow(torch.sum(fm_input, dim=1, keepdim=True), 2)
        sum_of_square = torch.sum(fm_input * fm_input, dim=1, keepdim=True)
        cross_term = square_of_sum - sum_of_square
        cross_term = 0.5 * torch.sum(cross_term, dim=2, keepdim=False)
        return cross_term


    def forward(self, input_tensor):
        batch_size = input_tensor.shape[0]
        embedding_all = []
        for sparse_col, embed_col in zip(self.param_dict["sparse_col_index_id"], self.embedding_cols):
            embedding_all.append(embed_col(input_tensor[:, sparse_col].unsqueeze(1).long()))

        seq_divisor_detail = input_tensor[:, self.param_dict["detail_seq_len_index"]].unsqueeze(1)
        seq_detail_ids = input_tensor[:, self.param_dict["detail_seq_col_index"]].long()  #to(dtype = torch.long)
        seq_sum_embed_detail = self.embedding_seq_detail(seq_detail_ids).sum(dim=1)
        seq_detail_embed_final = seq_sum_embed_detail / seq_divisor_detail
        embedding_all.append(seq_detail_embed_final.unsqueeze(1).float())


        seq_divisor_addf = input_tensor[:, self.param_dict["addf_seq_len_index"]].unsqueeze(1)
        seq_addf_ids = input_tensor[:, self.param_dict["addf_seq_col_index"]].to(dtype = torch.long)
        seq_sum_embed_addf = self.embedding_seq_addf(seq_addf_ids).sum(dim=1)
        seq_addf_embed_final = seq_sum_embed_addf / seq_divisor_addf
        embedding_all.append(seq_addf_embed_final.unsqueeze(1).float())

        embedding_concat = torch.cat(embedding_all, dim = 1)
        dense_input = input_tensor[:,self.param_dict["sparse_col_index_nonid"] + self.param_dict["dense_col_index"]].float()

        if self.use_fm and len(self.embedding_cols) > 0:
            #fm_linear_output = self.fm_linear()
            fm_cross_output = self.FM(embedding_concat)

            if self.fm_dense_cross:
                fm_dense_input = dense_input.unsqueeze(2)
                fm_dense_output = self.FM(fm_dense_input)
                fm_cross_output += fm_dense_output

            embedding_fm_1d = []
            for sparse_col, embed_col_1d in zip(self.param_dict["sparse_col_index_id"], self.embed_FM_1d_sparse_col):
                embedding_fm_1d.append(embed_col_1d(input_tensor[:, sparse_col].long()))

            seq_sum_embed_FM_1d_detail = self.embed_FM_1d_detail(seq_detail_ids).sum(dim=1)
            seq_detail_embed_FM_1d_final = seq_sum_embed_FM_1d_detail / seq_divisor_detail
            embedding_fm_1d.append(seq_detail_embed_FM_1d_final.float())

            seq_sum_embed_FM_1d_addf = self.embed_FM_1d_addf(seq_addf_ids).sum(dim=1)
            seq_addf_embed_FM_1d_final = seq_sum_embed_FM_1d_addf / seq_divisor_detail
            embedding_fm_1d.append(seq_addf_embed_FM_1d_final.float())

            embedding_concat_fm_1d = torch.cat(embedding_fm_1d, dim = 1)
            sum_embed_all_dim = torch.sum(embedding_concat_fm_1d, dim = 1).unsqueeze(1)

            fm_linear_output = self.fm_linear(dense_input)
            fm_final_output = fm_cross_output + fm_linear_output + sum_embed_all_dim
            fm_final_output = self.fm_bn(fm_final_output)
            if not self.use_dnn:  #  Only FM part, without DNN
                #print("FM only", torch.sigmoid(fm_final_output).shape)
                return torch.sigmoid(fm_final_output)

        dnn_embed_combine = embedding_concat.view(batch_size, \
                        self.num_categorical_feat * self.param_dict["embed_dim"])

        dnn_input = torch.cat([dnn_embed_combine, dense_input], dim = 1)

        if self.use_dnn:
            dnn_output = self.dnn(dnn_input)
            dnn_output = self.dnn_linear(dnn_output)
            if not self.use_fm:     #  Only DNN, no FM
                #print("DNN only", torch.sigmoid(dnn_output).shape)
                return torch.sigmoid(dnn_output)

        if self.use_fm and self.use_dnn:
            fm_dnn_merge = fm_final_output + dnn_output
            fm_dnn_output = torch.sigmoid(fm_dnn_merge)
            #print("Both FM and DNN", fm_dnn_output.shape)
            return fm_dnn_output   # Both FM and DNN


class FM_linear(nn.Module):
    def __init__(self, num_dense_feats):
        super(FM_linear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(num_dense_feats, 1))
        #init_std = 0.0001
        #torch.nn.init.normal_(self.weight, mean = 0, std = init_std)

    def forward(self, dense_input):
        output = dense_input.matmul(self.weight)
        return output


class DNN(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, dropout_rate = 0, use_bn = True):
        super(DNN,self).__init__()
        self.num_layers = num_layers
        #self.dnn_dims = [input_dim] + [hidden_dim] * num_layers
        self.dnn_dims = [input_dim] + [hidden_dim // (2**i) for i in range(0, num_layers)]
        self.linear_layers = nn.ModuleList([nn.Linear(self.dnn_dims[i],self.dnn_dims[i+1]) for i in range(len(self.dnn_dims) - 1)])
        self.use_bn = use_bn
        if self.use_bn:
            self.bn = nn.ModuleList([nn.BatchNorm1d(self.dnn_dims[i]) for i in range(1, len(self.dnn_dims))])

        self.activation_layers = nn.ModuleList([nn.ReLU(inplace=True) for i in range(1, len(self.dnn_dims))])
        self.dropout = nn.Dropout(dropout_rate)
        #self.to(torch_device)

    def forward(self, x):
        for i in range(len(self.linear_layers)):
            x = self.linear_layers[i](x)
            if self.use_bn:
                x = self.bn[i](x)
            x = self.activation_layers[i](x)
            x = self.dropout(x)
        out = x
        return out

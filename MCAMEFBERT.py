import torch
import torch.nn as nn
import torch.optim as optim

import torch.utils.data as Data
import torch.nn.functional as F
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools
import math
import numpy as np
import os
import random

import time
import torch
tqdm.pandas(ascii=True)
import os
from sklearn.metrics import auc, roc_curve, precision_recall_curve, average_precision_score
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, TensorDataset
from termcolor import colored
# from models.moss import Lucky, Moss
import datetime

from torch.utils.tensorboard import SummaryWriter

from mymodel import EncoderLayer,Binary_Encoder,NCPEncoder,EIIP_Encoder,ENAC_Encoder,MultiChannelAttention

import copy
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda", 0)

import torch
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("/home/yjl/DNABERT_2-main", trust_remote_code=True)
model = AutoModel.from_pretrained("/home/yjl/DNABERT_2-main", trust_remote_code=True)

class MCAMEFBERT(nn.Module):
    def __init__(self, params,hidden_size=256, device='cuda'):
        super(MCAMEFBERT, self).__init__()
        self.hidden_dim = hidden_size
        self.emb_dim = 768
        self.max_seq = params['seq_len']

        self.device = device

        self.tokenizer = tokenizer
        # self.tokenizer = AutoTokenizer.from_pretrained( "/mnt/sdb/home/zyx/SpliceBERT-main/models/SpliceBERT.1024nt", use_fast=True)
        self.model = model

        # 定义 CNN 层
        self.conv1 = nn.Conv1d(in_channels=self.emb_dim, out_channels=self.hidden_dim, kernel_size=3, padding=1,stride=1)
        self.conv2 = nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.hidden_dim, kernel_size=7,stride=3, padding=2)
        self.max_pool = nn.MaxPool1d(kernel_size=2,stride=2)

        # 定义补充特征编码
        self.binary = Binary_Encoder()
        self.ncp = NCPEncoder()
        self.eiip = EIIP_Encoder()
        self.enac = ENAC_Encoder()

        self._mca = MultiChannelAttention()

        self.layers = nn.ModuleList([EncoderLayer(d_model=256,
                                                    ffn_hidden=256,
                                                    n_head=8,
                                                    drop_prob=0.3
                                                    )
                                     for _ in range(6)])

        self.fuse_layers = nn.ModuleList([EncoderLayer(d_model=256,
                                                    ffn_hidden=256,
                                                    n_head=8,
                                                    drop_prob=0.3
                                                  )
                                     for _ in range(6)])

        self.bilstm = nn.LSTM(self.hidden_dim, self.hidden_dim//2, num_layers=3,batch_first=True, bidirectional=True,dropout=0.2)
        self.bilstm_fuse = nn.LSTM(self.hidden_dim, self.hidden_dim//2, num_layers=3,batch_first=True, bidirectional=True,dropout=0.2)

        # self.block = nn.Sequential(
        #     nn.Linear(256, 64),  # 1001:128384  51:6784  510：65536
        #     nn.Dropout(0.2),
        #     nn.LeakyReLU(),
        #     nn.Linear(64,2)
        # )

        self.fnn_maxpool = nn.MaxPool1d(kernel_size=2, stride=2, padding=1, ceil_mode=True)

        self.block = nn.Sequential(
            nn.Linear(256, 128),  # 1001:128384  51:6784  510：65536 GRU:50150  lsr:128512
            nn.LayerNorm(128),
            # nn.Dropout(0.2),
            nn.LeakyReLU(),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.LeakyReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )


    def forward(self, text):

        inputs = self.tokenizer(text, return_tensors="pt", padding='max_length', max_length=self.max_seq,
                                truncation=True)
        inputs = inputs.to(device)
        input_ids = inputs.input_ids.squeeze(0)
        token_type_ids = inputs.token_type_ids.squeeze(0)
        attention_mask = inputs.attention_mask.squeeze(0)

        outputs = self.model(input_ids,token_type_ids,attention_mask)

        hidden_states = outputs[0]  #batch,201,768

        x = hidden_states.permute(0,2,1)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.max_pool(x)   #[128, 256, 33]

        # 新的分支：使用 encode_sequence_1mer 对 text 进行编码
        encoded_sequences = encode_sequence_1mer(text, self.max_seq)  # 返回 numpy 数组
        encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long, device=self.device)  # 转换为张量

        x_onehot = self.binary(encoded_sequences)
        x_eiip = self.eiip(encoded_sequences)
        x_ncp = self.ncp(encoded_sequences)
        x_enac = self.enac(encoded_sequences)

        x_fuse = torch.cat(( x_onehot, x_eiip, x_ncp, x_enac), dim=1)

        x_fuse = self._mca(x_fuse)

        x_fuse = x_fuse.permute(0,2,1)
        x_fuse_bilstm,_ = self.bilstm_fuse(x_fuse)

        x = x.permute(0,2,1)

        x_bilstm,_ = self.bilstm(x)

        for layer in self.layers:
            x_bilstm,_= layer(x_bilstm, None)

        for layer in self.fuse_layers:
            x_fuse_bilstm,_= layer(x_fuse_bilstm, None)

        x_final = x_bilstm * 0.9 + x_fuse_bilstm * 0.1
        x_final = self.fnn_maxpool(x_final.permute(0,2,1))
        x_final = x_final.permute(0,2,1)
        x_final = x_final[:,x_final.shape[1]//2,:]

        out = self.block(x_final)

        return out.squeeze(-1)

def read_fasta(file):
    seq = []
    label = []
    with open(file) as fasta:
        for line in fasta:
            line = line.replace('\n', '')
            if line.startswith('>'):
                # label.append(int(line[-1]))
                if 'neg' in line:
                    label.append(0)
                else:
                    label.append(1)
            else:
                seq.append(line.replace('U', 'T'))

    return seq, label

class MyDataSet(Dataset):
    def __init__(self, data, label, tokenizer):
        self.data = data  # ['今天天气很好', 'xxxx', ....]
        self.label = label  # [1, 0, 2]
        self.tokenizer = tokenizer

    def __getitem__(self, idx):
        # 获取原始文本和标签
        text = self.data[idx]  # str
        label = self.label[idx]

        return  text, label

    def __len__(self):
        return len(self.data)


def encode_sequence_1mer(sequences, max_seq):
    k = 1
    overlap = False

    all_kmer = [''.join(p) for p in itertools.product(['A', 'T', 'C', 'G', '-'], repeat=k)]
    kmer_dict = {all_kmer[i]: i for i in range(len(all_kmer))}

    encoded_sequences = []
    max_length = max_seq - k + 1 if overlap else max_seq // k

    for seq in sequences:
        encoded_seq = []
        start_site = len(seq) // 2 - max_length // 2
        for i in range(start_site, start_site + max_length, k):
            encoded_seq.append(kmer_dict.get(seq[i:i + k], 0))

        # 补齐到 max_length 长度
        encoded_sequences.append(encoded_seq + [0] * (max_length - len(encoded_seq)))

    return np.array(encoded_sequences)

def evaluation_method(params):
    train_x, train_y = read_fasta('data/train/train.fasta')
    valid_x, valid_y = read_fasta('data/valid/val.fasta')
    test_x, test_y = read_fasta('data/test/test.fasta')

    train_x, train_y = list(train_x), list(train_y)
    valid_x, valid_y = list(valid_x),list(valid_y)
    test_x, test_y = list(test_x),list(test_y)

    train_dataset = MyDataSet(train_x, train_y, tokenizer)
    valid_dataset = MyDataSet(valid_x, valid_y, tokenizer)
    test_dataset = MyDataSet(test_x, test_y, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], shuffle=False)

    train_model(train_loader, valid_loader, test_loader, params)

def to_log(log, params, start_time):
    # 创建保存路径
    seq_len = params['seq_len']
    seed = params['seed']
    log_dir = f"results/seq_len{seq_len}"
    os.makedirs(log_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹

    # 日志文件名包含 seed、seq_len 和开始时间
    log_path = f"{log_dir}/train_diff_len_seed{seed}_seq_len{seq_len}_{start_time}.log"

    # 写入日志
    with open(log_path, "a+") as f:
        f.write(log + '\n')

def train_model(train_loader, valid_loader, test_loader, params):
    # 获取训练开始的时间戳
    start_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    model = MCAMEFBERT(params).to(device)

    # Optimizer and loss
    opt = optim.Adam(model.parameters(), lr=params['lr'], betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
    criterion_BCE = nn.BCELoss()
    best_acc = 0
    patience = params['patience']
    now_epoch = 0

    best_model = None
    for epoch in range(params['epoch']):
        model.train()
        loss_ls = []
        t0 = time.time()
        for seq, label in tqdm(train_loader):
            seq, label = seq, label.to(device)

            output = model(seq)

            loss = criterion_BCE(output, label.float())

            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_ls.append(loss.item())

        model.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data = evaluate(train_loader, model)
            valid_performance, valid_roc_data, valid_prc_data = evaluate(valid_loader, model)

        results = f"\nepoch: {epoch + 1}, loss: {np.mean(loss_ls):.5f}\n"
        results += f'Train: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC, \tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],
            valid_performance[4], valid_performance[5]) + '\n' + '=' * 60
        valid_acc = valid_performance[0]  # test_performance: [ACC, Sensitivity, Specificity, AUC, MCC]
        print(results)
        to_log(results, params, start_time)


        if valid_acc > best_acc:
            best_acc = valid_acc
            now_epoch = 0
            best_model = copy.deepcopy(model)
            to_log('here occur best!\n', params, start_time)

            # 创建保存路径
            seq_len = params['seq_len']
            seed = params['seed']
            save_dir = f"save/seq_len{seq_len}"
            os.makedirs(save_dir, exist_ok=True)  # 如果文件夹不存在，创建文件夹

            # 文件名包含 seed、开始时间和 best_acc
            save_path = f"{save_dir}/seed{seed}_{start_time}_acc{best_acc:.4f}.pth"

            # 保存模型
            checkpoint = {
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch + 1,
                'params': params
            }
            torch.save(checkpoint, save_path)

            print(f"Checkpoint saved to {save_path}")

        else:
            now_epoch += 1
            print('now early stop target = ', now_epoch)
        test_performance, test_roc_data, test_prc_data = evaluate(test_loader, model)
        test_results = '\n' + '=' * 16 + colored(' Test Performance. Epoch[{}] ', 'red').format(
            epoch + 1) + '=' * 16 \
                       + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            test_performance[0], test_performance[1], test_performance[2], test_performance[3],
            test_performance[4], test_performance[5]) + '\n' + '=' * 60
        print(test_results)
        to_log(test_results, params, start_time)

        if now_epoch > patience:
            print('early stop!!!')
            best_performance, best_roc_data, best_prc_data = evaluate(test_loader, best_model)
            best_results = '\n' + '=' * 16 + colored(' Test Performance. Early Stop ', 'red').format(
                epoch + 1) + '=' * 16 \
                           + '\n[ACC,\tBACC, \tSE,\t\tSP,\t\tMCC,\tAUC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_performance[0], best_performance[1], best_performance[2], best_performance[3],
                best_performance[4], best_performance[5]) + '\n' + '=' * 60
            print(best_results)
            to_log(best_results, params, start_time)
            break


def caculate_metric(pred_prob, label_pred, label_real):
    test_num = len(label_real)
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for index in range(test_num):
        if label_real[index] == 1:
            if label_real[index] == label_pred[index]:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if label_real[index] == label_pred[index]:
                tn = tn + 1
            else:
                fp = fp + 1

    # Accuracy
    ACC = float(tp + tn) / test_num

    # Sensitivity
    if tp + fn == 0:
        Recall = Sensitivity = 0
    else:
        Recall = Sensitivity = float(tp) / (tp + fn)

    # Specificity
    if tn + fp == 0:
        Specificity = 0
    else:
        Specificity = float(tn) / (tn + fp)

    # MCC
    if (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) == 0:
        MCC = 0
    else:
        MCC = float(tp * tn - fp * fn) / np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)

    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)

    if (tp + fp) == 0:
        PRE = 0
    else:
        PRE = float(tp) / (tp + fp)

    BACC = 0.5 * Sensitivity + 0.5 * Specificity

    performance = [ACC, BACC, Sensitivity, Specificity, MCC, AUC]
    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AP]
    return performance, roc_data, prc_data

def evaluate(data_iter, net):
    net.eval()
    pred_prob_main = []
    pred_prob_cnn = []
    pred_prob_multi = []
    pred_prob_gru = []
    pred_prob_word = []
    label_real = []

    with torch.no_grad():
        for data, labels in data_iter:
            data, labels = data, labels.to(device)
            outputs_main = net(data)

            # 主输出的评估数据
            if outputs_main.dim() == 2 and outputs_main.shape[1] == 1:
                pred_prob_main.extend(outputs_main.squeeze(-1).cpu().numpy().tolist())
            else:
                pred_prob_main.extend(outputs_main.cpu().numpy().tolist())


            label_real.extend(labels.cpu().numpy().tolist())

    # 主输出的性能指标
    performance_main, roc_data_main, prc_data_main = caculate_metric(
        pred_prob_main, (np.array(pred_prob_main) > 0.5).astype(int).tolist(), label_real
    )


    return performance_main, roc_data_main, prc_data_main

def main():

    params = {
        'kernel_num': 4096,
        'topk': 128,
        'lr': 0.0001,
        'batch_size': 64,
        'epoch': 100,
        'seq_len': 201,
        'saved_model_name': 'diff_len_',
        'seed': 3407,
        'patience': 10
    }
    seed = params['seed']
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    evaluation_method(params)

if __name__ == '__main__':
    main()
"""
# -*- coding:utf-8 -*-
@Project : multiLabel
@File : multilabel_cls_fuse.py
@Author : FanJunyi
@Time : 2025/1/13 14:42

"""
'''
多模态融合后的多标签分类
'''

import os
os.environ['PATH'] = '/usr/local/cuda-11.3/bin:' + os.environ['PATH']
os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda-11.3/lib64:' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("show device %s" % device)

import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score
import pickle
import numpy as np

# 自定义数据集
class SampleDataset(Dataset):
    def __init__(self, names, labels):
        self.names = names
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.names[idx], self.labels[idx]

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_frames):
        super(CrossAttention, self).__init__()
        self.num_frames = num_frames
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)

    def forward(self, query, key, value):
        # fuse_output = {}
        # video_ids = query.keys()
        # for id in video_ids:
        #     # q =
        #     q = torch.tensor(np.expand_dims(query[id], axis = 1))
        #     k = torch.tensor(key[id])
        #     v = torch.tensor(value[id])
        #     attn_outputs = []
        #     for i in range(len(q)):
        #         attn_output, _ = self.multihead_attn(q[i], k, v)
        #         attn_outputs.append(attn_output)
        #     attn_outputs = torch.stack(attn_outputs)
        #     fuse_output[id] = attn_outputs
        query = query.unsqueeze(2)
        query = query.view(-1, query.shape[-2], query.shape[-1])
        key = key.view(-1, key.shape[-2], key.shape[-1])
        value = value.view(-1, value.shape[-2], value.shape[-1])
        fuse_output = []
        for idx in range(len(query)):
            attn_output, _ = self.multihead_attn(query[idx], key[idx], value[idx])
            fuse_output.append(attn_output)
        fuse_output = torch.cat(fuse_output, dim=0)
        fuse_output = fuse_output.view(-1, self.num_frames, fuse_output.shape[-1])
        return fuse_output

# 二分类头
class BinaryClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_frames = 8, num_classes = 8, dropout_rate=0.5):
        super(BinaryClassificationHead, self).__init__()
        self.num_frames = num_frames
        self.cross_attention = CrossAttention(embed_dim=256, num_heads=8, num_frames = self.num_frames)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        # self.sigmoid = nn.Sigmoid()

    def forward(self, query, key0, value0, key1, value1, key2, value2):
        key0 = key0.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        value0 = value0.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        key1 = key1.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        value1 = value1.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        key2 = key2.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        value2 = value2.unsqueeze(1).repeat(1, self.num_frames, 1, 1)
        x = self.cross_attention(query, key0, value0)
        x = self.cross_attention(x, key1, value1)
        x = self.cross_attention(x, key2, value2)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.sigmoid(x)
        x = torch.mean(x, dim=1)
        # x, _ = torch.max(x, dim=1)
        return x


# 测试函数
def test(model, dataloader, device):
    model.eval()
    all_names, all_preds, all_labels, all_score = [], [], [], []
    with torch.no_grad():
        for names, labels in dataloader:
            query_embeddings = [img_embeddings[name] for name in names]
            query_embeddings = np.array(query_embeddings)
            query_embeddings = torch.tensor(query_embeddings).to(device)

            # caption embeddings
            video_names = [name.split('_')[0] for name in names]
            kv0_embeddings = [caption_embeddings[name] for name in video_names]
            kv0_embeddings = np.array(kv0_embeddings)
            kv0_embeddings = torch.tensor(kv0_embeddings).to(device)

            # asr embeddings
            kv1_embeddings = [asr_embeddings[name] for name in video_names]
            kv1_embeddings = np.array(kv1_embeddings)
            kv1_embeddings = torch.tensor(kv1_embeddings).to(device)

            # ocr embeddings
            kv2_embeddings = [ocr_embeddings[name] for name in video_names]
            kv2_embeddings = np.array(kv2_embeddings)
            kv2_embeddings = torch.tensor(kv2_embeddings).to(device)

            outputs = model(query_embeddings, kv0_embeddings, kv0_embeddings,
                            kv1_embeddings, kv1_embeddings,
                            kv2_embeddings, kv2_embeddings).squeeze(1)
            # preds = torch.sigmoid(outputs).round().cpu().numpy()
            preds = (torch.sigmoid(outputs) > 0.5).float().cpu().numpy()
            scores = (torch.sigmoid(outputs)).float().cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_names.extend(names)
            all_score.extend(scores)

    # accuracy = accuracy_score(all_labels, all_preds)
    test_df = pd.DataFrame({'video': all_names, 'real_label': all_labels, 'pred_label': all_preds, 'pred_score': all_score})
    test_df['OE'] = test_df.apply(
        lambda row: np.max(row['pred_score'][np.where(np.atleast_1d(row['real_label']) == 0)[0]])
        if np.any(row['real_label'] == 0)  # 检查是否有值为 0 的元素
        else np.nan,  # 如果没有，返回 np.nan
        axis=1
    )
    test_df.to_csv("./test_result.csv", index=False, sep=',')

    # precision = precision_score(all_labels, all_preds, average='micro')
    # recall = recall_score(all_labels, all_preds, average='micro')
    precision = precision_score(all_labels, all_preds, average=None)
    recall = recall_score(all_labels, all_preds, average=None)
    accuracy = accuracy_score(all_labels, all_preds)
    oe = sum(list(test_df['OE'])) / len(list(test_df['OE']))
    return precision, recall, accuracy, oe


def process_data(data_label_path):
    train_df = pd.read_csv(os.path.join(data_label_path, "train_data.csv"), sep=',')
    train_names = train_df['video'].tolist()
    tmp = train_df[['blood', 'normal', 'sexy', 'smoke', 'violent', 'policy', 'abusive', 'money']]
    train_labels = torch.tensor(tmp.values, dtype=torch.int64)
    val_df = pd.read_csv(os.path.join(data_label_path, "val_data.csv"), sep=',')
    val_names = val_df['video'].tolist()
    tmp = val_df[['blood', 'normal', 'sexy', 'smoke', 'violent', 'policy', 'abusive', 'money']]
    val_labels = torch.tensor(tmp.values, dtype=torch.int64)
    # --------------------------
    # test_df = pd.read_csv(os.path.join(data_label_path, "test_data.csv"), sep=',')
    test_df = pd.read_csv(os.path.join(data_label_path, "test_data_fuse.csv"), sep=',')
    test_names = test_df['video'].tolist()
    tmp = test_df[['blood', 'normal', 'sexy', 'smoke', 'violent', 'policy', 'abusive', 'money']]
    test_labels = torch.tensor(tmp.values, dtype=torch.int64)

    train_dataset = SampleDataset(train_names, train_labels)
    val_dataset = SampleDataset(val_names, val_labels)
    test_dataset = SampleDataset(test_names, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

# 超参数
input_dim = 256
hidden_dim = 128
dropout_rate = 0.5
batch_size = 32
num_epochs = 15
learning_rate = 1e-3
num_classes = 8
num_frames = 8

# 数据准备，读取csv文件
data_label_path = "./"
train_loader, val_loader, test_loader = process_data(data_label_path)

with open("./frame_embeddings_total.pkl", 'rb') as f:
    img_embeddings = pickle.load(f)
    print("Loaded embeddings from embeddings.pkl")

with open("./caption_embeddings_total.pkl", 'rb') as f:
    caption_embeddings = pickle.load(f)
    print("Loaded embeddings from embeddings.pkl")

with open("./ocr_embeddings_blood.pkl", 'rb') as f:
    ocr_embeddings = pickle.load(f)
    print("Loaded embeddings from embeddings.pkl")

with open("./asr_embeddings_blood.pkl", 'rb') as f:
    asr_embeddings = pickle.load(f)
    print("Loaded embeddings from embeddings.pkl")

# 模型、损失函数、优化器
model = BinaryClassificationHead(input_dim, hidden_dim, num_frames, num_classes, dropout_rate).to(device)
model.load_state_dict(torch.load("./model/best_model_v2.pth"))

# criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
# optimizer = optim.Adam(model.parameters(), lr=learning_rate)



# 测试
precision, recall, accuracy, oe = test(model, test_loader, device)
print("blood | normal | sexy | smoke | violent | policy | abusive | money ")
precision = list(precision)
precision = [f"{num:.2f}" for num in precision]
print("Test Predictions:\n", precision)
recall = list(recall)
recall = [f"{num:.2f}" for num in recall]
print("Test Recall:\n", recall)
print("Test Accuracy:\n", accuracy)
print("Test One-Erorr:\n", oe)


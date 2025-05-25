#! /usr/bin/env python

import torch
import numpy as np
import scipy
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import os
import pandas as pd
from util import *
from model import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pytorchtools import EarlyStopping
import time

torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("cuda")
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
np.random.seed(123)

# Parameters
# ==================================================
NUM = 90        # number of graph nodes

parser = ArgumentParser("TSGR_wei", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--dataset", default="E:\lib\dHCP", help="Path of the dataset.")
parser.add_argument("--prj_dir", default=r"E:\OneDrive\MyUniversity\011-研究生\工作\project_SFC", help="Path of the project.")
parser.add_argument("--num_node", default=NUM, help="number of node")
parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")
parser.add_argument("--batch_size", default=16, type=int, help="Batch Size")
parser.add_argument("--num_epochs", default=500, type=int, help="Number of training epochs")
parser.add_argument("--num_samples", default=0.0, type=int, help="Weight decay")

parser.add_argument("--dropout", default=0.5, type=float, help="")
parser.add_argument("--lambda_reg", default=0.01, type=float, help="Regularization parameter")

parser.add_argument("--num_hidden_layers", default=1, type=int, help="Number of GCN layers")
parser.add_argument("--num_timesteps", default=1, type=int, help="Number of self-attention layers within each UGformer layer")
parser.add_argument("--ff_hidden_size", default=NUM * 2, type=int, help="The hidden size for the feedforward layer")

parser.add_argument('--nhead', type=int, default=3)
parser.add_argument('--encoder_self', type=int, default=1, help="Number of UGformer layers of encoder")
parser.add_argument('--decoder_self', type=int, default=1, help="Number of UGformer layers of decoder")
parser.add_argument('--model_name', default='ALL_data', type=str, help="The name of model")
args = parser.parse_args()
print(args)

# Load data
print("Loading data...")
data_path = os.path.join(args.dataset, 'FT_436.mat')
selector_path = os.path.join(args.prj_dir, "DATA\idx281.mat")
mat = scipy.io.loadmat(data_path)
sel_mat = scipy.io.loadmat(selector_path)
SC = np.transpose(mat['SC'], (2, 0, 1))
FC = np.transpose(mat['FC'], (2, 0, 1))
selector = sel_mat['idx281']
SC = np.squeeze(SC[selector-1])
FC = np.squeeze(FC[selector-1])
args.num_samples = SC.shape[0]

beha_data = pd.read_csv(os.path.join(args.prj_dir, "R/csv/PI_data.csv"))
beha_data = beha_data.iloc[:,7:10].to_numpy()

# data prepare
graphs = construct_graph(SC, FC)

feature_size = graphs[0].node_features.shape[1]
print(feature_size)

# batch_node_features, batch_edge_adj = get_batch_data(graphs[0:32])


mse_losses = []
predictions = []
data_idx = np.linspace(0, args.num_samples-1, args.num_samples, dtype=int)
for cv in range(args.num_samples):
    model = TSGR(input_size=feature_size, feature_embedding_size=64, hidden_size=1, output_size=3, nhead=4).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    test_idx = data_idx[cv]
    dataset_idx = np.delete(data_idx, cv)

    train_idx, val_idx, _, _ = train_test_split(dataset_idx, dataset_idx, test_size=0.2, random_state=42)

    save_path = f"./checkpoints/EarlyStoppingCheckpoint_{cv:03d}.pth"
    early_stopping = EarlyStopping(patience=20, verbose=False, delta=0.000,
                                   path=save_path)

    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()
        # define batch data
        train_graphs = [graphs[i] for i in train_idx]
        batch_data = get_batch_data(train_graphs, beha_data[train_idx], args.batch_size, train_state=True)

        train_losses = []

        model.train()
        for batch_idx, (node_batch, edge_batch, y_batch) in enumerate(batch_data):
            outputs = model(node_batch, edge_batch)
            loss = criterion(outputs, y_batch)
            train_losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validation
        model.eval()

        val_graph = [graphs[i] for i in val_idx]
        batch_data = get_batch_data(val_graph, beha_data[val_idx], args.batch_size, train_state=False)
        val_losses = []
        with torch.no_grad():
            for batch_idx, (node_batch, edge_batch, y_batch) in enumerate(batch_data):
                outputs = model(node_batch, edge_batch)
                val_loss = criterion(outputs, y_batch)
                val_losses.append(val_loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)

        early_stopping(val_loss, model)
        # 达到早停止条件时，early_stop会被置为True
        if early_stopping.early_stop:
            print("Early stopping")
            break  # 跳出迭代，结束训练

        print('| CV{:3d} | epoch{:3d} | time: {:5.2f}s | train_loss {:5.2f} | val_loss {:5.2f} |'.format(
            cv, epoch, (time.time() - epoch_start_time), train_loss, val_loss))

    # output cv results
    model.eval()
    with torch.no_grad():
        tensor_node = torch.tensor(graphs[test_idx].node_features, dtype=torch.float32).to(device)
        tensor_adj = torch.tensor(graphs[test_idx].adj, dtype=torch.float32).to(device)
        y_pred = model(tensor_node.unsqueeze(0), tensor_adj)
        # mse = criterion(y_pred, tensor_cvy)
        predictions.append(y_pred.squeeze().cpu().numpy())

# output results
predictions = np.array(predictions)
final_predictions = predictions # * std_y + mean_y
r = np.zeros(3)

def pearson_correlation(output, target):
    vx = output - np.mean(output) + 1e-8
    vy = target - np.mean(target) + 1e-8
    cost = np.sum(vx * vy) / (np.sqrt(np.sum(vx ** 2)) * np.sqrt(np.sum(vy ** 2)))
    return cost

r[0] = pearson_correlation(final_predictions[:,0], beha_data[:,0])
r[1] = pearson_correlation(final_predictions[:,1], beha_data[:,1])
r[2] = pearson_correlation(final_predictions[:,2], beha_data[:,2])

print(f'pearson r:{r}\n')

plt.scatter(beha_data[:,0], final_predictions[:, 0], c='blue')
plt.scatter(beha_data[:,1], final_predictions[:, 1], c='red')
plt.scatter(beha_data[:,2], final_predictions[:, 2], c='green')
plt.show()


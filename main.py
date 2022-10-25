import numpy as np
from sklearn.preprocessing import MinMaxScaler
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import argparse
from sklearn.utils import shuffle
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy
import numba as nb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
metric_accuracy_1 = Accuracy()
metric_accuracy_2 = Accuracy()
metric_mean_1 = MeanMetric()
metric_mean_2 = MeanMetric()
metric_mean_3 = MeanMetric()
metric_mean_4 = MeanMetric()
metric_mean_5 = MeanMetric()
metric_mean_6 = MeanMetric()
metric_mean_7 = MeanMetric()

def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--data_dir', type=str, default="data\\5HP", help='')
    parser.add_argument("--pretrained", type=bool, default=True, help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batchsize of the training process')
    parser.add_argument('--step_len', type=list, default=range(210, 430, 10), help='')
    parser.add_argument('--sample_len', type=int, default=420, help='')
    parser.add_argument('--rate', type=list, default=[0.7, 0.15, 0.15], help='')
    parser.add_argument('--acces', type=list, default=[], help='initialization list')
    parser.add_argument('--epochs', type=int, default=80, help='max number of epoch')
    parser.add_argument('--losses', type=list, default=[], help='initialization list')
    parser.add_argument('--decay_rate', type=float, default=0.9, help='initialization list')
    parser.add_argument('--step', type=list, default=[], help='initialization list')
    parser.add_argument('--decay_steps', type=list, default=10, help='initialization list')
    args = parser.parse_args()
    return args


class Dataset(da.Dataset):
    def __init__(self, X, y):
        self.Data = X
        self.Label = y
    def __getitem__(self, index):
        txt = self.Data[index]
        label = self.Label[index]
        return txt, label
    def __len__(self):
        return len(self.Data)

def load_data(batch_size=256):
    source_data = np.load("H:\\DDTLN\\12k_de_cwru_data_load2_0.007.npy")
    source_label = np.load("H:\\DDTLN\\12k_de_cwru_label_load2_0.007.npy")
    target_data = np.load("H:\\DDTLN\\gt300data_data.npy")
    target_label = np.load("H:\\DDTLN\\gt300data_label.npy")
    source_data = MinMaxScaler().fit_transform(source_data.T).T
    target_data = MinMaxScaler().fit_transform(target_data.T).T
    source_data = np.expand_dims(source_data, axis=1).astype('float32')
    target_data = np.expand_dims(target_data, axis=1).astype('int64')
    X_source, Y_source = shuffle(source_data, source_label, random_state=2)
    X_target, Y_target = shuffle(target_data, target_label, random_state=2)
    Train_source = Dataset(torch.from_numpy(X_source), Y_source)
    Train_target = Dataset(torch.from_numpy(X_target), Y_target)
    source_loader = da.DataLoader(Train_source, batch_size=batch_size, shuffle=True)
    target_loader = da.DataLoader(Train_target, batch_size=batch_size, shuffle=True)
    return source_loader, target_loader


###############################################################

@nb.jit(nopython=True)
def mix_rbf_mmd2(m, n, X, Y, sigmas=(1,), wts=None, biased=True):
    K_XX, K_XY, K_YY, d = _mix_rbf_kernel(X, Y, sigmas, wts)
    return _mmd2(m, n, K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)


def _mix_rbf_kernel(X, Y, sigmas, wts=None):
    if wts is None:
        wts = [1] * len(sigmas)

    XX = np.matmul(X, X, transpose_b=True)
    XY = np.matmul(X, Y, transpose_b=True)
    YY = np.matmul(Y, Y, transpose_b=True)

    X_sqnorms = np.diag(XX)
    Y_sqnorms = np.diag(YY)

    r = lambda x: np.expand_dims(x, 0)
    c = lambda x: np.expand_dims(x, 1)

    K_XX, K_XY, K_YY = 0, 0, 0
    for sigma, wt in zip(sigmas, wts):
        gamma = 1 / (2 * sigma ** 2)
        K_XX += wt * np.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
        K_XY += wt * np.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
        K_YY += wt * np.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
        return K_XX, K_XY, K_YY, np.sum(wts)


def _mmd2(m, n, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
    if biased:
        mmd2 = (np.sum(K_XX) / (m * m)
                + np.sum(K_YY) / (n * n)
                - 2 * np.sum(K_XY) / (m * n))
    else:
        if const_diagonal is not False:
            trace_X = m * const_diagonal
            trace_Y = n * const_diagonal
        else:
            trace_X = np.trace(K_XX)
            trace_Y = np.trace(K_YY)

        mmd2 = ((np.sum(K_XX) - trace_X) / (m * (m - 1))
                + (np.sum(K_YY) - trace_Y) / (n * (n - 1))
                - 2 * np.sum(K_XY) / (m * n))
    return mmd2


def MDA(m, n, source, target, bandwidths=[1]):
    kernel_loss = mix_rbf_mmd2(m, n, source, target, sigmas=bandwidths) * 100
    eps = 1e-5
    d = source.shape[1]
    ns, nt = source.shape[0], target.shape[0]

    # source covariance
    tmp_s = np.matmul(np.ones(shape=(1, ns)), source)
    cs = (np.matmul(np.transpose(source), source) - np.matmul(np.transpose(tmp_s), tmp_s) / ns) / (ns - 1 + eps) * (
                ns / m)

    # target covariance
    tmp_t = np.matmul(np.ones(shape=(1, nt)), target)
    ct = (np.matmul(np.transpose(target), target) - np.matmul(np.transpose(tmp_t), tmp_t) / nt) / (nt - 1 + eps) * (
                nt / n)

    # frobenius norm
    loss = np.sqrt(np.reduce_sum(np.pow((cs - ct), 2)))
    loss = loss / (4 * d * d) * 10

    # if np.isnan(loss.numpy()):
    #    print(m,n,ns,nt,d,cs.numpy(),ct.numpy(),tmp_s.numpy(),tmp_t.numpy(),loss.numpy(),kernel_loss.numpy())

    return loss + kernel_loss


def classification_division(data, label):
    label = np.argmax(label, axis=-1)
    N = data.shape[0]
    a, b, c, d, e = [], [], [], [], []
    for i in range(N):
        if label[i] == 0:
            a.append(data[i])
        elif label[i] == 1:
            b.append(data[i])
        elif label[i] == 2:
            c.append(data[i])
        elif label[i] == 3:
            d.append(data[i])
    return np.array(a), np.array(b), np.array(c), np.array(d)


def CDA(m, n, output1, source_label, output2, pseudo_label):
    s_0, s_1, s_2, s_3 = classification_division(output1, source_label)
    t_0, t_1, t_2, t_3 = classification_division(output2, pseudo_label)

    CDA_loss = 0
    # print("*"*50)
    if t_0.shape[0] != 0:
        # print("***1***",MDA(m,n,s_0,t_0).numpy())
        CDA_loss += MDA(m, n, s_0, t_0)
    if t_1.shape[0] != 0:
        # print("***2***",MDA(m,n,s_1,t_1).numpy())
        CDA_loss += MDA(m, n, s_1, t_1)
    if t_2.shape[0] != 0:
        # print("***3***",MDA(m,n,s_2,t_2).numpy())
        CDA_loss += MDA(m, n, s_2, t_2)
    if t_3.shape[0] != 0:
        # print("***4***",MDA(m,n,s_3,t_3).numpy())
        CDA_loss += MDA(m, n, s_3, t_3)
    # print("*"*50)
    return CDA_loss / 4.

#############################################################33

def class_angle(m, n, a, la, da):
    if len(la) == 0:
        return a
    else:
        index = np.argmax(la[0], axis=-1)
    for i in range(len(a)):
        c = a[i]
        part1 = c[:index]
        part2 = c[index + 1:]
        if c[index] > 0:
            val = [c[index] / m - n]
        elif c[index] <= 0:
            val = [c[index] * m - n]
        if i == 0:
            new_tensor = np.concatenate([part1, val, part2], axis=0)
        else:
            tensor = np.concatenate([part1, val, part2], axis=0)
            new_tensor = np.vstack([new_tensor, tensor])

    return new_tensor


def I_Softmax(m, n, source_out1, source_output1, source_label):
    label_argmax = np.argmax(source_label, axis=-1)
    la, lb, lc, ld, le = [], [], [], [], []
    a, b, c, d, e = [], [], [], [], []
    da, db, dc, dd, de = [], [], [], [], []
    for i in range(source_label.shape[0]):
        if label_argmax[i] == 0:
            a.append(source_output1[i])
            la.append(source_label[i])
            da.append(source_out1[i])
        elif label_argmax[i] == 1:
            b.append(source_output1[i])
            lb.append(source_label[i])
            db.append(source_out1[i])
        elif label_argmax[i] == 2:
            c.append(source_output1[i])
            lc.append(source_label[i])
            dc.append(source_out1[i])
        elif label_argmax[i] == 3:
            d.append(source_output1[i])
            ld.append(source_label[i])
            dd.append(source_out1[i])
        elif label_argmax[i] == 4:
            e.append(source_output1[i])
            le.append(source_label[i])
            de.append(source_out1[i])
    # print(b[0])
    b = torch.from_numpy(class_angle(m, n, b, lb, db))
    a = torch.from_numpy(class_angle(m, n, a, la, da))
    c = torch.from_numpy(class_angle(m, n, c, lc, dc))
    d = torch.from_numpy(class_angle(m, n, d, ld, dd))
    e = torch.from_numpy(class_angle(m, n, e, le, de))
    # print(b[0])

    data_set = []
    label_set = []
    if len(a) != 0:
        data_set.append(a)
        label_set.append(la)
    if len(b) != 0:
        data_set.append(b)
        label_set.append(lb)
    if len(c) != 0:
        data_set.append(c)
        label_set.append(lc)
    if len(d) != 0:
        data_set.append(d)
        label_set.append(ld)
    if len(e) != 0:
        data_set.append(e)
        label_set.append(le)

    data = np.vstack(data_set)
    data1 = torch.vstack(data_set)
    label = np.vstack(label_set)
    label1 = torch.vstack(label_set)
    loss = criterion(label, data)
    return torch.from_numpy(data), torch.from_numpy(label), loss

###################################################################################################3

class DDTLN(nn.Module):
    def __init__(self):
        super(DDTLN, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU(inplace=True))
        self.p1_2 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU(inplace=True))
        self.p2_2 = nn.MaxPool1d(2, 2)
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU(inplace=True))
        self.p3_2 = nn.MaxPool1d(2, 2)
        self.p4_1 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU(inplace=True))
        self.p4_2 = nn.MaxPool1d(2, 2)
        self.p5_1 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True))
        self.p5_2 = nn.MaxPool1d(2, 2)
        self.p6 = nn.AdaptiveAvgPool1d(1)
        self.p7_1 = nn.Sequential(nn.Linear(512, 512),     #需要根据输出修改前置神经元个数
                                  nn.ReLU(inplace=True))   #全连接层之后还需要加激活函数
        self.p7_2 = nn.Sequential(nn.utils.weight_norm(nn.Linear(512, 4), name='weight'))

    def forward(self, x):
        x = self.p1_2(self.p1_1(x))
        x = self.p2_2(self.p2_1(x))
        x = self.p3_2(self.p3_1(x))
        x = self.p4_2(self.p4_1(x))
        x = self.p5_2(self.p5_1(x))
        x1 = self.p7_1(self.p6(x).squeeze())
        x2 = self.p7_2(x1)
        return x1, x2

model = DDTLN().to(device)

def train(model, source_loader, target_loader, optimizer):
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)


    for i in range(0, num_iter):
        source_data, source_label = next(iter_source)
        target_data, target_label = next(iter_target)
        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data = target_data.cuda()
        m = source_data.shape[0]
        n = target_data.shape[0]
        optimizer.zero_grad()
        out1, output1 = model(source_data)
        out2, output2 = model(target_data)
        data, label, clc_loss_step = I_Softmax(3, 16, out1, output1, source_label)
        pre_pseudo_label = nn.functional.one_hot(torch.argmax(output2, dim=-1), num_classes=target_label.shape[-1])
        pseudo_data, pseudo_label, pseudo_loss_step = I_Softmax(3, 16, out2, output2, pre_pseudo_label)
        CDA_loss = CDA(m, n, output1, source_label, output2, pre_pseudo_label)
        loss_step = clc_loss_step + MDA(m, n, output1, output2) + pseudo_loss_step * 0.1 + CDA_loss * 0.1
        loss_step.backward()
        optimizer.step()
        metric_accuracy_1.update(data, label.long())
        metric_accuracy_2.update(target_label,output2)

        metric_mean_1.update(loss_step)
        metric_mean_2.update(criterion(output1, source_label))
        metric_mean_3.update(criterion(output2, target_label))
        metric_mean_4.update(pseudo_loss_step)
        metric_mean_5.update(clc_loss_step)
        metric_mean_6.update(CDA_loss)
        metric_mean_7.update(MDA(m,n,output1,output2))
    train_acc = metric_accuracy_1.compute()
    test_acc = metric_accuracy_2.compute()
    train_all_loss = metric_mean_1.compute()
    train_loss = metric_mean_2.compute()
    test_loss = metric_mean_3.compute()
    target_cla_loss = metric_mean_4.compute()
    source_cla_loss = metric_mean_5.compute()
    cda_loss = metric_mean_6.compute()
    mda_loss = metric_mean_7.compute()
    metric_accuracy_1.reset()
    metric_accuracy_2.reset()
    metric_mean_1.reset()
    metric_mean_2.reset()
    metric_mean_3.reset()
    metric_mean_4.reset()
    metric_mean_5.reset()
    metric_mean_6.reset()
    metric_mean_7.reset()
    return train_acc, test_acc, train_all_loss, train_loss, test_loss, target_cla_loss, source_cla_loss, cda_loss, mda_loss


if __name__ == '__main__':
    seed_everything(2022)
    args = parse_args()
    t_test_acc = 0.0
    stop = 0
    criterion = nn.CrossEntropyLoss()
    source_loader, target_loader = load_data(batch_size=args.batch_size)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001, weight_decay=0.0001)
    for epoch in range(0, args.nepoch):
        stop += 1
        train_acc, test_acc, train_all_loss, train_loss, test_loss, target_cla_loss, source_cla_loss, cda_loss, mda_loss = train(
            model, source_loader, target_loader, optimizer)
        if t_test_acc > test_loss:
            test_loss = t_test_acc
            stop = 0
            torch.save(model, 'model.pkl')
        if epoch % 10 == 0:
            print(
                'Epoch{}, train_loss is {:.5f},test_loss is {:.5f}, train_accuracy is {:.5f},test_accuracy is {:.5f},train_all_loss is {:.5f},target_cla_loss is {:.5f},source_cla_loss is {:.5f},cda_loss is {:.5f},mda_loss is {:.5f}'.format(
                    epoch + 1,
                    train_loss,
                    test_loss,
                    train_acc,
                    test_acc,
                    train_all_loss,
                    target_cla_loss,
                    source_cla_loss,
                    cda_loss,
                    mda_loss))
            break
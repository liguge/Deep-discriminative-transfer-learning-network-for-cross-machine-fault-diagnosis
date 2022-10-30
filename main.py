import numpy as np
from sklearn.preprocessing import StandardScaler
from pytorch_lightning.utilities.seed import seed_everything
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils import data as da
from torchmetrics import MeanMetric, Accuracy


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--cwru_data', type=str, default="H:\\DDTLN\\gt300data_data.npy", help='')
    parser.add_argument('--cwru_label', type=str, default="H:\\DDTLN\\gt300data_label.npy", help='')
    parser.add_argument('--bjtu_data', type=str, default="H:\\DDTLN\\gt160data_data.npy", help='')
    parser.add_argument('--bjtu_label', type=str, default="H:\\DDTLN\\gt160data_label.npy", help='')
    parser.add_argument('--batch_size', type=int, default=256, help='batchsize of the training process')
    parser.add_argument('--nepoch', type=int, default=100, help='max number of epoch')
    parser.add_argument('--s_m', type=int, default=3, help='')
    parser.add_argument('--s_n', type=int, default=16, help='')
    parser.add_argument('--num_classes', type=int, default=4, help='')
    parser.add_argument('--lr', type=float, default=0.001, help='')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='initialization list')
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


def load_data():
    source_data = np.load(args.cwru_data)
    source_label = np.load(args.cwru_label).argmax(axis=-1)
    target_data = np.load(args.bjtu_data)
    target_label = np.load(args.bjtu_label).argmax(axis=-1)
    source_data = StandardScaler().fit_transform(source_data.T).T
    target_data = StandardScaler().fit_transform(target_data.T).T
    source_data = np.expand_dims(source_data, axis=1)
    target_data = np.expand_dims(target_data, axis=1)
    Train_source = Dataset(source_data, source_label)
    Train_target = Dataset(target_data, target_label)
    return Train_source, Train_target


###############################################################
class MMD(nn.Module):
    def __init__(self, m, n):
        super(MMD, self).__init__()
        self.m = m
        self.n = n

    def _mix_rbf_mmd2(self, X, Y, sigmas=(10,), wts=None, biased=True):
        K_XX, K_XY, K_YY, d = self._mix_rbf_kernel(X, Y, sigmas, wts)
        return self._mmd2(K_XX, K_XY, K_YY, const_diagonal=d, biased=biased)

    def _mix_rbf_kernel(self, X, Y, sigmas, wts=None):
        if wts is None:
            wts = [1] * len(sigmas)
        XX = torch.matmul(X, X.t())
        XY = torch.matmul(X, Y.t())
        YY = torch.matmul(Y, Y.t())

        X_sqnorms = torch.diagonal(XX)
        Y_sqnorms = torch.diagonal(YY)

        r = lambda x: torch.unsqueeze(x, 0)
        c = lambda x: torch.unsqueeze(x, 1)

        K_XX, K_XY, K_YY = 0., 0., 0.
        for sigma, wt in zip(sigmas, wts):
            gamma = 1 / (2 * sigma ** 2)
            K_XX += wt * torch.exp(-gamma * (-2 * XX + c(X_sqnorms) + r(X_sqnorms)))
            K_XY += wt * torch.exp(-gamma * (-2 * XY + c(X_sqnorms) + r(Y_sqnorms)))
            K_YY += wt * torch.exp(-gamma * (-2 * YY + c(Y_sqnorms) + r(Y_sqnorms)))
            return K_XX, K_XY, K_YY, torch.sum(torch.tensor(wts))

    def _mmd2(self, K_XX, K_XY, K_YY, const_diagonal=False, biased=False):
        if biased:
            mmd2 = torch.sum(K_XX) / (self.m * self.m) + torch.sum(K_YY) / (self.n * self.n)
            - 2 * torch.sum(K_XY) / (self.m * self.n)
        else:
            if const_diagonal is not False:
                trace_X = self.m * const_diagonal
                trace_Y = self.n * const_diagonal
            else:
                trace_X = torch.trace(K_XX)
                trace_Y = torch.trace(K_YY)

            mmd2 = ((torch.sum(K_XX) - trace_X) / (self.m * (self.m - 1))
                    + (torch.sum(K_YY) - trace_Y) / (self.n * (self.n - 1))
                    - 2 * torch.sum(K_XY) / (self.m * self.n))
        return mmd2

    def _if_list(self, list):
        if len(list) == 0:
            list = torch.tensor(list)
        else:
            list = torch.vstack(list)
        return list

    def _classification_division(self, data, label):
        N = data.size()[0]
        a, b, c, d = [], [], [], []
        for i in range(N):
            if label[i] == 0:
                a.append(data[i])
            elif label[i] == 1:
                b.append(data[i])
            elif label[i] == 2:
                c.append(data[i])
            elif label[i] == 3:
                d.append(data[i])

        return self._if_list(a), self._if_list(b), self._if_list(c), self._if_list(d)

    def MDA(self, source, target, bandwidths=[10]):
        kernel_loss = self._mix_rbf_mmd2(source, target, sigmas=bandwidths) * 100.
        eps = 1e-5
        d = source.size()[1]
        ns, nt = source.size()[0], target.size()[0]

        # source covariance
        tmp_s = torch.matmul(torch.ones(size=(1, ns)).cuda(), source)
        cs = (torch.matmul(torch.t(source), source) - torch.matmul(torch.t(tmp_s), tmp_s) / (ns + eps)) / (ns - 1 + eps) * (
                     ns / (self.m))

        # target covariance
        tmp_t = torch.matmul(torch.ones(size=(1, nt)).cuda(), target)
        ct = (torch.matmul(torch.t(target), target) - torch.matmul(torch.t(tmp_t), tmp_t) / (nt + eps)) / (
                    nt - 1 + eps)* (
                     nt / (self.n))
        # frobenius norm
        # loss = torch.sqrt(torch.sum(torch.pow((cs - ct), 2)))
        loss = torch.norm((cs - ct))
        loss = loss / (4 * d * d) * 10.

        return loss + kernel_loss

    def CDA(self, output1, source_label, output2, pseudo_label):
        s_0, s_1, s_2, s_3 = self._classification_division(output1, source_label)
        t_0, t_1, t_2, t_3 = self._classification_division(output2, pseudo_label)

        CDA_loss = 0.
        if t_0.size()[0] != 0:
            CDA_loss += self.MDA(s_0, t_0)
        if t_1.size()[0] != 0:
            CDA_loss += self.MDA(s_1, t_1)
        if t_2.size()[0] != 0:
            CDA_loss += self.MDA(s_2, t_2)
        if t_3.size()[0] != 0:
            CDA_loss += self.MDA(s_3, t_3)
        return CDA_loss / 4.

#############################################################33
class I_Softmax(nn.Module):

    def __init__(self, m, n, source_output1, source_label):
        super().__init__()
        self.m = torch.tensor([m]).cuda()
        self.n = torch.tensor([n]).cuda()
        self.source_output1 = source_output1
        self.source_label = source_label
        self.la, self.lb, self.lc, self.ld = [], [], [], []
        self.a, self.b, self.c, self.d = [], [], [], []
        self.data_set = []
        self.label_set = []

    def _combine(self):
        for i in range(self.source_label.size()[0]):
            if self.source_label[i] == 0:
                self.a.append(self.source_output1[i])
                self.la.append(self.source_label[i])
            elif self.source_label[i] == 1:
                self.b.append(self.source_output1[i])
                self.lb.append(self.source_label[i])
            elif self.source_label[i] == 2:
                self.c.append(self.source_output1[i])
                self.lc.append(self.source_label[i])
            elif self.source_label[i] == 3:
                self.d.append(self.source_output1[i])
                self.ld.append(self.source_label[i])

        a = self._class_angle(self.a, self.la)
        b = self._class_angle(self.b, self.lb)
        c = self._class_angle(self.c, self.lc)
        d = self._class_angle(self.d, self.ld)

        if len(a) != 0:
            self.data_set.append(a)
            self.label_set.append(torch.tensor(self.la).unsqueeze(1))
        if len(b) != 0:
            self.data_set.append(b)
            self.label_set.append(torch.tensor(self.lb).unsqueeze(1))
        if len(c) != 0:
            self.data_set.append(c)
            self.label_set.append(torch.tensor(self.lc).unsqueeze(1))
        if len(d) != 0:
            self.data_set.append(d)
            self.label_set.append(torch.tensor(self.ld).unsqueeze(1))
        data = torch.vstack(self.data_set)
        label = torch.vstack(self.label_set)
        return data.cuda(), label.squeeze().cuda()

    def _class_angle(self, a, la):

        if len(la) == 0:
            return a
        else:
            index = la[0]
        for i in range(len(a)):
            c = a[i]
            part1 = c[:index]
            part2 = c[index + 1:]
            if c[index] > 0:
                val = c[index] / (self.m + 1e-5) - self.n
            elif c[index] <= 0:
                val = c[index] * (self.m + 1e-5) - self.n
            if i == 0:
                new_tensor = torch.concat((part1, val, part2))
            else:
                tensor = torch.concat((part1, val, part2), dim=0)
                new_tensor = torch.vstack([new_tensor, tensor])

        return new_tensor

    def forward(self):
        data, label = self._combine()
        loss = F.nll_loss(F.log_softmax(data, dim=-1), label)
        return data, label, loss


###################################################################################################3

class DDTLN(nn.Module):
    def __init__(self):
        super(DDTLN, self).__init__()
        self.p1_1 = nn.Sequential(nn.Conv1d(1, 32, kernel_size=64, stride=16, padding=24),
                                  nn.BatchNorm1d(32),
                                  nn.ReLU())
        self.p1_2 = nn.MaxPool1d(2, 2)
        self.p2_1 = nn.Sequential(nn.Conv1d(32, 64, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(64),
                                  nn.ReLU())
        self.p2_2 = nn.MaxPool1d(2, 2)
        self.p3_1 = nn.Sequential(nn.Conv1d(64, 128, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(128),
                                  nn.ReLU())
        self.p3_2 = nn.MaxPool1d(2, 2)
        self.p4_1 = nn.Sequential(nn.Conv1d(128, 256, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(256),
                                  nn.ReLU())
        self.p4_2 = nn.MaxPool1d(2, 2)
        self.p5_1 = nn.Sequential(nn.Conv1d(256, 512, kernel_size=3, stride=1, padding='same'),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU())
        self.p5_2 = nn.MaxPool1d(2, 2)
        self.p6 = nn.AdaptiveAvgPool1d(1)
        self.p7_1 = nn.Sequential(nn.Linear(512, 512),
                                  nn.ReLU())
        self.p7_2 = nn.Sequential(nn.Linear(512, 4))
        # self._weights_init()

    def forward(self, x, y):
        x = self.p1_2(self.p1_1(x))
        x = self.p2_2(self.p2_1(x))
        x = self.p3_2(self.p3_1(x))
        x = self.p4_2(self.p4_1(x))
        x = self.p5_2(self.p5_1(x))
        x = self.p7_1(self.p6(x).squeeze())
        x = self.p7_2(x)
        y = self.p1_2(self.p1_1(y))
        y = self.p2_2(self.p2_1(y))
        y = self.p3_2(self.p3_1(y))
        y = self.p4_2(self.p4_1(y))
        y = self.p5_2(self.p5_1(y))
        y = self.p7_1(self.p6(y).squeeze())
        y = self.p7_2(y)
        return x, y

    def predict(self, y):
        y = self.p1_2(self.p1_1(y))
        y = self.p2_2(self.p2_1(y))
        y = self.p3_2(self.p3_1(y))
        y = self.p4_2(self.p4_1(y))
        y = self.p5_2(self.p5_1(y))
        y = self.p7_1(self.p6(y).squeeze())
        y = self.p7_2(y)
        return y

    def _weights_init(self):
        for L in self.modules():
            if isinstance(L, nn.Conv1d):
                n = L.kernel_size[0] * L.out_channels
                L.weight.data.normal_(mean=0, std=np.sqrt(2.0 / float(n)))
            elif isinstance(L, nn.BatchNorm1d):
                L.weight.data.fill_(1)
                L.bias.data.fill_(0)
            elif isinstance(L, nn.Linear):
                L.weight.data.normal_(0, 0.01)
                if L.bias is not None:
                    L.bias.data.fill_(1)


losses = []


def train(model, source_loader, target_loader, optimizer):
    model.train()
    iter_source = iter(source_loader)
    iter_target = iter(target_loader)
    num_iter = len(source_loader)
    for i in range(0, num_iter):
        source_data, source_label = next(iter_source)
        target_data, _ = next(iter_target)
        source_data, source_label = source_data.cuda(), source_label.cuda()
        target_data = target_data.cuda()
        optimizer.zero_grad()
        output1, output2 = model(source_data.float(), target_data.float())
        data, label, clc_loss_step = I_Softmax(args.s_m, args.s_n, output1, source_label).forward()
        pre_pseudo_label = torch.argmax(output2, dim=-1)
        pseudo_data, pseudo_label, pseudo_loss_step = I_Softmax(args.s_m, args.s_n, output2, pre_pseudo_label).forward()
        CDA_loss = MMD(source_data.size()[0], target_data.size()[0]).CDA(output1, source_label, output2, pre_pseudo_label)
        MDA_loss = MMD(source_data.size()[0], target_data.size()[0]).MDA(output1, output2)
        loss_step = clc_loss_step + (MDA_loss + 0.1 * CDA_loss) + 0.1 * pseudo_loss_step
        loss_step.backward()
        optimizer.step()
        metric_accuracy_1.update(output1.max(1)[1], source_label)
        # metric_accuracy_2.update(output2.max(1)[1], target_label)
        metric_mean_1.update(loss_step)
        metric_mean_2.update(criterion(output1, source_label))
        # metric_mean_3.update(criterion(output2, target_label))
        metric_mean_4.update(pseudo_loss_step)
        metric_mean_5.update(clc_loss_step)
        metric_mean_6.update(CDA_loss)
        metric_mean_7.update(MDA_loss)
    train_acc = metric_accuracy_1.compute()
    ###############################################
    # test_acc = metric_accuracy_2.compute()
    #################################################
    train_all_loss = metric_mean_1.compute()  # loss_step
    train_loss = metric_mean_2.compute()
    # test_loss = metric_mean_3.compute()
    target_cla_loss = metric_mean_4.compute()
    #################################################
    source_cla_loss = metric_mean_5.compute()
    #################################################
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
    return train_acc, train_all_loss, train_loss, target_cla_loss, source_cla_loss, cda_loss, mda_loss

def test(model, target_loader):
    model.eval()
    iter_target = iter(target_loader)
    num_iter = len(target_loader)
    for i in range(0, num_iter):
        target_data, target_label = next(iter_target)
        target_data, target_label = target_data.cuda(), target_label.cuda()
        output2 = model.predict(target_data.float())
        metric_accuracy_2.update(output2.max(1)[1], target_label)
        metric_mean_3.update(criterion(output2, target_label))
    test_acc = metric_accuracy_2.compute()
    test_loss = metric_mean_3.compute()
    metric_accuracy_2.reset()
    metric_mean_3.reset()
    return test_acc, test_loss

if __name__ == '__main__':
    seed_everything(42)
    args = parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metric_accuracy_1 = Accuracy().cuda()
    metric_accuracy_2 = Accuracy().cuda()
    metric_mean_1 = MeanMetric().cuda()
    metric_mean_2 = MeanMetric().cuda()
    metric_mean_3 = MeanMetric().cuda()
    metric_mean_4 = MeanMetric().cuda()
    metric_mean_5 = MeanMetric().cuda()
    metric_mean_6 = MeanMetric().cuda()
    metric_mean_7 = MeanMetric().cuda()
    t_test_acc = 0.0
    stop = 0
    Train_source, Train_target = load_data()
    g = torch.Generator()
    source_loader = da.DataLoader(dataset=Train_source, batch_size=args.batch_size, shuffle=True, generator=g)
    g = torch.Generator()
    target_loader = da.DataLoader(dataset=Train_target, batch_size=args.batch_size, shuffle=True, generator=g)
    model = DDTLN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    for epoch in range(0, args.nepoch):
        stop += 1
        train_acc, train_all_loss, train_loss, target_cla_loss, source_cla_loss, cda_loss, mda_loss = train(
            model, source_loader, target_loader, optimizer)
        # if epoch == 1:
        #     for name, parms in model.named_parameters():
        #         print('-->name:', name)
        #         print('-->para:', parms)
        #         print('-->grad_requirs:', parms.requires_grad)
        #         print('-->grad_value:', parms.grad)
        #         print("===")
        # if t_test_acc > test_acc:
        #     test_acc = t_test_acc
        #     stop = 0
        #     torch.save(model, 'model.pkl')
        test_acc, test_loss = test(model, target_loader)
        print(
            'Epoch{}, train_loss is {:.5f},test_loss is {:.5f}, train_accuracy is {:.5f},test_accuracy is {:.5f},train_all_loss is {:.5f},target_cla_loss is {:.5f},source_cla_loss is {:.5f},cda_loss is {:.5f},mda_loss is {:.5f}'.format(
                epoch + 1, train_loss, test_loss, train_acc, test_acc, train_all_loss, target_cla_loss, source_cla_loss,
                cda_loss, mda_loss))

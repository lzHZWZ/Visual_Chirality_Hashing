from utils.tools import *
from network import *
from utils.config import *

import os
import torch
import torch.optim as optim
import time
import numpy as np

torch.multiprocessing.set_sharing_strategy('file_system')


class HashNetLoss(torch.nn.Module):
    def __init__(self, args, bit):
        super(HashNetLoss, self).__init__()
        self.U = torch.zeros(args.num_train, bit).float().to(args.device)
        self.Y = torch.zeros(args.num_train, args.n_class).float().to(args.device)

        self.scale = 1

    def forward(self, u, y, ind, args):
        u = torch.tanh(self.scale * u)

        self.U[ind, :] = u.data
        self.Y[ind, :] = y.float()

        similarity = (y @ self.Y.t() > 0).float()
        dot_product = args.alpha * u @ self.U.t()

        mask_positive = similarity.data > 0
        mask_negative = similarity.data <= 0

        exp_loss = (1 + (-dot_product.abs()).exp()).log() + dot_product.clamp(min=0) - similarity * dot_product

        S1 = mask_positive.float().sum()
        S0 = mask_negative.float().sum()
        S = S0 + S1
        exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)

        loss = exp_loss.sum() / S

        return loss


def train_val(args, bit):
    device = args.device
    train_loader, test_loader, dataset_loader, num_train, num_test, num_dataset = get_data(args)
    args.num_train = num_train
    net = AlexNet(bit).to(device)

    optimizer = get_optimizer(net,
                          args.optimizer,
                          args.learning_rate,
                          args.momentum,
                          args.weight_decay)

    criterion = HashNetLoss(args, bit)

    Best_mAP = 0

    for epoch in range(args.epoch):
        criterion.scale = (epoch // args.step_continuation + 1) ** 0.5

        current_time = time.strftime('%H:%M:%S', time.localtime(time.time()))

        print("%s[%2d/%2d][%s] bit:%d, dataset:%s, scale:%.3f, training...." % (
            args.info, epoch + 1, args.epoch, current_time, bit, args.dataset, criterion.scale), end="")

        net.train()

        train_loss = 0
        for image, label, ind in train_loader:

            image = image.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            u = net(image)

            loss = criterion(u, label.float(), ind, args)
            train_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_loss = train_loss / len(train_loader)

        print("\b\b\b\b\b\b\b loss:%.3f" % (train_loss))

        if (epoch + 1) % args.test_map == 0:
            tst_binary, tst_label = compute_result(test_loader, net, device=device)

            trn_binary, trn_label = compute_result(dataset_loader, net, device=device)

            mAP = CalcTopMap(trn_binary.numpy(), tst_binary.numpy(), trn_label.numpy(), tst_label.numpy(),
                             args.topK)

            if mAP > Best_mAP:
                Best_mAP = mAP

                if "save_path" in args:
                    if not os.path.exists(args.save_path):
                        os.makedirs(args.save_path)
                    np.save(os.path.join(args.save_path, args.dataset + str(mAP) + "-" + "trn_binary.npy"),
                            trn_binary.numpy())
            print("%s epoch:%d, bit:%d, dataset:%s, MAP:%.3f, Best MAP: %.3f" % (
                args.info, epoch + 1, bit, args.dataset, mAP, Best_mAP))
            print(args)


if __name__ == "__main__":
    args,_ = get_config()
    print(args)
    train_val(args, args.bit_size)

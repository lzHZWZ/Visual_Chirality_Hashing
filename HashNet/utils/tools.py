import numpy as np
import torch.utils.data as util_data
from torchvision import transforms
import torch
from PIL import Image
from tqdm import tqdm
import utils.config
import torchvision.datasets as dsets


def config_dataset(args):
    if "cifar" in  args.dataset:
        args.topK = -1
        args.n_class = 10
    elif args.dataset in ["nuswide_21", "nuswide_21_m"]:
        args.topK = 5000
        args.n_class = 21
    elif args.dataset == "nuswide_81_m":
        args.topK = 5000
        args.n_calss = 81
    elif args.dataset == "coco":
        args.topK = 5000
        args.n_calss = 80
    elif args.dataset == "imagenet":
        args.topK = 1000
        args.n_class = 100
    elif args.dataset == "mirflickr":
        args.topK = -1
        args.n_class = 38
    elif args.dataset == "voc2012":
        args.topK = -1
        args.n_class = 20
    elif args.dataset == "voc2007":
        args.topK = -1
        args.n_class = 20

    if args.dataset == "nuswide_21":
        args.data_path = "/dataset/NUS-WIDE/"
    if args.dataset in ["nuswide_21_m", "nuswide_81_m"]:
        args.data_path = "/dataset/nus_wide_m/"
    if args.dataset == "coco":
        args.data_path = "/dataset/COCO_2014/"
    if args.dataset == "voc2012":
        args.data_path = "/dataset/"
    args.data = {
        "train_set": {"list_path": "./data/" + args.dataset + "/train.txt", "batch_size": args.batch_size},
        "database": {"list_path": "./data/" + args.dataset + "/database.txt", "batch_size": args.batch_size},
        "test": {"list_path": "./data/" + args.dataset + "/test.txt", "batch_size": args.batch_size}}
    return args


draw_range = [1, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500,
              9000, 9500, 10000]

def pr_curve(rF, qF, rL, qL, draw_range=draw_range):
    n_query = qF.shape[0]
    Gnd = (np.dot(qL, rL.transpose()) > 0).astype(np.float32)
    Rank = np.argsort(CalcHammingDist(qF, rF))
    P, R = [], []
    for k in tqdm(draw_range):
        p = np.zeros(n_query)
        r = np.zeros(n_query)
        for it in range(n_query):
            gnd = Gnd[it]
            gnd_all = np.sum(gnd)
            if gnd_all == 0:
                continue
            asc_id = Rank[it][:k]
            gnd = gnd[asc_id]
            gnd_r = np.sum(gnd)
            p[it] = gnd_r / k
            r[it] = gnd_r / gnd_all
        P.append(np.mean(p))
        R.append(np.mean(r))
    return P, R


class ImageList(object):

    def __init__(self, data_path, image_list, transform):
        self.imgs = [(data_path + val.split()[0], np.array([int(la) for la in val.split()[1:]])) for val in image_list]
        self.transform = transform

    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, target, index

    def __len__(self):
        return len(self.imgs)


def image_transform(resize_size, rotation_size):
    step = [transforms.RandomRotation(rotation_size)]
    return transforms.Compose([transforms.Resize((resize_size,resize_size))]
                              + step +
                              [transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                    std=[0.229, 0.224, 0.225])
                               ])


class MyCIFAR10(dsets.CIFAR10):
    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        img = self.transform(img)
        target = np.eye(10, dtype=np.int8)[np.array(target)]
        return img, target, index


def cifar_dataset(args):
    batch_size = args.batch_size

    train_size = 500
    test_size = 100

    if args.dataset == "cifar10-2":
        train_size = 5000
        test_size = 1000

    transform = transforms.Compose([
        transforms.Resize(args.corp_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    cifar_dataset_root = '/dataset/cifar/'
    train_dataset = MyCIFAR10(root=cifar_dataset_root,
                              train=True,
                              transform=transform,
                              download=True)

    test_dataset = MyCIFAR10(root=cifar_dataset_root,
                             train=False,
                             transform=transform)

    database_dataset = MyCIFAR10(root=cifar_dataset_root,
                                 train=False,
                                 transform=transform)

    X = np.concatenate((train_dataset.data, test_dataset.data))
    L = np.concatenate((np.array(train_dataset.targets), np.array(test_dataset.targets)))

    first = True
    for label in range(10):
        index = np.where(L == label)[0]

        N = index.shape[0]
        perm = np.random.permutation(N)
        index = index[perm]

        if first:
            test_index = index[:test_size]
            train_index = index[test_size: train_size + test_size]
            database_index = index[train_size + test_size:]
        else:
            test_index = np.concatenate((test_index, index[:test_size]))
            train_index = np.concatenate((train_index, index[test_size: train_size + test_size]))
            database_index = np.concatenate((database_index, index[train_size + test_size:]))
        first = False

    if args.dataset == "cifar10":
        pass
    elif args.dataset == "cifar10-1":
        database_index = np.concatenate((train_index, database_index))
    elif args.dataset == "cifar10-2":
        database_index = train_index

    train_dataset.data = X[train_index]
    train_dataset.targets = L[train_index]
    test_dataset.data = X[test_index]
    test_dataset.targets = L[test_index]
    database_dataset.data = X[database_index]
    database_dataset.targets = L[database_index]

    print("train_dataset", train_dataset.data.shape[0])
    print("test_dataset", test_dataset.data.shape[0])
    print("database_dataset", database_dataset.data.shape[0])

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               num_workers=4)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=4)

    database_loader = torch.utils.data.DataLoader(dataset=database_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=4)

    return train_loader, test_loader, database_loader, \
           train_index.shape[0], test_index.shape[0], database_index.shape[0]


def get_data(args):
    if "cifar" in args.dataset:
        return cifar_dataset(args)

    dsets = {}
    dset_loaders = {}
    data_config = {
        "train_set": {"list_path": "./data/" + args.dataset + "/train.txt", "batch_size": args.batch_size},
        "database": {"list_path": "./data/" + args.dataset + "/database.txt", "batch_size": args.batch_size},
        "test": {"list_path": "./data/" + args.dataset + "/test.txt", "batch_size": args.batch_size}}
    for data_set in ["train_set", "test", "database"]:
        dsets[data_set] = ImageList(args.data_path,
                                    open(data_config[data_set]["list_path"]).readlines(),
                                    transform=image_transform(args.resize_size, args.rotation_size))
        print(data_set, len(dsets[data_set]))
        dset_loaders[data_set] = util_data.DataLoader(dsets[data_set],
                                                      batch_size=data_config[data_set]["batch_size"],
                                                      shuffle=True, num_workers=4)

    return dset_loaders["train_set"], dset_loaders["test"], dset_loaders["database"], \
           len(dsets["train_set"]), len(dsets["test"]), len(dsets["database"])


def get_optimizer(model,
                  optim,
                  learning_rate,
                  momentum,
                  weight_decay):
    if optim == 'sgd':
        optim_module = torch.optim.SGD
        optim_param = {"lr" : learning_rate,
                       "momentum": momentum}
        if weight_decay != None:
            optim_param["weight_decay"] = weight_decay
    elif optim == "RMSprop":
        optim_module = torch.optim.RMSprop
        optim_param = {"lr": learning_rate,
                       "weight_decay": weight_decay}
    else:
        print("Not supported")

    optimizer = optim_module(
                    filter(lambda x : x.requires_grad, model.parameters()),
                    **optim_param
                )
    return optimizer


def compute_result(dataloader, net, device):
    bs, clses = [], []
    net.eval()
    for img, cls, _ in tqdm(dataloader):
        clses.append(cls)
        bs.append((net(img.to(device))).data.cpu())
    return torch.cat(bs).sign(), torch.cat(clses)


def CalcHammingDist(B1, B2):
    q = B2.shape[1]
    distH = 0.5 * (q - np.dot(B1, B2.transpose()))
    return distH


def CalcTopMap(rB, qB, retrievalL, queryL, topk):
    num_query = queryL.shape[0]
    topkmap = 0
    for iter in tqdm(range(num_query)):
        gnd = (np.dot(queryL[iter, :], retrievalL.transpose()) > 0).astype(np.float32)
        hamm = CalcHammingDist(qB[iter, :], rB)
        ind = np.argsort(hamm)
        gnd = gnd[ind]

        tgnd = gnd[0:topk]
        tsum = np.sum(tgnd).astype(int)
        if tsum == 0:
            continue
        count = np.linspace(1, tsum, tsum)

        tindex = np.asarray(np.where(tgnd == 1)) + 1.0
        topkmap_ = np.mean(count / (tindex))
        topkmap = topkmap + topkmap_
    topkmap = topkmap / num_query
    return topkmap

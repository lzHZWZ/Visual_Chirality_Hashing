import torch
import time
from model_factory import get_resnet_model
import os, copy, sys
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import get_config
from torchvision import models
from datasets_factory import VOCDataset
from datasets_factory import get_transform
from tools import get_optimizer, get_scheduler,get_log_name,get_dir_name

args, _ = get_config()

log_name = get_log_name(args)
log_dir = get_dir_name(args.out_dir,
                        log_name)

if os.path.exists(log_dir):
    print(f"Log dir {log_dir} already exists. It will be overwritten.")
else:
    os.makedirs(log_dir)


model = get_resnet_model(pretrained=False,num_classes=2).to("cuda")

criterion = torch.nn.CrossEntropyLoss()

optimizer = get_optimizer(model,
                          args.optimizer,
                          args.learning_rate,
                          args.momentum,
                          args.weight_decay,
                          amsgrad=args.amsgrad)
scheduler = get_scheduler(optimizer, args.decay_step)

model_save_path = os.path.join(log_dir, "model.pt")
result_save_path = os.path.join(log_dir, "result.txt")

data_transform = get_transform(args.need_size)

train_dataset = VOCDataset(data_dir='./mirflickr/train', transform=data_transform)
valid_dataset = VOCDataset(data_dir='./mirflickr/valid', transform=data_transform)

train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, num_workers=args.num_workers)

best_val_acc = 0.0
since = epoch_time_stamp = time.time()

for epoch in range(args.epoch):
    print('Epoch {}/{}'.format(epoch, args.epoch))
    print('-' * 10)

    for phase in ['train', 'val']:
        if phase == "train":
            model.train()
            train_dataset = train_loader.dataset
        else:
            model.eval()

        running_loss = 0.0
        running_corrects = 0
        curr = 0.
        if phase == "train":
            pbar = tqdm(train_loader, ncols=120)
        else:
            pbar = tqdm(valid_loader, ncols=120)

        for batch, data in enumerate(pbar):
            curr += data[0].size(0)
            if batch >= 1:
                pbar.set_postfix(loss=running_loss / curr,
                                 acc=float(running_corrects) / curr,
                                 epoch=epoch,
                                 phase=phase)
            inputs, labels = data
            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / curr
        epoch_acc = running_corrects.double() / curr
        print('{} Loss: {:.6f} Acc: {:.6f}'.format(phase, epoch_loss, epoch_acc))
        if phase == 'train':
            scheduler.step()
        elif phase == 'val':
            epoch_use_time = time.time() - epoch_time_stamp
            epoch_time_stamp = time.time()
            print('Epoch {:d} complete in {:.0f}m {:.0f}s'.format(epoch, epoch_use_time // 60, epoch_use_time % 60))
            if epoch_acc > best_val_acc:
                best_val_acc = epoch_acc
                best_model_wts = copy.deepcopy(model).cpu()
                print(f"Best model saved to {model_save_path}")
                torch.save(best_model_wts, model_save_path)
                logging_str = 'Best val Acc thus far: {:4f} at epoch {:4d}'.format(best_val_acc, epoch)
                print(logging_str)
                with open(result_save_path, "a+") as file:
                    file.write(logging_str + "\n")
    print()
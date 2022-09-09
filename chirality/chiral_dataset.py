import os
import torch
from PIL import Image
import torchvision.transforms as transforms


def get_filelist(path):
    data_info = list()
    for home, dirs, files in os.walk(path):
        for filename in files:
            file_name = os.path.join(home, filename)
            label = 1
            data_info.append((file_name,label))
    return data_info

data_path = 'Your_Path/voc/voc2007/VOCdevkit/VOC2007/JPEGImages'

model_path = 'Your_Path/chirality_voc/result/voc_model'
model = torch.load(model_path + '/model.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

model.eval()

image_info = get_filelist(data_path)
for i in range(len(image_info)):
    
    image = Image.open(image_info[i][0]).convert('RGB')
    image_flip = image.transpose(Image.FLIP_LEFT_RIGHT)
    transform = transforms.Compose([transforms.Resize((448,448)),
                                   transforms.ToTensor()])
    img = transform(image)
    img_flip = transform(image_flip)
    img = img.unsqueeze(0).to(device)
    img_flip = img_flip.unsqueeze(0).to(device)
    
    with torch.no_grad():
        pre = model(img)
    _, predicted = torch.max(pre, 1)

    with torch.no_grad():
        pre_flip = model(img_flip)
    _, predicted_flip = torch.max(pre_flip,1)

    if predicted.item() == 1 and predicted_flip.item() == 0:
        image.save(data_path + '/chiral/' + str((image_info[i][0].split('/'))[-1]),quality=95)
    elif predicted.item() == 0 and predicted_flip.item() == 1:
        image.save(data_path + '/achiral/' + str((image_info[i][0].split('/'))[-1]), quality=95)
    else:
        image.save(data_path + '/medium/' + str((image_info[i][0].split('/'))[-1]), quality=95)

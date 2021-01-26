from models.net import get_net
import torch
import torchvision.transforms as T
from torch.nn import functional as F
from PIL import Image
# import ttach as tta

class CFG:
    model_name = 'efficientnet-b4'
    img_size = 512
    scheduler = 'CosineAnnealingWarmRestarts'
    T_max = 10
    T_0 = 10
    lr = 1e-4
    min_lr = 1e-6
    batch_size = 32
    weight_decay = 1e-6
    seed = 42
    num_classes = 5
    num_epochs = 60
    n_fold = 5
    NUM_FOLDS_TO_RUN = [1,2,3,4,5]
    PIXEL_MEAN = [0.485, 0.456, 0.406]
    PIXEL_STD = [0.229, 0.224, 0.225]
    smoothing = 0.2
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(weight_path, imgs_path):
    outputs = None
    model = get_net(CFG.model_name, 5, 'kaggle')
    state_dict = torch.load(weight_path)
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()
    
    transform = T.Compose([
        T.Resize([CFG.img_size, CFG.img_size]),
        T.ToTensor(),
        T.Normalize(mean=CFG.PIXEL_MEAN, std=CFG.PIXEL_STD)
    ])

    for img_path in imgs_path:
        img = Image.open(img_path).convert('RGB')
        x = transform(img)
        x = torch.stack([x, x.flip(-1), x.flip(-2), x.flip(-1, -2),
                             x.transpose(-1, -2), x.transpose(-1, -2).flip(-1),
                             x.transpose(-1, -2).flip(-2), x.transpose(-1, -2).flip(-1, -2)], 0)
        img = x.view(-1, 3, CFG.img_size, CFG.img_size)
#         img = img.unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output = model(img)
            if outputs is None:
                outputs = output
            else:
                outputs = torch.cat((outputs, output), dim=0)
    return outputs



if __name__=='__main__':

    import os
    from os.path import join
    from tqdm import tqdm
    import pandas as pd

    weights = ['../input/taloydiy112/weights/Fold2_0.8835705045278137_epoch3.bin', '../input/taloydiy112/weights/Fold2_0.884679356865644_epoch6.bin', '../input/taloydiy112/weights/Fold2_0.8926261319534282_epoch7.bin', '../input/taloydiy112/weights/Fold2_0.8944742191831454_epoch9.bin']
    root = '../input/cassava-leaf-disease-classification'
    path_dir = join(root, 'test_images')
    imgs_name = os.listdir(path_dir)
    imgs_path = [os.path.join(path_dir, e) for e in imgs_name]
    pred = torch.zeros(size=[len(imgs_name), 5])
    for weight in weights:
        output = load_model(weight, imgs_path)
        pred += output.cpu()/4
    pred = F.softmax(pred, dim=1).cpu().numpy()
    pred = pred.argmax(1)
    sub = pd.DataFrame({'image_id': imgs_name, 'label': pred})
    # print(sub)
    sub.to_csv("submission.csv", index=False)









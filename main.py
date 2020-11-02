from helpers import make_data, score_iou
import numpy as np
import torch
from tqdm import tqdm
from network import Net
from helpers import unnormalize
from torchsummary import summary
import pdb

def eval():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    np.random.seed(seed=46)

    model = Net()
    path = 'model.pth.tar'

    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    ious = []
    for _ in tqdm(range(1000)):
        img, label = make_data()
        img = torch.from_numpy(np.asarray(img, dtype=np.float32))
        img = torch.unsqueeze(img, 0)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)

        pred = model.predict(img)
        ious.append(score_iou(label, pred))

    ious = np.asarray(ious, dtype="float")
    ious = ious[~np.isnan(ious)]  # remove true negatives
    print((ious > 0.7).mean())

if __name__ == "__main__":
    eval()

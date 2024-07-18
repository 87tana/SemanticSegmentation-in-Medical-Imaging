import torch
import numpy as np
import cv2
from tqdm import tqdm
from src.utils.metrics import accuracy, precision_score_, recall_score_, dice_coef, iou


def evaluate(model, weights, device, dataset):

  checkpoint = torch.load(weights)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.eval()
  sigmoid = torch.nn.Sigmoid()  # If the model does not have embedded sigmoid

  thresh=0.2   #convert prediction from probability into a category

  iou_val = []
  acc_val = []
  dice_val = []
  prec_val = []
  rec_val = []

  for idx in tqdm(range(dataset.__len__())):

    imge, mask, image_orig, mask_orig  = dataset.__getitem__(idx)

    mask = mask.squeeze(0).numpy()

    img = imge.unsqueeze(0).to(device)
    with torch.no_grad():
      pred = model(img)
    pred = pred.squeeze(0)
    pred = pred.squeeze(0)
    pred = sigmoid(pred).cpu().numpy()

    if mask_orig.shape[0] != pred.shape[0]:
      pred = cv2.resize(pred, (mask_orig.shape[0],mask_orig.shape[1]),interpolation=cv2.INTER_LINEAR)

    pred[pred>thresh]= 1
    pred[pred<=thresh]= 0

    iou_val.append(iou(mask_orig, pred))
    acc_val.append(accuracy(mask_orig, pred))
    dice_val.append(dice_coef(mask_orig, pred))
    prec_val.append(precision_score_(mask_orig, pred))
    rec_val.append(recall_score_(mask_orig, pred))

  iou_val = np.mean(iou_val)
  acc_val = np.mean(acc_val)
  dice_val = np.mean(dice_val)
  prec_val = np.mean(prec_val)
  rec_val = np.mean(rec_val)

  return iou_val, acc_val, dice_val, prec_val, rec_val

  
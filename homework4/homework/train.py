import torch
import numpy as np
import torch.nn.functional as F
from .models import Detector, save_model
from .utils import load_detection_data,DetectionSuperTuxDataset
from . import dense_transforms
import torch.utils.tensorboard as tb
import os
import random
dir = os.path.dirname(os.path.abspath("__file__"))
dataset_path2 = os.path.join(dir,'dense_data','train')
dataset_path3 = os.path.join(dir,'dense_data','valid')
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.set_printoptions(profile="full")

def seed_torch(seed=1029):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.enabled = False
  torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
  torch.backends.cudnn.benchmark = False
  torch.backends.cudnn.deterministic = True


def point_in_box(pred, lbl):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return (x0 <= px) & (px < x1) & (y0 <= py) & (py < y1)

                                                                                                                                                                                                                                                                                                       
def point_close(pred, lbl, d=5):
    px, py = pred[:, None, 0], pred[:, None, 1]
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    return ((x0 + x1 - 1) / 2 - px) ** 2 + ((y0 + y1 - 1) / 2 - py) ** 2 < d ** 2


def box_iou(pred, lbl, t=0.5):
    px, py, pw2, ph2 = pred[:, None, 0], pred[:, None, 1], pred[:, None, 2], pred[:, None, 3]
    px0, px1, py0, py1 = px - pw2, px + pw2, py - ph2, py + ph2
    x0, y0, x1, y1 = lbl[None, :, 0], lbl[None, :, 1], lbl[None, :, 2], lbl[None, :, 3]
    iou = (abs(torch.min(px1, x1) - torch.max(px0, x0)) * abs(torch.min(py1, y1) - torch.max(py0, y0))) / \
          (abs(torch.max(px1, x1) - torch.min(px0, x0)) * abs(torch.max(py1, y1) - torch.min(py0, y0)))
    return iou > t


def _one_hot(x, n):
    return (x.view(-1, 1) == torch.arange(n, dtype=x.dtype, device=x.device)).int()
    
class ConfusionMatrix(object):
    def _make(self, preds, labels):
        label_range = torch.arange(self.size, device=preds.device)[None, :]
        preds_one_hot, labels_one_hot = _one_hot(preds, self.size), _one_hot(labels, self.size)
        return (labels_one_hot[:, :, None] * preds_one_hot[:, None, :]).sum(dim=0).detach()

    def __init__(self, size=5):
        """
        This class builds and updates a confusion matrix.
        :param size: the number of classes to consider
        """
        self.matrix = torch.zeros(size, size)
        self.size = size

    def add(self, preds, labels):
        """
        Updates the confusion matrix using predictions `preds` (e.g. logit.argmax(1)) and ground truth `labels`
        """
        self.matrix = self.matrix.to(preds.device)
        self.matrix += self._make(preds, labels).float()

    @property
    def class_iou(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(0) + self.matrix.sum(1) - true_pos + 1e-5)

    @property
    def iou(self):
        return self.class_iou.mean()

    @property
    def global_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos.sum() / (self.matrix.sum() + 1e-5)

    @property
    def class_accuracy(self):
        true_pos = self.matrix.diagonal()
        return true_pos / (self.matrix.sum(1) + 1e-5)

    @property
    def average_accuracy(self):
        return self.class_accuracy.mean()

    @property
    def per_class(self):
        return self.matrix / (self.matrix.sum(1, keepdims=True) + 1e-5)


class PR:
    def __init__(self, min_size=20, is_close=point_in_box):
        self.min_size = min_size
        self.total_det = 0
        self.det = []
        self.is_close = is_close

    def add(self, d, lbl):
        lbl = torch.as_tensor(lbl.astype(float), dtype=torch.float32).view(-1, 4)
        d = torch.as_tensor(d, dtype=torch.float32).view(-1, 5)
        all_pair_is_close = self.is_close(d[:, 1:], lbl)

        # Get the box size and filter out small objects
        sz = abs(lbl[:, 2]-lbl[:, 0]) * abs(lbl[:, 3]-lbl[:, 1])

        # If we have detections find all true positives and count of the rest as false positives
        if len(d):
            detection_used = torch.zeros(len(d))
            # For all large objects
            for i in range(len(lbl)):
                if sz[i] >= self.min_size:
                    # Find a true positive
                    s, j = (d[:, 0] - 1e10 * detection_used - 1e10 * ~all_pair_is_close[:, i]).max(dim=0)
                    if not detection_used[j] and all_pair_is_close[j, i]:
                        detection_used[j] = 1
                        self.det.append((float(s), 1))

            # Mark any detection with a close small ground truth as used (no not count false positives)
            detection_used += all_pair_is_close[:, sz < self.min_size].any(dim=1)

            # All other detections are false positives
            for s in d[detection_used == 0, 0]:
                self.det.append((float(s), 0))

        # Total number of detections, used to count false negatives
        self.total_det += int(torch.sum(sz >= self.min_size))


    @property
    def curve(self):
        true_pos, false_pos = 0, 0
        r = []
        for t, m in sorted(self.det, reverse=True):
            if m:
                true_pos += 1
            else:
                false_pos += 1
            prec = true_pos / (true_pos + false_pos)
            recall = true_pos / self.total_det
            r.append((prec, recall))
        return r

    @property
    def average_prec(self, n_samples=11):
        import numpy as np
        pr = np.array(self.curve, np.float32)
        return np.mean([np.max(pr[pr[:, 1] >= t, 0], initial=0) for t in np.linspace(0, 1, n_samples)])
"""
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.04, gamma=2,  reduce1=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce1 = reduce1

    def forward(self, inputs, targets):
        pred = inputs.sigmoid()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = 'none')
        alpha = targets * self.alpha + (1 - targets) *(1-self.alpha)
        pt = torch.where(targets == 1,pred,1 - pred)
        F_loss = alpha*(1-pt) ** self.gamma * BCE_loss
        #pt = torch.exp(-BCE_loss)
        #F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        #if self.reduce1:
        return torch.mean(F_loss)
        #else:
         #   return F_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
"""

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.04, gamma=2,  reduce1=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduce1 = reduce1
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction = 'none')
        p_t = torch.exp(-bce_loss)
        alpha_tensor = (1 - self.alpha) + targets * (2 * self.alpha - 1)  # alpha if target = 1 and 1 - alpha if target = 0
        f_loss = alpha_tensor * (1 - p_t) ** self.gamma * bce_loss
        return f_loss.mean()

    #pred_mask = torch.sigmoid(prediction[:, 0])
        
    #mask_loss = mask * ((1-pred_mask)**gamma)* torch.log(pred_mask+1e-12)\
     #          + (1 - mask)*torch.log(1-pred_mask+1e-12)
    #mask_loss = -mask_loss.mean(0).sum()
def accuracy(outputs, labels):
    #print(outputs.shape)
    #print(labels.shape)
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
def train(args):
    sig_layer = torch.nn.Sigmoid()
    from os import path
    seed_torch()
    model = Detector()
    model = model.to(device)
    transform2 = dense_transforms.Compose([dense_transforms.ColorJitter(brightness=0.3, contrast=0.4, saturation=0.2, hue=0.1),
                                                        dense_transforms.RandomHorizontalFlip(),
                                                        dense_transforms.ToTensor()])
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    #optimizer = torch.optim.Adam(model.parameters(),lr = 1e-6)
    optimizer3 = torch.optim.Adam(model.parameters(),lr = 1e-5)
    optimizer2 = torch.optim.Adam(model.parameters(),lr = 1e-5)
    #scheduler =  torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'max',patience = 10)
    optimizer1 = torch.optim.SGD(model.parameters(),lr = 1e-6,momentum = 0.9,weight_decay = 1e-3)
    n_epochs = 10
    train_global_step = 0
    
    dataset = DetectionSuperTuxDataset(dataset_path2,
                                       transform=transform2,min_size = 0)
    weights = [0.04]
    class_weights=torch.FloatTensor(weights).cuda()
    fl1 = FocalLoss(alpha = 0.03)
    fl2 = FocalLoss(alpha = 0.01)
    fl3 = FocalLoss(alpha = 0.01)
    
  
    #learn = cnn_learner(data, models.resnet50, metrics=[accuracy]).to_fp16()
    #weight1 = torch.ones(3,96,128).to(device)
    #weight1 = weight1 - 0.96
    #criterion = torch.nn.BCEWithLogitsLoss()
 
                                                                       
    batch_size = 4
    run = 0
    for iter in range(n_epochs):
        print("iter is {}".format(iter))
        #print(optimizer.param_groups[0]['lr'])
        permutation = torch.randperm(9998)
        train_accu = []
        
        for i in range(0,len(permutation)-batch_size+1,batch_size):
            run = run+1
            model.train()
            batch = permutation[i:i+batch_size]
            t_list =[]
            t_label = []
            for j in batch:
                (img,*dets1) = dataset[j]
                data =img
                t_list.append(data)
                label,train_size = dense_transforms.detections_to_heatmap(dets1,img.shape[1:])
                t_label.append(label)
            train_data = torch.stack(t_list)
            train_label = torch.stack(t_label)
            train_data = train_data.to(device)
            train_label = train_label.to(device)

            output = model(train_data)

            
            loss1 =  fl1(output[0],train_label[0]).float()
            loss2 =  fl2(output[1],train_label[1]).float()
            loss3 =  fl3(output[2],train_label[2]).float()
            loss = loss1+loss2+loss3
            #loss = torch.mean(loss)
            loss.backward()
            optimizer1.step()
            optimizer1.zero_grad()
            #loss2.backward(retain_graph = True)
            optimizer2.step()
            optimizer2.zero_grad()
            #loss3.backward()

            optimizer3.step()
            optimizer3.zero_grad()

            train_global_step +=1
            #del(train_data)
            #del(train_label)
            #print(loss)
            #if(run%50 == 0):
                #log(train_logger,train_data,train_label,output,train_global_step)
        model.eval()
        run = run+1
        p = 0;
                
        pr_box = [PR() for _ in range(3)]
        pr_dist = [PR(is_close=point_close) for _ in range(3)]
        pr_iou = [PR(is_close=box_iou) for _ in range(3)]
        c1 = ConfusionMatrix()
        c2 = ConfusionMatrix()
        c3 = ConfusionMatrix()
        for img1, *gts in DetectionSuperTuxDataset(dataset_path2, min_size=0):
            p = p+1
            with torch.no_grad():
                tlabel,train_size = dense_transforms.detections_to_heatmap(gts,img1.shape[1:])
                detections = model.detect(img1.to(device),0,tlabel)
                #detections = model.detect(tlabel.to(device),0)
                output2 = model(img1[None,:,:,:].to(device))
                output2 = (output2 > 0.5).long()
                output2 = output2.squeeze()
                c1.add(output2[0], tlabel[0].long().to(device))
                c2.add(output2[1], tlabel[1].long().to(device))
                c3.add(output2[2], tlabel[2].long().to(device))
                accu1 = output2[0].eq(tlabel[0].to(device)).float().mean()
                accu2 = output2[1].eq(tlabel[1].to(device)).float().mean()
                accu3 = output2[2].eq(tlabel[2].to(device)).float().mean() 
                #tlabel = tlabel[None,:,:,:].to(device)
  
                for i, gt in enumerate(gts):
                    pr_box[i].add(detections[i], gt)
                    pr_dist[i].add(detections[i], gt)
                    #pr_iou[i].add(detections[i], gt)


            if(p == 300):
                break;
                    
        with torch.no_grad():
            if(len(pr_box[0].det) >0):
                ap = pr_box[0].average_prec
                print("ap is {}".format(ap))
            if(len(pr_box[1].det) >0):
                ap1 = pr_box[1].average_prec
                print("ap 2 is {}".format(ap1))
            if(len(pr_box[2].det) >0):
                ap2 = pr_box[2].average_prec
                print("ap 3 is {}".format(ap2))
            if(len(pr_dist[0].det) >0):
                dist0 = pr_dist[0].average_prec
                print("dist 1 is {}".format(dist0))
            if(len(pr_dist[1].det) >0):
                dist1 = pr_dist[1].average_prec
                print("dist 2 is {}".format(dist1))
            if(len(pr_dist[2].det) >0):
                dist2 = pr_dist[2].average_prec
                print("dist 3 is {}".format(dist2))
            #if(len(pr_iou[0].det) >0):
             #   iou1 = pr_iou[0].average_prec
              #  print("iou 1 is {}".format(iou1))
            #if(len(pr_iou[1].det) >0):
             #   iou2 = pr_iou[1].average_prec
              #  print("iou 2 is {}".format(iou2))
            #if(len(pr_iou[2].det) >0):
             #   iou3 = pr_iou[2].average_prec
              #  print("iou 3 is {}".format(iou3))
            print("c1 is {}".format(c1.global_accuracy))
            print("c2 is {}".format(c2.global_accuracy))
            print("c3 is {}".format(c3.global_accuracy))
            print("accu1 is {}".format(accu1))
            print("accu2 is {}".format(accu2))
            print("accu3 is {}".format(accu3))
            image2, *det2 = dataset[100+1];
            tlabel,train_size = dense_transforms.detections_to_heatmap(gts,img1.shape[1:])
            kart,bomb,pickup = model.detect(image2.to(device),1,tlabel.to(device))
            print("kart is")
            print(kart)
            print("bomb is")
            print(bomb)
            print("pickup is")
            print(pickup)


            
    model.eval()        
    image2, *det2 = dataset[100+1];
    tlabel,train_size = dense_transforms.detections_to_heatmap(gts,img1.shape[1:])
    kart,bomb,pickup = model.detect(image2.to(device),1,tlabel.to(device))
    print("kart is")
    print(kart)
    print("bomb is")
    print(bomb)
    print("pickup is")
    print(pickup)

    """
    Your code here, modify your HW3 code
    Hint: Use the log function below to debug and visualize your model
    """
    #raise NotImplementedError('train')
    save_model(model)


def log(logger, imgs, gt_det, det, global_step):
    """
    logger: train_logger/valid_logger
    imgs: image tensor from data loader
    gt_det: ground-truth object-center maps
    det: predicted object-center heatmaps
    global_step: iteration
    """
    logger.add_images('image', imgs[:16], global_step)
    logger.add_images('label', gt_det[:16], global_step)
    logger.add_images('pred', torch.sigmoid(det[:16]), global_step)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--log_dir')
    # Put custom arguments here

    args = parser.parse_args()
    train(args)

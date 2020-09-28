import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=0, max_det=100):
    pool = torch.nn.MaxPool2d(max_pool_ks,stride = (max_pool_ks,max_pool_ks),ceil_mode = True,return_indices = True)
    heatmap_mod = heatmap[None,None]
    m = pool(heatmap_mod)
    score = m[0]

    cx,cy = torch.floor_divide(m[1],heatmap.shape[1]), m[1]%heatmap.shape[1] 
    score = torch.squeeze(score)
    cx = torch.squeeze(cx)
    cy = torch.squeeze(cy)
    list_extracted = []
    wind_sizes = []
    for i in range(0,heatmap.size(0),max_pool_ks):
        win1_cx = i
        win2_cx = i+max_pool_ks
        iter = 0
        win1_cy = 0
        win2_cy = max_pool_ks
        for k in range(0,heatmap.size(1),max_pool_ks):
            win1_cy = k
            iter = iter +1
            wind_sizes.append((win1_cx,win1_cy,win2_cx,win2_cy))
            win2_cy = win2_cy + max_pool_ks

    index = 0        
    for i in range(0,score.size(0)):
        for j in range(0,score.size(1)):
            if(score[i][j] > min_score):
                new_matrix = heatmap[wind_sizes[index][0]:wind_sizes[index][2],wind_sizes[index][1]:wind_sizes[index][3]].reshape(1,-1)
                if(new_matrix.size(1) != 1):
                    new_matrix = new_matrix.squeeze()1
                
                count_ele = new_matrix.tolist().count(score[i][j])
                m1 = []
                m2 = []
                if(count_ele > 1) and (count_ele == max_pool_ks*max_pool_ks):
                    _,indices = torch.topk(new_matrix,count_ele)
                    m1 = torch.floor_divide(indices,max_pool_ks)
                    m2 = indices % max_pool_ks
                    for s in range(0,len(m2)):
                        list_extracted.append((score[i][j].item(),wind_sizes[index][1]+m2[s].item(),wind_sizes[index][0]+m1[s].item()))
                else:    
                    list_extracted.append((score[i][j].item(),cy[i][j].item(),cx[i][j].item()))
            index = index+1
    
    final_list = [elem for elem in list_extracted]
    for k in list_extracted:
        score1 = k[0]
        cy1 = k[1]
        cx1 = k[2]
        if(max_pool_ks == 1):
            max_pool_ks = 0
        for ele in list_extracted:
            if(cy1-max_pool_ks <= ele[1] <= cy1+max_pool_ks):
                if(cx1-max_pool_ks <= ele[2] <= cx1+max_pool_ks):
                    if(score1 < ele[0]):
                        if k in final_list:
                            final_list.remove(k)
                            break
                    elif(score1 > ele[0]):
                        if ele in final_list:
                            final_list.remove(ele)
    return(final_list[0:max_det]) 

    
class Detector(torch.nn.Module):
    def __init__(self):
        """
           Your code here.
           Setup your detection network
        """
        super().__init__()
        raise NotImplementedError('Detector.__init__')

    def forward(self, x):
        """
           Your code here.
           Implement a forward pass through the network, use forward for training,
           and detect for detection
        """
        raise NotImplementedError('Detector.forward')

    def detect(self, image):
        """
           Your code here.
           Implement object detection here.
           @image: 3 x H x W image
           @return: Three list of detections [(score, cx, cy, w/2, h/2), ...], one per class,
                    return no more than 30 detections per image per class. You only need to predict width and height
                    for extra credit. If you do not predict an object size, return w=0, h=0.
           Hint: Use extract_peak here
           Hint: Make sure to return three python lists of tuples of (float, int, int, float, float) and not a pytorch
                 scalar. Otherwise pytorch might keep a computation graph in the background and your program will run
                 out of memory.
        """
        raise NotImplementedError('Detector.detect')


def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'det.th'))


def load_model():
    from torch import load
    from os import path
    r = Detector()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'det.th'), map_location='cpu'))
    return r


if __name__ == '__main__':
    """
    Shows detections of your detector
    """
    from .utils import DetectionSuperTuxDataset
    dataset = DetectionSuperTuxDataset('dense_data/valid', min_size=0)
    import torchvision.transforms.functional as TF
    from pylab import show, subplots
    import matplotlib.patches as patches

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    fig, axs = subplots(3, 4)
    model = load_model().eval().to(device)
    for i, ax in enumerate(axs.flat):
        im, kart, bomb, pickup = dataset[i]
        ax.imshow(TF.to_pil_image(im), interpolation=None)
        for k in kart:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='r'))
        for k in bomb:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='g'))
        for k in pickup:
            ax.add_patch(
                patches.Rectangle((k[0] - 0.5, k[1] - 0.5), k[2] - k[0], k[3] - k[1], facecolor='none', edgecolor='b'))
        detections = model.detect(im.to(device))
        for c in range(3):
            for s, cx, cy, w, h in detections[c]:
                ax.add_patch(patches.Circle((cx, cy), radius=max(2 + s / 2, 0.1), color='rgb'[c]))
        ax.axis('off')
    show()

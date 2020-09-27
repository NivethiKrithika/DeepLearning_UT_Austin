import torch
import torch.nn.functional as F


def extract_peak(heatmap, max_pool_ks=7, min_score=-5, max_det=100):
    pool = torch.nn.MaxPool2d(max_pool_ks,stride = (max_pool_ks,max_pool_ks),ceil_mode = True,return_indices = True)
    #print(heatmap.shape)
    heatmap_mod = heatmap[None,None]
    #print(heatmap_mod.shape)
    m = pool(heatmap_mod)
    #print(m[0])
    score = m[0]
    #print(a.shape[0])
    cx,cy = torch.floor_divide(m[1],heatmap.shape[1]), m[1]%heatmap.shape[1] 
    score = torch.squeeze(score)
    cx = torch.squeeze(cx)
    cy = torch.squeeze(cy)
    #print(cx.shape)
#print(cx[0,0,0,0])
    list_extracted = []

#for i in range(0,m[0].size(3)):
 #       list1.append((m[0][0][0][0][i].item(),cx[0][0][0][i].item(),cy[0][0][0][i].item()))
    for i in range(0,score.size(0)):
        for j in range(0,score.size(1)):
            if(score[i][j] > min_score):
                list_extracted.append((score[i][j].item(),cy[i][j].item(),cx[i][j].item()))
    return(list_extracted[0:max_det])


    
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

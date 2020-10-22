import torch
import torch.nn.functional as F
import torchvision
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_peak(heatmap, max_pool_ks=8, min_score=0, max_det=30):
  with torch.no_grad():
    pool = torch.nn.MaxPool2d(max_pool_ks,stride = (max_pool_ks,max_pool_ks),ceil_mode = True,return_indices =True) 
    if(max_pool_ks == 1):
        max_pool_ks = 0
    max_pool_ks1 = torch.Tensor(1).to(device)
    max_pool_ks1[0] = max_pool_ks
    k = heatmap[None,None].float().to(device)
    score = pool(k)[0].to(device)
    cx,cy = torch.floor_divide(pool(k)[1],heatmap.shape[1]).to(device), (pool(k)[1]%heatmap.shape[1]).to(device)

    score_f = torch.squeeze(torch.squeeze(score).float().view(1,-1)).to(device)
    cx1_f = torch.squeeze(torch.squeeze(cx).float().view(1,-1)).to(device)
    cy1_f = torch.squeeze(torch.squeeze(cy).float().view(1,-1)).to(device)
    points = torch.ones(cx1_f.shape).to(device)
    points = score_f > min_score
    score_f = score_f[points].to(device)
    cx1_f = cx1_f[points].to(device)
    cy1_f = cy1_f[points].to(device)
    points2 = torch.ones(cx1_f.shape).to(device)
    points3 = torch.ones(cx1_f.shape).to(device)
    points5 = torch.ones(cx1_f.shape).to(device)
    points6 = torch.ones(cx1_f.shape).to(device)
    cy_minus = torch.max(cy1_f - max_pool_ks1[0].item(),torch.tensor([0.]).to(device)).long().to(device)
    cy_minus[cy_minus > heatmap.size(1)-1] = heatmap.size(1)-1
    cy_plus = torch.max(cy1_f + max_pool_ks1[0].item(),torch.tensor([0.]).to(device)).long().to(device)
    cy_plus[cy_plus > heatmap.size(1)-1] = heatmap.size(1)-1
    cx_minus = torch.max(cx1_f - max_pool_ks1[0].item(),torch.tensor([0.]).to(device)).long().to(device)
    cx_minus[cx_minus > heatmap.size(0)-1] = heatmap.size(0)-1
    cx_plus = torch.max(cx1_f + max_pool_ks1[0].item(),torch.tensor([0.]).to(device)).long().to(device)
    cx_plus[cx_plus > heatmap.size(0)-1] = heatmap.size(0)-1
    points2 = (cy_minus <= cy1_f).to(device)
    points5 = (cy1_f <= cy_plus).to(device)
    points3 = (cx_minus <= cx1_f).to(device)
    points6 = (cx1_f <= cx_plus).to(device)
    matrix = torch.Tensor().to(device)
    for cx_minuse,cx_pluse,cy_minuse,cy_pluse in zip(cx_minus,cx_plus,cy_minus,cy_plus):
        matrix = torch.cat((matrix,torch.topk(heatmap[cx_minuse:cx_pluse+1,cy_minuse:cy_pluse+1].reshape(1,-1),k = 1)[0]),dim = 0)
    matrix = matrix.squeeze()
    final_points = [matrix == score_f]
    final_cx = cx1_f[final_points].long().to(device)
    final_cy = cy1_f[final_points].long().to(device)
    final_score = score_f[final_points]
    final_points1 =[(element0,element1,element2) for element0,element1,element2 in zip(final_score.cpu().tolist(),final_cy.cpu().tolist(),final_cx.cpu().tolist())]
    return final_points1[0:max_det]

class Detector(torch.nn.Module):
    class construct_layer(torch.nn.Module):
        def __init__(self,in_channels,out_channels):
            super().__init__()
            self.concat_layers = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels,3,padding = 1,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Dropout(p = 0.1),
                                                     torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Dropout(p = 0.1))
        def forward(self,x): 
            return self.concat_layers(x)


    class up_conv(torch.nn.Module):
        def __init__(self,in_channels,out_channels):
            super().__init__()  
            self.concat_layers1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels,out_channels,3,padding = 1,stride =2,output_padding = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
            #self.concat_layers1 = torch.nn.Sequential(torch.nn.Upsample(mode='bilinear', scale_factor=2),
             #                                         torch.nn.Conv2d(in_channels, out_channels, kernel_size=1),
              #                                        torch.nn.BatchNorm2d(out_channels),
               #                                       torch.nn.ReLU())
            #self.concat_layers1 = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels,3,padding = 1,stride = 1),
             #                                        torch.nn.BatchNorm2d(out_channels),
              #                                       torch.nn.ReLU(),
               #                                      torch.nn.Conv2d(out_channels,out_channels,3,padding = 1,stride = 1),
                #                                     torch.nn.BatchNorm2d(out_channels),
                 #                                    torch.nn.ReLU(),
                  #                                   torch.nn.ConvTranspose2d(out_channels,out_channels,3,padding = 1,stride =2,output_padding = 1))
           

        def forward(self,x): 
            return self.concat_layers1(x)

        #raise NotImplementedError('FCN.__init__')
    def __init__(self):
        super().__init__()
        self.first_conv = self.construct_layer(3,64)
        self.second_conv = self.construct_layer(64,128)
        self.third_conv = self.construct_layer(128,256)
        self.first_up_conv = self.up_conv(256,128)
        self.second_up_conv = self.up_conv(256,64)
        self.third_up_conv = self.up_conv(128,3)
        self.out_conv  = torch.nn.Conv2d(3,3,kernel_size = 1)
        self.pool = torch.nn.MaxPool2d(2)
        self.sig_layer =torch.nn.Sigmoid()
        self.batch_norm = torch.nn.BatchNorm2d(3)
        self.relu = torch.nn.ReLU()
        self.transform3 = torchvision.transforms.Normalize(mean = [0.485,0.456,0.406],
                                                                       std = [0.229,0.224,0.225])

    def forward(self,x):
        mean = torch.Tensor([0.485, 0.456, 0.406]).to(x.device)
        mean_mod = mean[None,:,None,None]
        x = x - mean_mod
        #print("mean is  {}".format(x))
        std= torch.Tensor([0.229, 0.224, 0.225]).to(x.device)
        std_mod =std[None,:,None,None]
        x = x/std_mod
        first_res = self.first_conv(x)
        max_pool_first = self.pool(first_res)
        second_res =  self.second_conv(max_pool_first)
        max_pool_sec = self.pool(second_res)
        third_res =  self.third_conv(max_pool_sec)       
        max_pool_third = self.pool(third_res)
        first_up_res = self.first_up_conv(max_pool_third)
        second_up_res = self.second_up_conv(torch.cat([first_up_res,max_pool_sec],1))
        final = self.third_up_conv(torch.cat([second_up_res,max_pool_first],1))
        final_final = self.sig_layer(self.out_conv(final))
        return final_final
        
        

    def detect(self, image,to_print,label4):
        with torch.no_grad(): 
            image = self.transform3(image)
            y = image[None,:,:,:]
        
        
            first_res1 = self.first_conv(y)
            max_pool_first1 = self.pool(first_res1)
        #print("max_y shape is {}".format(max_pool_first.shape))
        
            second_res1 =  self.second_conv(max_pool_first1)
            max_pool_sec1 = self.pool(second_res1)
        #print("max_z shape is {}".format(max_pool_sec.shape))
        
            third_res1 =  self.third_conv(max_pool_sec1)       
            max_pool_third1 = self.pool(third_res1)
        #print("max_m size is {}".format(max_pool_third.shape))
        
            first_up_res1 = self.first_up_conv(max_pool_third1)
        #print(first_up_res.shape)
        
            second_up_res1 = self.second_up_conv(torch.cat([first_up_res1,max_pool_sec1],1))
        #print ("n shape is {}".format(second_up_res.shape))
        
            final1 = self.third_up_conv(torch.cat([second_up_res1,max_pool_first1],1))
            final12 =self.out_conv(final1)
            final_final1 = self.sig_layer(final12)
            final_final1 = final_final1.squeeze()
            if(to_print == 1):
            #print(final_final1)
                list_11 = extract_peak(label4[0],min_score = 0,max_det = 100)
                kart_det1 = []
                for ele4 in list_11:
                    kart_det1.append((ele4[0],ele4[1],ele4[2],0,0))
                print(kart_det1)
                bomb_det1 = []
                list_21 = extract_peak(label4[1],min_score =0,max_det = 100)
                for ele5 in list_21:
                    bomb_det1.append((ele5[0],ele5[1],ele5[2],0,0))
                print(bomb_det1)
                pickup_det1 = []
                list_31 = extract_peak(label4[2],min_score = 0,max_det = 100)
                for ele6 in list_31:
                    pickup_det1.append((ele6[0],ele6[1],ele6[2],0,0))
                print(pickup_det1)  

        
        #final_final1 = y.squeeze()
            list_1 = extract_peak(final_final1[0],min_score = 0.55,max_det = 100)
            kart_det = []
            for ele1 in list_1:
                kart_det.append((ele1[0],ele1[1],ele1[2],0,0))
            bomb_det = []
            list_2 = extract_peak(final_final1[1],min_score =0.5,max_det = 100)
            for ele2 in list_2:
                bomb_det.append((ele2[0],ele2[1],ele2[2],0,0))
            pickup_det = []
            list_3 = extract_peak(final_final1[2],min_score = 0.5,max_det = 100)
            for ele3 in list_3:
                pickup_det.append((ele3[0],ele3[1],ele3[2],0,0))
        
            return kart_det,bomb_det,pickup_det
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
        #raise NotImplementedError('Detector.detect')


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

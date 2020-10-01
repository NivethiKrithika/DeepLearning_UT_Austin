import torch
import torch.nn.functional as F
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def extract_peak(heatmap, max_pool_ks=4, min_score=0, max_det=30):
    pool = torch.nn.MaxPool2d(max_pool_ks,stride = (max_pool_ks,max_pool_ks),ceil_mode = True,return_indices = True)
    heatmap_mod = heatmap[None,None]
    heatmap_mod = heatmap_mod.to(device)
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
                    new_matrix = new_matrix.squeeze()
                
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
    class construct_layer(torch.nn.Module):
        def __init__(self,in_channels,out_channels,kernel_size,pad):
            super().__init__()
            self.concat_layers = torch.nn.Sequential(torch.nn.Conv2d(in_channels, out_channels,kernel_size,padding = pad,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU(),
                                                     torch.nn.Conv2d(out_channels,out_channels,kernel_size,padding = pad,stride = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
            #self.down_sample = torch.nn.Conv2d(in_channels,out_channels,kernel_size = 1,stride = 1)
        def forward(self,x): 
            return self.concat_layers(x)
        
    class up_conv(torch.nn.Module):
        def __init__(self,in_channels,out_channels):
            super().__init__()  
            self.concat_layers1 = torch.nn.Sequential(torch.nn.ConvTranspose2d(in_channels,out_channels,3,padding = 1,stride =2,output_padding = 1),
                                                     torch.nn.BatchNorm2d(out_channels),
                                                     torch.nn.ReLU())
           
        def forward(self,x): 
            return self.concat_layers1(x)

       #raise NotImplementedError('FCN.__init__')
    def __init__(self):
        super().__init__()
        self.first_conv = self.construct_layer(3,64,3,1)
        self.second_conv = self.construct_layer(64,128,3,1)
        self.third_conv = self.construct_layer(128,256,3,1)
        self.first_up_conv = self.up_conv(256,128)
        self.second_up_conv = self.up_conv(256,64)
        self.third_up_conv = self.up_conv(128,3)
        #self.third_up_conv = torch.nn.Sequential(torch.nn.ConvTranspose2d(128,3,3,padding = 1,stride =2,output_padding = 1),
                                                   # torch.nn.BatchNorm2d(3),
                                                    # torch.nn.Sigmoid())
        
        self.first_conv_sc = torch.nn.Sequential(torch.nn.Conv2d(3,128,7,padding = 3,stride =1),
                                                    torch.nn.BatchNorm2d(128),
                                                    torch.nn.ReLU())
        self.second_conv_sc =torch.nn.Sequential(torch.nn.Conv2d(128,128,7,padding = 3,stride =1),
                                                    torch.nn.BatchNorm2d(128),
                                                    torch.nn.ReLU())
        self.third_conv_sc = torch.nn.Sequential(torch.nn.Conv2d(128,3,7,padding = 3,stride =1),
                                                    torch.nn.BatchNorm2d(3),
                                                    torch.nn.Sigmoid())
        #self.fourth_conv_sc = torch.nn.Sequential(torch.nn.Conv2d(128,3,7,padding = 3,stride =1),
         #                                           torch.nn.BatchNorm2d(3),
          #                                          torch.nn.ReLU())
      
        
       # layers= []
       # L = [32,64,128]
       # c = 3
       # for l in L:
        #    layers.append(self.construct_layer(c,l))
         #   layers.append(torch.nn.Maxpool2d(2))
          #  c = l
        #layers.append(self.up_conv(c,64))
        
            
        self.pool = torch.nn.MaxPool2d(2)
        self.pool_reduce = torch.nn.MaxPool2d(4)
        #self.final_layers = torch.nn.Sequential(*layers)
        #self.final = torch.nn.Sequential(*layers1)
        self.out_conv = torch.nn.Conv2d(32,5,1)
        self.upsample = torch.nn.Upsample(scale_factor = 4, mode = 'bicubic')


        
    def forward(self,x):
        padding_done = 0
        padded_oh = 0
        padded_ow = 0
        if x.size(2) < 16 or x.size(3) < 16:
            padding_done = 1
            ow, oh = x.size(2),x.size(3)
            #print(oh)
            #print(ow)
            padh = 16 - oh if oh < 16 else 0
            padw = 16 - ow if ow < 16 else 0
            padded_ow = ow
            padded_oh = oh
            x = R.pad(x, (0, padh,0, padw), value =0)
        
        first_res = self.first_conv(x)
        max_pool_first = self.pool(first_res)
        #print("max_y shape is {}".format(max_pool_first.shape))
        
        second_res =  self.second_conv(max_pool_first)
        max_pool_sec = self.pool(second_res)
        #print("max_z shape is {}".format(max_pool_sec.shape))
        
        third_res =  self.third_conv(max_pool_sec)
        #print("third_res is {}".format(third_res.shape))
        max_pool_third = self.pool(third_res)
        #print("max_m size is {}".format(max_pool_third.shape))
        
        first_up_res = self.first_up_conv(max_pool_third)
        #print(first_up_res.shape)
        
        second_up_res = self.second_up_conv(torch.cat([first_up_res,max_pool_sec],1))
        #print ("n shape is {}".format(second_up_res.shape))
        
        final = self.third_up_conv(torch.cat([second_up_res,max_pool_first],1))
        #print("final is {}".format(final.shape))
        pool_sc =self.pool_reduce(final)
        #print("shape after pooling is {}".format(pool_sc.shape))
        first_sc = self.first_conv_sc(pool_sc)
        #print("shape after first sc is {}".format(first_sc.shape))
        second_sc = self.second_conv_sc(first_sc)
        #print("shape after second sc is {}".format(second_sc.shape))
        third_sc = self.third_conv_sc(second_sc)
        #print("shape after third sc is {}".format(third_sc.shape))
        final_sc= self.upsample(third_sc)
        #print("shape after upsampling is {}".format(final_sc.shape))
        final_final = final_sc * final
        #print("third shape is {}".format(third_up_res.shape))
        #final = self.out_conv(third_up_res)
        if padding_done == 1:
            final_final = final_final[:,:,0:padded_ow,0:padded_oh]
        return final_final


    def detect(self, image):
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
        pool_sc1 =self.pool_reduce(final1)
        #print("shape after pooling is {}".format(pool_sc.shape))
        first_sc1 = self.first_conv_sc(pool_sc1)
        #print("shape after first sc is {}".format(first_sc.shape))
        second_sc1 = self.second_conv_sc(first_sc1)
        #print("shape after second sc is {}".format(second_sc.shape))
        third_sc1 = self.third_conv_sc(second_sc1)
        #print("shape after third sc is {}".format(third_sc.shape))
        final_sc1= self.upsample(third_sc1)
        #print("shape after upsampling is {}".format(final_sc.shape))
        final_final1 = final_sc1 * final1
        final_final1 = final_final1.squeeze()
        print(final_final1)
        list_1 = extract_peak(final_final1[0])
        kart_det = []
        for ele in list_1:
            kart_det.append((ele[0],ele[1],ele[2],0,0))
        bomb_det = []
        list_2 = extract_peak(final_final1[1])
        for ele in list_2:
            bomb_det.append((ele[0],ele[1],ele[2],0,0))
        pickup_det = []
        list_3 = extract_peak(final_final1[2])
        for ele in list_3:
            pickup_det.append((ele[0],ele[1],ele[2],0,0))
        
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

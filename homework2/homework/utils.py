from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, dataset_path):
        list1 = []
        tr_image = torchvision.transforms.ToTensor()
        self.transform =  torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()])          
        to_image=transforms.ToPILImage()
        i = 1;
        #print(dataset_path1)
        for filename in sorted(os.listdir(dataset_path)):
            file_path = os.path.abspath(os.path.join(dataset_path, filename))
            _, file_extension = os.path.splitext(file_path)
            if(file_extension == ".jpg"):
                image = self.transform(Image.open(file_path)).float()
                list1.append(image)

                
        self.data = torch.stack(list1)
        list1.clear()
        #print("loaded succesfully")
        #print(self.data.shape)

        csv_path = os.path.join(dataset_path,"labels.csv")
        with open(csv_path,'r') as dest_f:
            data_iter = csv.reader(dest_f,
                           delimiter = ',')
            label = [data1 for data1 in data_iter]
        
        label = np.array(label)
        label = (label[1:,1])
        mod_label = [LABEL_NAMES.index(label_ind) for label_ind in label]
        self.label = torch.from_numpy(np.array(mod_label))
        self.nsamples = self.data.shape[0]

    def __len__(self):

        return self.nsamples
        raise NotImplementedError('SuperTuxDataset.__len__')

    def __getitem__(self, idx):
        return self.data[idx],self.label[idx].item();
        """
        Your code here
        return a tuple: img, label
        """
        raise NotImplementedError('SuperTuxDataset.__getitem__')


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()

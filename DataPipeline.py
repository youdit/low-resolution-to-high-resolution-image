from os import listdir
from os.path import join
from PIL import Image

import torch.utils.data.dataset as dt 
import torchvision.transforms as ttf 

def is_image(filename):
    """ 
        used to check the image given to us is in a image format
    """
    image = [filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']]
    for x in image:
        if x:
            return True
    else:
        return False

def valid_crop_size(crop_size, upscale_factor):
    """
        crop size of the image to create the data augmentation 
    """
    val = crop_size - (crop_size%upscale_factor)
    return val

def train_hr_transform(crop_size):
    """
        RandomCrop : it will crop the image at random location (crop will be according to crop_size), it tke a PIL image [H,W,C]
        ToTansor : it convert the PIL image [H,W,C] to torch image [C,H,W]
        Compose : it act like torch.nn.Sequential, it contain the list of operation to perform on image 
        train_hr_transform: it is used to create the crop image of High Resolution Image for Data Augmentation
    """
    return ttf.Compose[ttf.RandomCrop(crop_size), ttf.ToTensor()]

def train_lr_transform(crop_size, upscale_factor=4):
    """
    train_lr_transform : this function is created to convert the higher resolution image to lower resolution
    ToPILImage: convert the pytorch image tensor [C,H,W] ---> PIL Image [H,W,C]
    Resize: it is usto lower the resotuion to required factor by using upscale_factor
    ToTensor: Conert PIL Image [C,H,W] --> pytorch Image [C,H,W]
    """
    return ttf.Compose([ttf.ToPILImage(), ttf.Resize(crop_size//upscale_factor, interpolation=Image.BICUBIC), ttf.ToTensor()])

def display_transform():
    return ttf.Compose([ttf.ToPILImage(),ttf.Resize(400),ttf.CenterCrop(400),ttf.ToTensor()])


class Train_Dataset_Folder(dt.Dataset):
    def __init__(self, dataset_dir, crop_size, upscale_factor=4):
        super(Train_Dataset_Folder,self).__init__()
        """ 
        INPUT: dataset_dir: location of the folder in which the training images are present 
            filename: it is the list of the path of the image to be used for the training these are HR images
            hr_transform: to create the crops of higher resolution image for data augmentation
            lr_transform: to convert HR Images to LR Images 
        """
        self.filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image(x)]
        crop_size = valid_crop_size(crop_size,upscale_factor)
        self.hr_transform = train_hr_transform(crop_size)
        self.lr_transform = train_lr_transform(crop_size, upscale_factor)
    
    def __getitem__(self,index):
        """
            it is a getter method to compile lr_image with hr_images to for the dataset or indexing methods
            if we index the data it return two images lr_image and hr_image
        """
        hr_image = self.train_hr_transform(Image.open(self.filenames[index]))
        lr_image = self.lr_transform(hr_image)
        return lr_image, hr_image
    
    def __len__(self):
        # length is used because Dataset should have a indexing and length method
        return len(self.filenames)

class Valid_dataset_folder(dt.Dataset):
    def __init__(self, dataset_dir, upscale_factor=4):
        super(Valid_dataset_folder,self).__init__()
        """ 
            in validation dataset we pass the path of the folder which contain the validation Image in it
            self.fienames: it contain a list of all the path of the validation image
        """
        self.upscale_factor = upscale_factor
        self.filenames = [join(dataset_dir, x) for x in listdir(dataset_dir) if is_image(x)]
    
    def __getitem__(self, index):
        """
            in the getter method we perform the folowing task: open the image as PIL -> find the valid cropsize for data augmentation ->  find lr_scale & hr_scale -> centercrop hr image -> 
            -> lower the resolution -> higher resolution (by resizing)
        Return :
               lower resolution image, higher resolution restored image by resizing, higher resolution image. in pytorch image format[C,H,W]
        """
        hr = Image.open(self.filenames[index])
        w, h = hr.size 
        crop_size = valid_crop_size(min(w,h),self.upscale_factor)
        lr_scale = ttf.Resize(crop_size//self.upscale_factor, interpolation=Image.BICUBIC)
        hr_scale = ttf.Resize(crop_size, interpolation=Image.BICUBIC)
        hr = ttf.CenterCrop(crop_size)(hr)
        lr = lr_scale(hr)
        hr_restore = hr_scale(lr)
        return ttf.ToTensor()(lr), ttf.ToTensor()(hr_restore), ttf.ToTensor()(hr)
    
    def __len__(self):
        return len(self.filenames)


class Test_dataset_folder(dt.Dataset):
    def __init__(self, dataset_dir, upscale_factor=4):
        super(Train_Dataset_Folder,self).__init__()
        self.lr_path = dataset_dir + 'test' + '/data/'
        self.hr_path = dataset_dir + 'test' + '/target/'
        self.upscale_factor = upscale_factor
        self.lr_files = [join(self.lr_path, x) for x in listdir(self.lr_path) if is_image(x)]
        self.hr_files = [join(self.hr_path, x) for x in listdir(self.hr_path) if is_image(x)]
    
    def __getitem__(self,index):
        image_name = self.lr_file[index].split('/'[-1])
        lr = Image.open(self.lr_file[index])
        w,h = lr.size
        hr = Image.open(self.hr_file[index])
        hr_scale = ttf.Resize((self.upscale_factor*h, self.upscale_factor*w),interpolation=Image.BICUBIC)
        hr_restore = hr_scale(lr)
        return image_name, ttf.ToTensor()(lr), ttf.ToTensor()(hr_restore), ttf.ToTensor()(hr)
    
    def __len__(self):
        return len(self.lr_file)
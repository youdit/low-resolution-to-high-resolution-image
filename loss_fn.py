import torch
from torchvision.models.vgg import vgg16



class Perception_loss(torch.nn.Module):
    def __init__(self):
        """
            we will input VGG16 network (pretrain=true) only conv layers only
            so we get feature layer of last layer to comapre 
        """
        super(Perception_loss,self).__init__()
        vgg = vgg16(pretrained=True)
        loss_network = torch.nn.Sequential(*list(vgg.features)[:31]).eval()
        for param in loss_network.parameters():
            param.requires_grad = False
        self.loss_network = loss_network
        self.mse = torch.nn.MSELoss()
    
    def forward(self, out_image, target_image):
        """
         we compare the vgg layer MeanSquareError of out output image and input image
        """
        perception_loss = self.mse(self.loss_network(out_image), self.loss_network(target_image))
        return perception_loss*0.006



class Generator_loss(torch.nn.Module):
    def __init__(self):
        super(Generator_loss,self).__init__()
        self.loss_network = Perception_loss()
        self.mse= torch.nn.MSELoss()
    
    def forward(self, out_label, out_image, target_image):
        adversarial_loss = torch.mean(1- out_label)*0.001
        perception_loss = self.loss_network(out_image, target_image)
        image_loss = self.mse(out_image, target_image)

        return adversarial_loss + perception_loss + image_loss


if __name__=='__main__':
    loss = Generator_loss()
    print(loss)
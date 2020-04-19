import torch
from helper import normalize, normalize0, denormalize
import math

class Res_block(torch.nn.Module):
    def __init__(self, filters, momentum=0.8):
        """
        Res_block structure:
             
            x_in -> [con2D] -> [Batchnoramlization] -> [conv2D] -> [batchnormalization] -> [PRelU] ->  x_out + x_in -> return x
               |                                                                                       |
                ->   ----->                  ----->        ------>           ----->      ---->      -->
            only condition is number of filters is same as input & output,  kernal size=3 and padding same by formula padding = (kernal_size - 1)/2
        """
        super(Res_block,self).__init__()
        self.conv1 = torch.nn.Conv2d(filters, filters, 3, padding=1)
        self.batch1 = torch.nn.BatchNorm2d(filters, momentum=momentum)
        self.conv2 = torch.nn.Conv2d(filters, filters, 3, padding=1)
        self.batch2 = torch.nn.BatchNorm2d(filters, momentum=momentum)
        self.act = torch.nn.PReLU()

    def forward(self, x_in):
        x = self.conv1(x_in)
        x = self.batch1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.batch2(x)
        return x + x_in

class Upsample(torch.nn.Module):
    def __init__(self, filters, up_scale=2):
        """ 
            [conv2D] --> [PixelShuffle (depth_to_space)] --> [PReLU]
        """
        super(Upsample, self).__init__()
        self.conv1 = torch.nn.Conv2d(filters, filters*(up_scale**2), 3, padding=1) #upscale factor = 2
        self.shuffle = torch.nn.PixelShuffle(up_scale)
        self.act = torch.nn.PReLU()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.act(self.shuffle(x))
        return x

class Generator(torch.nn.Module):  #upscale_factor=4 (integer)
    def __init__(self, filters=64, num_res=16, upscale_factor=4):
        """
            [normalize(0,1)] --> [conv2D] --> [PReLU] --> [res_block]*(num_res) --> [conv2D] --> [Batchnormalize] --> [Upsample]*(num_upsample_block) --> [Conv2D] --> [tanh] --> [denormalize]
                                           |                                                                       |
                                            ---->   ----->   ----->  ----->   ------>    ------>  ------->  ------->
        """
        num_upsample_block = int(math.log(upscale_factor,2))
        super(Generator, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, filters, 9, padding=4)
        self.conv2 = torch.nn.Conv2d(filters, filters, 3, padding=1)
        self.prelu = torch.nn.PReLU()
        self.batch1 = torch.nn.BatchNorm2d(filters)
        self.conv3 = torch.nn.Conv2d(filters, 3, 9, padding=4)
        self.tanh = torch.nn.Tanh()
        self.residual_part = torch.nn.ModuleList([Res_block(filters) for i in range(num_res)])
        self.upsample = torch.nn.ModuleList([Upsample(filters) for i in range(num_upsample_block)])
    
    def forward(self, x):
        x = normalize(x)
        x = self.conv1(x)
        x = x_1 = self.prelu(x)
        for block in self.residual_part:
            x = block(x)
        
        x = self.conv2(x)
        x = self.batch1(x)
        x.add_(x_1)
        for block in self.upsample:
            x = block(x)
        
        x = self.conv3(x)
        x = self.tanh(x)
        x = denormalize(x)

        return x


# discriminator

class Dis_block(torch.nn.Module):
    def __init__(self, filter_in, filter_out, stride=1, momentum=0.8):
        """
            [Conv2D] --> [batchnormalization (if True)] --> [LeakyReLU]
        """
        super(Dis_block,self).__init__()
        self.conv = torch.nn.Conv2d(filter_in, filter_out, 3, padding=1)
        self.batch = torch.nn.BatchNorm2d(filter_out, momentum=momentum)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)
    
    def forward(self, x, batchnorm=True):
        x = self.conv(x)
        if batchnorm:
            x = self.batch(x)
        x = self.leaky(x)
        return x
    

class Discriminator(torch.nn.Module):
    def __init__(self, filters, num_dis):
        super(Discriminator,self).__init__()
        """
            [Dis_block]*(2*num_dis) --> [AdaptiveAveragePooling] --> [Conv2D] --> [LeakyReLU] --> [Conv2D] --> [Sigmoid]
            the result will be Conv Form so convert into normal form
        """
        self.dis_blocks = torch.nn.ModuleList([Dis_block(3, filters) , Dis_block(filters, filters, stride=2)])
        for i in range(1,num_dis):
            self.dis_blocks.append(Dis_block(filters, filters*2))
            self.dis_blocks.append(Dis_block(filters*2, filters*2, stride=2))
            filters*=2
        self.Adaptive = torch.nn.AdaptiveAvgPool2d(1)
        self.conv1 = torch.nn.Conv2d(512, 1024, 1)
        self.leaky = torch.nn.LeakyReLU(negative_slope=0.2)
        self.conv2 = torch.nn.Conv2d(1024, 1, 1)
        self.sig = torch.nn.Sigmoid()
    
    def forward(self, x):
        shape = x.size()
        batch_size = shape[0]
        for block in self.dis_blocks:
            x = block(x)
        x = self.Adaptive(x)
        x = self.conv1(x)
        x = self.leaky(x)
        x = self.conv2(x)
        x = self.sig(x)    
        return torch.squeeze(x).view(batch_size)

if __name__=='__main__':
    net = Discriminator(64 ,4)
    print(net)
    x = torch.randn(3,3,96,96)
    y = net(x)
    print(y)
    print(torch.squeeze(y).view(3,1)) # use this to take input to get in form [batch_size, 1] as result is single neuron







# net = Generator()
# print(net)
# x = torch.randn(1,3,24,24)
# y = torch.randn(1,3,96,96)
# loss_fn = torch.nn.MSELoss(reduce='sum')
# optimizer = torch.optim.Adam(net.parameters(),lr=0.001)

# for i in range(20):
#     y_pred = net(x)

#     loss = loss_fn(y_pred, y)
#     print(i, loss.item())

#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
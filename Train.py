import os
import math
import pandas as pd 
import torch
import torchvision
from loss_fn import Generator_loss
from networks import Generator, Discriminator
from DataPipeline import Train_Dataset_Folder, Valid_dataset_folder, Test_dataset_folder

UPSCALE_FACTOR = 4
NUM_EPOCHS = 5
train_path= " "     #enter the folder path
validation_path = " " # enter the folder path validation
test_path = " "  # enter the folder path test

train_set = Train_Dataset_Folder(train_path,upscale_factor=UPSCALE_FACTOR)
#val_set = Valid_dataset_folder(validation_path, upscale_factor=UPSCALE_FACTOR)

train_loader = torch.utils.data.DataLoader(dataset=train_set, num_workers=4, batch_size=16,shuffle=True)
#val_loader = torch.utils.data.DataLoader(dataset=val_set, num_workers=4, batch_size=1,shuffle=False)

generator = Generator()
discriminator = Discriminator(64,4)

G_loss = Generator_loss()

need_cuda=False   # make it True to activate GPU processing
if torch.cuda.is_available() and need_cuda: # to shift model to GPU
    generator.cuda()
    discriminator.cuda()
    G_loss.cuda()

optimG = torch.optim.Adam(generator.parameters())
optimD = torch.optim.Adam(discriminator.parameters())

result = {'d_loss':[], 'g_loss':[], 'd_score':[], 'g_score':[], 'psnr':[], 'ssim':[]}

for epoch in range(1, NUM_EPOCHS+1):
    run_result = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    for data, target in train_loader:
        batch_size = data.size(0)
        run_result['batch_size']+=batch_size

        real_image = torch.autograd.Variable(target) # high resolution image
        z = torch.autograd.Variable(data) #low resolution image is put in pytorch tensor variable
        if torch.cuda.is_available() and need_cuda: # to shift to GPU
            real_image = real_image.cuda()
            z = z.cuda()
        
        gen_image = generator(z)
        discriminator.zero_grad()

        real_out = discriminator(real_image).mean()
        gen_out = discriminator(gen_image).mean()
        d_loss = 1- real_out + gen_out
        d_loss.backward(retain_graph=True)
        optimD.step()

        generator.zero_grad()
        g_loss = G_loss(gen_out, gen_image, real_image)
        g_loss.backward()

        gen_img = generator(z)
        gen_out = discriminator(gen_image).mean()

        optimG.step()
        run_result['g_loss'] += g_loss.item() * batch_size
        run_result['d_loss'] += d_loss.item() * batch_size
        run_result['d_score'] += real_out.item() * batch_size
        run_result['g_score'] += gen_out.item() * batch_size

        optimG.zero_grad()
        optimD.zero_grad()
        print(g_loss.item()* batch_size, d_loss.item()* batch_size)
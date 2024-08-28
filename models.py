import torch
from torch import nn
from torch import optim

class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # input is ``(nc) x 64 x 64``
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. ``(ndf) x 32 x 32``
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. ``(ndf*2) x 16 x 16``
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. ``(ndf*4) x 8 x 8``
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            # state size. ``(ndf*8) x 4 x 4``
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        noise = torch.randn_like(input) * 0.05  # Adding noise to the inputs
        input = input + noise
        return self.main(input)
    


class Generator(nn.Module):
    def __init__(self, latent_size, ngf, nc, device):
        super(Generator, self).__init__()
        self.device = device
        self.block1 = self.block(latent_size, ngf * 8, 4, 1, 0)
        self.block2 = self.block(ngf * 8, ngf * 4, 4, 2, 1)
        self.block3 = self.block(ngf * 4, ngf * 2, 4, 2, 1)
        self.block4 = self.block(ngf * 2, ngf, 4, 2, 1)

        self.last_layer = nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False)
        self.tanh = nn.Tanh()
    
    def block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            NoiseInjection(device=self.device),
            nn.ReLU()
            )

    def forward(self, input):
        output = self.block1(input)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.last_layer(output)
        output = self.tanh(output)

        return output
    
class NoiseInjection(nn.Module):
    def __init__(self, device):
        self.device = device
        super(NoiseInjection, self).__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image):
        noise = torch.randn_like(image, device=self.device)
        return image + self.weight * noise
    



class GAN:
    def __init__(self, generator, discriminator, device, latent_size, batch_size, lr_discriminator=0.0002, lr_generator=0.0002):
        self.device = device
        self.criterion = nn.BCELoss()
        self.fixed_noise = torch.randn(10, latent_size, 1, 1, device=device)
        self.latent_size = latent_size
        self.batch_size = batch_size

        self.models = {"generator": generator.to(device),
                      "discriminator": discriminator.to(device)}
        
        self.labels = {"real": torch.ones(batch_size).to(device),
                       "fake": torch.zeros(batch_size).to(device)}
        
        self.optimizers = {"generator": optim.Adam(self.models["generator"].parameters(), lr=lr_generator, betas=(0.5, 0.999)),
                           "discriminator": optim.Adam(self.models["discriminator"].parameters(), lr=lr_discriminator, betas=(0.5, 0.999))}
    
    def train_step(self, images, model_name, label_name):
        model = self.models[model_name]
        model.zero_grad()
        error, accuracy = self.get_error_accuracy(images, label_name)
        error.backward()
        self.optimizers[model_name].step()
        return error, accuracy

    def generate(self, fix=False):
        self.models["generator"].zero_grad()
        noise = torch.randn(self.batch_size, self.latent_size, 1, 1, device=self.device) if not fix else self.fixed_noise
        fake = self.models["generator"](noise)
        return fake

    def discriminate(self, images):
        output = self.models["discriminator"](images).view(-1)
        return output
    
    def get_error_accuracy(self, images, label_name):
        output = self.models["discriminator"](images).view(-1)
        error = self.criterion(output, self.labels[label_name])
        accuracy = ((output > 0.5) == self.labels[label_name]).sum() / len(output)
        return error, accuracy
    




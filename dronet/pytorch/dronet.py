import torch
from torch import nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn import functional as F
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
import os

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes):
        super(ResidualBlock, self).__init__()

        self.bn1 = torch.nn.BatchNorm2d(in_planes)
        self.act_fn = torch.nn.ReLU()
        self.conv1 = torch.nn.Conv2d(in_planes,planes,(3,3),stride=(2,2),padding=(1,1))
        self.bn2 = torch.nn.BatchNorm2d(planes)
        self.conv2 = torch.nn.Conv2d(planes,planes,(3,3),stride=(1,1),padding=(1,1))
        self.conv3 = torch.nn.Conv2d(planes,planes,(1,1),stride=(2,2),padding=(0,0))

    def forward(self, input):
        residual = input
        #batch_norm -> ReLU -> conv
        output = self.conv1(self.act_fn(self.bn1(input)))
        output = self.conv2(self.act_fn(self.bn2(output)))

        output = self.conv3(output)
        output += residual
        return output

class Dronet(pl.LightningModule):

  def __init__(self,input_channels = 3, output_dims = 1):
    super(Dronet, self).__init__()
    self.input_channels = input_channels
    self.act_fn = torch.nn.ReLU()
    self.conv1 = torch.nn.Conv2d(self.input_channels,32,(5,5),stride=(2,2),padding=(2,2))
    self.max_pool = torch.nn.MaxPool2d(kernel_size=(3,3),stride=(2,2))

    # Residual blocks
    self.res_block1 = ResidualBlock(in_planes=32, planes=32)
    self.res_block2 = ResidualBlock(in_planes=32, planes=64)
    self.res_block3 = ResidualBlock(in_planes=64, planes=128)

    self.dropout = torch.nn.Dropout(0.5)
    self.fc = torch.nn.Linear(6272,output_dims)

  def forward(self, input):
    batch_size, channels, width, height = input.size()
    input = self.conv1(input)
    input = self.max_pool(input)

    #Residual blocks
    input = self.res_block1(input)
    input = self.res_block2(input)
    input = self.res_block3(input)

    input = torch.flatten(input)
    input = self.act_fn(input)
    input = self.dropout(input)
    output = self.fc(input)

    return output

  def cross_entropy_loss(self, logits, labels):
    #defining the loss function
    # return F.nll_loss(logits, labels)

  def training_step(self, train_batch, batch_idx):
    #Use of loss and output labels
    # x, y = train_batch
    # logits = self.forward(x)
    # loss = self.cross_entropy_loss(logits, y)
    #
    # logs = {'train_loss': loss}
    # return {'loss': loss, 'log': logs}

  def validation_step(self, val_batch, batch_idx):
    # x, y = val_batch
    # logits = self.forward(x)
    # loss = self.cross_entropy_loss(logits, y)
    # return {'val_loss': loss}

  def validation_epoch_end(self, outputs):
      # called at the end of the validation epoch
      # outputs is an array with what you returned in validation_step for each batch
      # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
      # avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
      # tensorboard_logs = {'val_loss': avg_loss}
      # return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

  def prepare_data(self):
    # transforms for images
    # transform=transforms.Compose([transforms.ToTensor(),
    #                               transforms.Normalize((0.1307,), (0.3081,))])
    #
    # # prepare transforms standard to MNIST
    # mnist_train = MNIST(os.getcwd(), train=True, download=True, transform=transform)
    # mnist_test = MNIST(os.getcwd(), train=False, download=True, transform=transform)
    #
    # self.mnist_train, self.mnist_val = random_split(mnist_train, [55000, 5000])

  def train_dataloader(self):
    #return DataLoader(self.mnist_train, batch_size=64)

  def val_dataloader(self):
    #return DataLoader(self.mnist_val, batch_size=64)

  def test_dataloader(self):
    #return DataLoader(self,mnist_test, batch_size=64)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
    return optimizer

# train
model = LightningMNISTClassifier()
trainer = pl.Trainer()

trainer.fit(model)

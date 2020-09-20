import numpy as np
import torch
from torch import nn
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from skimage import io,transform
from torchvision import datasets, transforms
import os
import re

class SteeringAngleDataset(Dataset):
    """Udacity steering angle dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.samples = 0
        self.num_experiments = 0
        self.formats = {'png', 'jpg'}
        # Associate each filename with a corresponding steering or label
        self.filenames = []
        #The ground_truth list consists of labels for both the type of experiments
        self.ground_truth = []
        # Determine the type of experiment (steering or collision) to compute
        # the loss. This is also used to associate labels with the corresponding experiment
        self.exp_type = []


    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        '''First count how many experiments/folders are out there. Experiments/set of images corresponding
        to steering angle prediction and collision classification are considered together
        '''
        experiments = []
        for subdir in sorted(os.listdir(self.root_dir)):
            if os.path.isdir(os.path.join(self.root_dir, subdir)):
                experiments.append(subdir)
        self.num_experiments = len(experiments)


        for subdir in experiments:
            subpath = os.path.join(self.root_dir, subdir)
            self._decode_experiment_dir(subpath)

        # Conversion of list into array
        #a single groundtruth array which contains labels of the images loaded.
        # Nature of the labels(integer or float probably decides their correspondence)
        self.ground_truth = np.array(self.ground_truth, dtype = np.float)

        assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))

        # img_name = os.path.join(self.root_dir,
        #                         self.landmarks_frame.iloc[idx, 0])
        img_name = self.filenames[idx]
        image = io.imread(img_name)
        print(type(image))
        # landmarks = self.landmarks_frame.iloc[idx, 1:]
        # landmarks = np.array([landmarks])
        label = self.ground_truth[idx]
        label=np.float(label)
        sample = {'image': image, 'label': label}

        if self.transform:
            #image: np.ndarray is converted first to a PIL Image to apply transformations and then converted
            # back to a np.ndarray)
            sample['image'] = np.asarray(self.transform(sample['image']))

        return sample

    def _decode_experiment_dir(self, dir_subpath):
        # Load steerings or labels in the experiment dir
        steerings_filename = os.path.join(dir_subpath, "sync_steering.txt")
        labels_filename = os.path.join(dir_subpath, "labels.txt")

        # Try to load steerings first. Make sure that the steering angle or the
        # label file is in the first column. Note also that the first line are
        # comments so it should be skipped.
        try:
            ground_truth = np.loadtxt(steerings_filename, usecols=0,
                                  delimiter=',', skiprows=1)
            exp_type = 1
        except OSError as e:
            # Try load collision labels if there are no steerings
            try:
                ground_truth = np.loadtxt(labels_filename, usecols=0)
                exp_type = 0
            except OSError as e:
                print("Neither steerings nor labels found in dir {}".format(
                dir_subpath))
                raise IOError


        # Now fetch all images in the image subdir
        image_dir_path = os.path.join(dir_subpath, "images")
        for root, _, files in self._recursive_list(image_dir_path):
            sorted_files = sorted(files,
                    key = lambda fname: int(re.search(r'\d+',fname).group()))
            for frame_number, fname in enumerate(sorted_files):
                is_valid = False
                for extension in self.formats:
                    if fname.lower().endswith('.' + extension):
                        is_valid = True
                        break
                if is_valid:
                    absolute_path = os.path.join(root, fname)
                    # self.filenames.append(os.path.relpath(absolute_path,
                    #         self.root_dir))
                    self.filenames.append(absolute_path)
                    self.ground_truth.append(ground_truth[frame_number])
                    self.exp_type.append(exp_type)
                    self.samples += 1

    def _recursive_list(self, subpath):
        return sorted(os.walk(subpath, followlinks=False),key=lambda tpl: tpl[0])

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

  def mse_loss(self, prediction, target):
    #defining the loss function
    return F.mse_loss(prediction,target)

  def training_step(self, train_batch, batch_idx):
    #Use of loss and output labels
    inputs, targets = train_batch
    predictions = self.forward(inputs)
    loss = self.mse_loss(predictions, targets)
    logs = {'train_loss': loss}
    return {'loss': loss, 'log': logs}

  def validation_step(self, val_batch, batch_idx):
    inputs, targets = val_batch
    predictions = self.forward(inputs)
    loss = self.mse_loss(predictions, targets)
    return {'val_loss': loss}

  def validation_epoch_end(self, outputs):
    # called at the end of the validation epoch
    # outputs is an array with what you returned in validation_step for each batch
    # outputs = [{'loss': batch_0_loss}, {'loss': batch_1_loss}, ..., {'loss': batch_n_loss}]
    avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
    tensorboard_logs = {'val_loss': avg_loss}
    return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}

  def prepare_data(self):
      pass
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
      pass
    #return DataLoader(self.mnist_train, batch_size=64)

  def val_dataloader(self):
      pass
    #return DataLoader(self.mnist_val, batch_size=64)

  def test_dataloader(self):
      pass
    #return DataLoader(self,mnist_test, batch_size=64)

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.parameters(), lr=1e-2,weight_decay=1e-5)
    return optimizer

# train
# model = Dronet()
# trainer = pl.Trainer()
#
# trainer.fit(model)
if __name__ == '__main__':
    image_transformations = transforms.Compose([transforms.ToPILImage(),
                                   transforms.Resize((240,320)),
                                   transforms.CenterCrop((200,200)),
                                   transforms.Grayscale(1)])
    face_dataset = SteeringAngleDataset(root_dir='/home/ash/HBRS/Research/ResearchThesis/SourceCodes/'
                                        'benchmark-uncertainty-estimation-methods-regression/Dataset/'
                                                 'udacity_steering_angle/converted_bag/trial/',
                                        transform=image_transformations)
    i = face_dataset[1]
    io.imshow(np.asarray(i['image']))
    io.show()
    exit(0)

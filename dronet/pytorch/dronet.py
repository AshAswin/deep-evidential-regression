import numpy as np
import torch
from torch import nn,Tensor
import pytorch_lightning as pl
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torch.optim.optimizer import Optimizer
from skimage import io,transform
from torchvision import datasets, transforms
from pytorch_lightning import metrics
import os
import re
#to ensure reproducibility while splitting the dataset
torch.manual_seed(0)

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

        self.ground_truth = np.array(self.ground_truth, dtype = np.float)

        assert self.samples > 0, "Did not find any data"

        print('Found {} images belonging to {} experiments.'.format(
                self.samples, self.num_experiments))

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Conversion of list into array
        #a single groundtruth array which contains labels of the images loaded.
        # Nature of the labels(integer or float probably decides their correspondence)


        img_name = self.filenames[idx]
        image = io.imread(img_name)
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
        self.conv3 = torch.nn.Conv2d(in_planes,planes,(1,1),stride=(2,2),padding=(0,0))

    def forward(self, input):
        residual = input
        #batch_norm -> ReLU -> conv
        output = self.conv1(self.act_fn(self.bn1(input)))
        output = self.conv2(self.act_fn(self.bn2(output)))

        residual = self.conv3(residual)
        output += residual
        return output

class Dronet(pl.LightningModule):

    def __init__(self,input_channels = 1, output_dims = 1,data_dir = os.getcwd()):
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

        self.data_dir = data_dir

        self.metric = metrics.regression.RMSE()
        #Initialize convolution layer weights of residual blocks using kaiming_normal intializer
        self.__init_res_block_conv2D()

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
        #This is to group feature of an image in the second dimension
        input = input.view(-1,6272)
        output = self.fc(input)

        return output

    def mse_loss(self, prediction, target):
        #defining the loss function
        return F.mse_loss(prediction,target)


    def on_batch_start(self,batch)-> None:
        #tensors need to be converted to PIL Image before applying any augmentation transformations
        tensor_PIL = transforms.ToPILImage()
        augment_transforms = transforms.RandomAffine(degrees=0.2,translate=(0.2,0.2))
        #After applying augmentation transformations, PIL image is again converted to tensor
        normalize_image = transforms.ToTensor()
        batch_transforms = transforms.Compose([tensor_PIL, augment_transforms, normalize_image])
        #Applying transformations to images separately
        for image_idx in (range(batch['image'].shape[0])):
            (batch['image'])[image_idx] = batch_transforms((batch['image'])[image_idx])

    def training_step(self, train_batch, batch_idx):
        #Use of loss and output labels
        inputs, targets = train_batch['image'], train_batch['label']
        targets = targets.float()
        targets = torch.unsqueeze(targets,dim = 1)
        predictions = self.forward(inputs)

        rmse = self.metric(predictions,targets)

        loss = self.mse_loss(predictions, targets)
        logs = {'train_loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, val_batch, batch_idx):
        inputs, targets = val_batch['image'], val_batch['label']
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
        pre_process_transforms  = transforms.Compose([transforms.ToPILImage(),
                                     transforms.Resize((240,320)),
                                     transforms.CenterCrop((200,200)),
                                     transforms.Grayscale(1),
                                    transforms.ToTensor()])
        #Need to figure out how augmentation (which is applied every epoch) works in PyTorch lightning


        self.steer_angle_data = SteeringAngleDataset(self.data_dir, transform = pre_process_transforms)
        # self.steer_angle_val_data = SteeringAngleDataset(os.path.join(self.data_dir,'validation'),
        #                                                  transform=pre_process_transforms)
        # self.steer_angle_test_data = SteeringAngleDataset(os.path.join(self.data_dir,'testing'),
        #                                                   transform=pre_process_transforms)
        train_count, val_count, test_count = self._find_data_split(self.steer_angle_data.__len__(),
                                                                   train_percent = 0.8,
                                                                   val_percent = 0.1,
                                                                   test_percent= 0.1)
        self.steer_angle_train_data, self.steer_angle_val_data, self.steer_angle_test_data =\
                                    random_split(self.steer_angle_data,[train_count,val_count,test_count])

    def train_dataloader(self):
        return DataLoader(self.steer_angle_train_data,batch_size=32,shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.steer_angle_val_data,batch_size=32)
  

    def test_dataloader(self):
        return DataLoader(self.steer_angle_test_data,batch_size=32)


    def test_step(self, test_batch, test_batch_idx):
        inputs, targets = test_batch['image'], test_batch['label']
        predictions = self(inputs)
        return {'metric':self.metric(predictions,targets)}

    def test_epoch_end(self, outputs):

        #avg_metric = torch.stack([x['metric'] for x in outputs]).mean()
        final_rmse = 0
        for batch_rmse in outputs:
            #square batch rmse and multiply by batch size to get the deviation
            batch_rmse = (batch_rmse**2)*batch_rmse.__len__()
            final_rmse+=batch_rmse

        final_rmse = (final_rmse/outputs.__len__())**0.5
        logs = {'rmse':final_rmse}
        return {'rmse':final_rmse,'log':logs}

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-2,weight_decay=1e-5)
        return optimizer

    def backward(self, trainer, loss: Tensor, optimizer: Optimizer, optimizer_idx: int) -> None:
        #Add L2 Regularizers for Conv layers of Res blocks
        l2_loss = torch.tensor(0, dtype=torch.float,requires_grad=True)
        reg = 1e-4

        for name, param in model.named_parameters():
            if 'res_block' in name and 'conv' in name and 'weight' in name:
                l2_loss = l2_loss + (0.5 * reg * torch.sum(torch.pow(param, 2)))

        loss += l2_loss
        loss.backward()

    def _find_data_split(self,dataset_length,train_percent = 0.8, val_percent = 0.1,test_percent= 0.1):
        train_count = int(train_percent*dataset_length)
        val_count = int(val_percent*dataset_length)
        test_count = int(test_percent*dataset_length)
        #To ensure that lengths of the splits sum upto the original dataset size
        length_difference = dataset_length - (train_count + val_count + test_count)
        if length_difference != 0:
            train_count = train_count + length_difference
        return train_count, val_count, test_count


    def __init_res_block_conv2D(self):
        for block in [self.res_block1,self.res_block2,self.res_block3]:
            for layer in block.modules():
                if isinstance(layer, (nn.Conv2d)):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in')


if __name__ == '__main__':
    model = Dronet(input_channels=1,output_dims=1,data_dir='/home/ash/HBRS/Research/ResearchThesis/SourceCodes/'
                                                        'benchmark-uncertainty-estimation-methods-regression/Dataset/'
                                                        'udacity_steering_angle/converted_bag/training')

    path = os.path.join(os.getcwd(),'lightning_logs/version_4/checkpoints/epoch=0.ckpt')

    #test_model = Dronet.load_from_checkpoint(checkpoint_path=path)



    #use of callbacks is another option to handle with checkpoints
    trainer = pl.Trainer(auto_select_gpus=True,
                         auto_lr_find=False,#auto learning rate finder
                         gpus=None,#number of gpus to train the model
                         log_save_interval=100,#frequency to save logs
                         max_epochs = 1,
                         min_epochs = 1,
                         reload_dataloaders_every_epoch=False,#consider using this to apply augmentation(random
                         resume_from_checkpoint=None,         # transforms) before every epoch
                         weights_save_path=os.getcwd()
                         )

    trainer.fit(model)
    trainer.test()

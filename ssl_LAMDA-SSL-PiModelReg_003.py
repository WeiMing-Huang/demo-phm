# %%
import csv
import utils
import augmentation as aug
from simsiam import SimSiam
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import train_test_split
import numpy as np
import gc
from sklearn.model_selection import KFold
from datetime import datetime
import argparse
import optuna
from proposed_model import SsDAT


import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = False
# device = torch.device("cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--modelname',
                    type=str,
                    default='Proposed',
                    help='the model name (default=Proposed)')
parser.add_argument('--dataset',
                    type=str,
                    default='FD003',
                    help='name of the dataset (default=FD001)')
parser.add_argument('--sequence_length',
                    type=int,
                    default=30,
                    help='input sequence length (default=30)')
parser.add_argument('--alpha',
                    type=float,
                    default=0.1,
                    help='input smoothing intensity (default=0.1)')
parser.add_argument('--threshold',
                    type=int,
                    default=130,
                    help='input max rul (default=130)')
parser.add_argument('--wp',
                    type=bool,
                    default=True,
                    help='using wavelet packet or not (default=True)')
parser.add_argument('--batch',
                    type=int,
                    default=32,
                    help='input batch size (default=64)')
parser.add_argument('--fold',
                    type=int,
                    default=5,
                    help='how many folds cross validations are used (default=5)')
parser.add_argument('--label_size',
                    type=float,
                    default=0.2,
                    help='how many proportions in the dataset have labels (default=0.2)')
parser.add_argument('--train_epochs',
                    type=int,
                    default=100,
                    help='how many epochs in training loop (default=100)')
parser.add_argument('--train_patients',
                    type=int,
                    default=2,
                    help='how many patients in training loop (default=2)')
parser.add_argument('--tune_epochs',
                    type=int,
                    default=100,
                    help='how many epochs in tuning loop (default=100)')
parser.add_argument('--tune_patients',
                    type=int,
                    default=10,
                    help='how many patients in tuning loop (default=10)')
parser.add_argument('--save_place',
                    type=str,
                    default='result.csv',
                    help='where to save the result (default=result.csv)')

config = parser.parse_args()


experiment = f'{config.dataset}_{config.modelname}_{config.label_size}_optuna_model_best_cv_dict_{datetime.now().today().strftime("%m_%d_%H_%M_%S")}'

FILE = f'./savemodel/{experiment}.pt'


random_aug = [aug.jitter, aug.scaling,
              aug.permutation, aug.magnitude_warp, aug.time_warp]

sensors = ['s_{}'.format(i+1) for i in range(0, 21)]

x_train, y_train, x_val, y_val, x_test, y_test = utils.get_data(dataset=config.dataset,
                                                                sensors=sensors,
                                                                sequence_length=config.sequence_length,
                                                                alpha=config.alpha,
                                                                threshold=config.threshold,
                                                                wp=config.wp)

x_final = np.concatenate((x_train, x_val))
y_final = np.concatenate((y_train, y_val))

train_dataset = utils.ElecDataset(x_final, y_final, train=True)


x_train=np.reshape(x_train,(x_train.shape[0],-1))
x_val=np.reshape(x_val,(x_val.shape[0],-1))
x_test=np.reshape(x_test,(x_test.shape[0],-1))


unlabeled_X, labeled_X, unlabeled_y, labeled_y = train_test_split(x_train, y_train, test_size=0.2, random_state=42)



from LAMDA_SSL.Augmentation.Tabular.Noise import Noise
import torch.nn as nn
from LAMDA_SSL.Dataset.Tabular.Boston import Boston
from LAMDA_SSL.Opitimizer.SGD import SGD
from LAMDA_SSL.Opitimizer.Adam import Adam
from LAMDA_SSL.Scheduler.CosineAnnealingLR import CosineAnnealingLR
from LAMDA_SSL.Network.MLPReg import MLPReg
from LAMDA_SSL.Dataset.LabeledDataset import LabeledDataset
from LAMDA_SSL.Dataset.UnlabeledDataset import UnlabeledDataset
from LAMDA_SSL.Dataloader.LabeledDataloader import LabeledDataLoader
from LAMDA_SSL.Dataloader.UnlabeledDataloader import UnlabeledDataLoader
from LAMDA_SSL.Algorithm.Regression.PiModelReg import PiModelReg
from LAMDA_SSL.Sampler.RandomSampler import RandomSampler
from LAMDA_SSL.Sampler.SequentialSampler import SequentialSampler
from LAMDA_SSL.Evaluation.Regressor.Mean_Absolute_Error import Mean_Absolute_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Error import Mean_Squared_Error
from LAMDA_SSL.Evaluation.Regressor.Mean_Squared_Log_Error import Mean_Squared_Log_Error
import numpy as np



labeled_sampler=RandomSampler(replacement=True,num_samples=64*(4000))
unlabeled_sampler=RandomSampler(replacement=True)
valid_sampler=SequentialSampler()
test_sampler=SequentialSampler()

#dataloader
labeled_dataloader=LabeledDataLoader(batch_size=64,num_workers=1,drop_last=True)
unlabeled_dataloader=UnlabeledDataLoader(num_workers=1,drop_last=True)
valid_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=1,drop_last=False)
test_dataloader=UnlabeledDataLoader(batch_size=64,num_workers=1,drop_last=False)

# augmentation

augmentation=Noise(noise_level=0.01)

# optimizer
optimizer=Adam()
scheduler=CosineAnnealingLR(eta_min=0,T_max=4000)

# network
network=MLPReg(hidden_dim=[1000,500,100,50,10],activations=[nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU(),nn.ReLU()],dim_in=labeled_X.shape[-1])

evaluation={
    'Mean_Absolute_Error':Mean_Absolute_Error(),
    'Mean_Squared_Error':Mean_Squared_Error(),
    'Mean_Squared_Log_Error':Mean_Squared_Log_Error()
}

file = open("./LAMDA-SSL-PiModelReg3.txt", "w")

model=PiModelReg(lambda_u=0.01,warmup=0.4,
               mu=1,weight_decay=5e-4,ema_decay=0.999,
               epoch=20,num_it_epoch=200,
               num_it_total=4000,
               eval_it=200,device='cuda:0',
               labeled_sampler=labeled_sampler,
               unlabeled_sampler=unlabeled_sampler,
               valid_sampler=valid_sampler,
               test_sampler=test_sampler,
               labeled_dataloader=labeled_dataloader,
               unlabeled_dataloader=unlabeled_dataloader,
               valid_dataloader=valid_dataloader,
               test_dataloader=test_dataloader,
               augmentation=augmentation,network=network,
               optimizer=optimizer,scheduler=scheduler,
               evaluation=evaluation,file=file,verbose=True)




model.fit(X=labeled_X,y=np.squeeze(labeled_y),unlabeled_X=unlabeled_X,valid_X=x_val,valid_y=np.squeeze(y_val))

performance=model.evaluate(X=x_test,y=np.squeeze(y_test))

result=model.y_pred

print(result,file=file)

print(performance,file=file)


rmse = utils.evaluate(y_test, np.expand_dims(result, axis=1))
res = utils.score(y_test, np.expand_dims(result, axis=1))

print(rmse,file=file)
print(res,file=file)
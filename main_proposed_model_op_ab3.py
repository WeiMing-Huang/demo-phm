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
from proposed_model_ab3 import SsDAT


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
                    default='FD001',
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
                    default=64,
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

train_losses = []
eval_train_losses = []
eval_val_losses = []
save_rmse = []
save_res = []


def train_loop(model, which_model, train_epochs, train_dataloader, train_loss_fn, train_optimizer, train_scheduler, train_patients):

    rec_loss = 999
    patient = 0
    train_model_state = model.state_dict()

    for epoch in range(train_epochs):

        print('epochs {}/{}'.format(epoch+1, train_epochs))
        training_loss = .0

        model.train()

        for idx, (inputs, labels) in enumerate(train_dataloader):

            inputs1 = torch.from_numpy(random.choice(
                random_aug)(inputs.numpy())).to(device)
            inputs2 = torch.from_numpy(random.choice(
                random_aug)(inputs.numpy())).to(device)
            inputs3 = torch.from_numpy(inputs.numpy()).to(device)

            if which_model == 1:
                pass
            elif which_model == 2:
                inputs1 = inputs1.unsqueeze(1)
                inputs2 = inputs2.unsqueeze(1)
                inputs3 = inputs3.unsqueeze(1)
            else:
                pass
            # Compute prediction and loss

            train_optimizer.zero_grad()

            z1, z2, z3, p1, p2, p3 = model(
                inputs1.float(), inputs2.float(), inputs3.float())

            loss = train_loss_fn(z1, z2, z3, p1, p2, p3)

            loss.backward()

            train_optimizer.step()

            training_loss += loss

        train_loss = training_loss/len(train_dataloader)

        train_losses.append(train_loss.cpu().detach().numpy())

        train_scheduler.step(train_loss)

        print(f'train_loss {train_loss:.4f}')

        gc.collect()

        pat_loss = train_loss.cpu().detach().numpy()

        if rec_loss > pat_loss:
            rec_loss = pat_loss
            patient = 0
            train_model_state = model.state_dict()
            print('now_patient=', patient)

        else:
            patient += 1
            print('now_patient=', patient)
            if patient > train_patients:
                model.load_state_dict(train_model_state)
                print('######training loop stop######')
                break


def tune_loop(model_tune, which_model, tune_epochs, tune_dataloader, val_loader, tune_loss_fn, tune_optimizer, tune_scheduler, tune_patients):

    rec_loss = 9999999
    patient = 0
    tune_model_state = model_tune.state_dict()

    for epoch in range(tune_epochs):

        print('epochs {}/{}'.format(epoch+1, tune_epochs))
        running_train_loss = .0
        running_val_loss = .0

        # train
        model_tune.train()

        for idx, (inputs, labels) in enumerate(tune_dataloader):

            if which_model == 1:
                pass
            elif which_model == 2:
                inputs = inputs.unsqueeze(1)
            else:
                pass

            inputs = inputs.to(device).float()
            labels = labels.to(torch.float32).to(device)

            tune_optimizer.zero_grad()

            preds = model_tune(inputs)

            loss = tune_loss_fn(preds, labels)

            loss.backward()

            tune_optimizer.step()

            running_train_loss += loss

        eval_train_loss = running_train_loss/len(tune_dataloader)
        eval_train_losses.append(eval_train_loss.cpu().detach().numpy())

        # val
        model_tune.eval()
        with torch.no_grad():
            for idx, (inputs, labels) in enumerate(val_loader):

                if which_model == 1:
                    pass
                elif which_model == 2:
                    inputs = inputs.unsqueeze(1)
                else:
                    pass

                inputs = inputs.to(device).float()
                labels = labels.to(torch.float32).to(device)

                preds = model_tune(inputs)

                loss = tune_loss_fn(preds, labels)

                running_val_loss += loss

        tune_scheduler.step(running_val_loss)

        eval_val_loss = running_val_loss/len(val_loader)
        eval_val_losses.append(eval_val_loss.cpu().detach().numpy())

        print(f'tune_train_loss {eval_train_loss:.4f}',
              f', tune_valid_loss {eval_val_loss:.4f}')

        gc.collect()

        if rec_loss > eval_val_loss:
            rec_loss = eval_val_loss
            patient = 0
            tune_model_state = model_tune.state_dict()
            print('now_patient=', patient)

        else:
            patient += 1
            print('now_patient=', patient)
            if patient > tune_patients:
                model_tune.load_state_dict(tune_model_state)
                print('######fine tune loop stop######')
                break


def test_loop(model_all, model_final, which_model, x_test, y_test):

    with torch.no_grad():

        if which_model == 1:
            preds = model_final(torch.from_numpy(x_test).to(
                device).float()).detach().cpu().numpy()
        elif which_model == 2:
            preds = model_final(torch.from_numpy(x_test).unsqueeze(
                1).to(device).float()).detach().cpu().numpy()
        else:
            pass

    rmse = utils.evaluate(y_test, preds)
    res = utils.score(y_test, preds)

    save_rmse.append(rmse)
    save_res.append(res)

    if min(save_rmse) == rmse:
        best_model_state = model_all.state_dict()
        torch.save(best_model_state, FILE)


def objective(trial):

    which_model = 2

    x_train_op, x_test_op, y_train_op, y_test_op = train_test_split(
        x_train, y_train, test_size=config.label_size)

    train_op_dataset = utils.ElecDataset(x_train_op, y_train_op, train=True)
    test_op_dataset = utils.ElecDataset(x_test_op, y_test_op, train=True)
    val_op_dataset = utils.ElecDataset(x_val, y_val, train=True)

    unlabel_op_loader = DataLoader(
        train_op_dataset, batch_size=config.batch, drop_last=True)
    label_op_loader = DataLoader(
        test_op_dataset, batch_size=config.batch, drop_last=True)
    val_op_loader = DataLoader(
        val_op_dataset, batch_size=config.batch, drop_last=True)

    hyperparams = {
        'in_channels': 512,
        'patchx_size': trial.suggest_categorical('patchx_size', [1, 2, 4]),
        'patchy_size': 6,
        'emb_size': trial.suggest_categorical('emb_size', [128, 256, 512]),
        'num_heads': trial.suggest_categorical('num_heads', [4, 8, 16]),
        'imgx_size': 4,
        'imgy_size': 6,
        'depth': trial.suggest_int('depth', 2, 8),
        'n_classes': 256,
    }

    model = SimSiam(backbone=SsDAT(**hyperparams), backbone_out=256).to(device)

    train_optimizer = torch.optim.SGD(model.parameters(), lr=trial.suggest_categorical(
        'lr', [0.5, 0.1, 0.05]), momentum=0.9, nesterov=True, weight_decay=0.01)

    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        train_optimizer, factor=0.09, verbose=True, patience=1)

    train_loop(model=model,
               which_model=which_model,
               train_epochs=config.train_epochs,
               train_dataloader=unlabel_op_loader,
               train_loss_fn=utils.ThreeBrenchLoss,
               train_optimizer=train_optimizer,
               train_scheduler=train_scheduler,
               train_patients=config.train_patients)

    model_final = model.final.to(device)

    tune_optimizer = torch.optim.Adam(model_final.parameters(), lr=0.1, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    tune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        tune_optimizer, 'min', patience=1, verbose=True)

    tune_loop(model_tune=model_final,
              which_model=which_model,
              tune_epochs=config.tune_epochs,
              tune_dataloader=label_op_loader,
              val_loader=val_op_loader,
              tune_loss_fn=nn.MSELoss(),
              tune_optimizer=tune_optimizer,
              tune_scheduler=tune_scheduler,
              tune_patients=config.tune_patients)

    with torch.no_grad():

        if which_model == 1:
            preds = model_final(torch.from_numpy(x_val).to(
                device).float()).detach().cpu().numpy()
        elif which_model == 2:
            preds = model_final(torch.from_numpy(x_val).unsqueeze(
                1).to(device).float()).detach().cpu().numpy()
        else:
            pass

    rmse = utils.evaluate(y_val, preds)
    gc.collect()
    return rmse
######


study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=5)


hyperparams = {
    'in_channels': 512,
    'patchx_size': study.best_trial.params['patchx_size'],
    'patchy_size': 6,
    'emb_size': study.best_trial.params['emb_size'],
    'num_heads': study.best_trial.params['num_heads'],
    'imgx_size': 4,
    'imgy_size': 6,
    'depth': study.best_trial.params['depth'],
    'n_classes': 256,
}


######

# %%
kfold = KFold(n_splits=config.fold, shuffle=True)

for fold, (train_ids, val_ids) in enumerate(kfold.split(train_dataset)):

    print(f'FOLD {fold}')
    print('--------------------------------')

    unlabel_ids, label_ids = train_test_split(
        train_ids, test_size=config.label_size)

    unlabel_subsampler = SubsetRandomSampler(unlabel_ids)
    label_subsampler = SubsetRandomSampler(label_ids)
    val_subsampler = SubsetRandomSampler(val_ids)

    unlabel_loader = DataLoader(
        train_dataset, batch_size=config.batch, sampler=unlabel_subsampler, drop_last=True)
    label_loader = DataLoader(
        train_dataset, batch_size=config.batch, sampler=label_subsampler, drop_last=True)
    val_loader = DataLoader(
        train_dataset, batch_size=config.batch, sampler=val_subsampler, drop_last=True)

    model = SimSiam(backbone=SsDAT(**hyperparams), backbone_out=256)

    model = model.to(device)

    train_optimizer = torch.optim.SGD(model.parameters(
    ), lr=study.best_trial.params['lr'], momentum=0.9, nesterov=True, weight_decay=0.01)

    train_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        train_optimizer, factor=0.09, verbose=True, patience=1)

    print('######training loop start######')

    train_loop(model=model,
               which_model=2,
               train_epochs=config.train_epochs,
               train_dataloader=unlabel_loader,
               train_loss_fn=utils.ThreeBrenchLoss,
               train_optimizer=train_optimizer,
               train_scheduler=train_scheduler,
               train_patients=config.train_patients)

    model_final = model.final.to(device)

    tune_optimizer = torch.optim.Adam(model_final.parameters(), lr=0.1, betas=(
        0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

    tune_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        tune_optimizer, 'min', patience=1, verbose=True)

    print('######fine tune loop start######')

    tune_loop(model_tune=model_final,
              which_model=2,
              tune_epochs=config.tune_epochs,
              tune_dataloader=label_loader,
              val_loader=val_loader,
              tune_loss_fn=nn.MSELoss(),
              tune_optimizer=tune_optimizer,
              tune_scheduler=tune_scheduler,
              tune_patients=config.tune_patients)

    print('######test loop start######')

    test_loop(model_all=model, model_final=model_final,
              which_model=2, x_test=x_test, y_test=y_test)

    print('--------------------------------')

print(
    'Five-fold cv RMSE : {:.2f} ± {:.2f}'.format(np.mean(save_rmse), np.std(save_rmse)))

print(
    'Five-fold cv Score : {:.2f} ± {:.2f}'.format(np.mean(save_res), np.std(save_res)))


# Create the dictionary (=row)
row = {'model': experiment, 'RMSE_m': np.mean(save_rmse), 'RMSE_s': np.std(save_rmse), 'Score_m': np.mean(
    save_res), 'Score_s': np.std(save_res), 'RMSE_all': save_rmse, 'Score_all': save_res}

# Open the CSV file in "append" mode
with open(config.save_place, 'a', newline='') as f:

    # Create a dictionary writer with the dict keys as column fieldnames
    writer = csv.DictWriter(f, fieldnames=row.keys())

    # Append single row to CSV
    writer.writerow(row)

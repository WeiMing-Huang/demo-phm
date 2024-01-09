# %%
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pywt
from sklearn.metrics import mean_squared_error, r2_score
import torch.nn.functional as F

class ElecDataset(Dataset):
    def __init__(self, feature, target=None, train=True):
        self.feature = feature
        self.target = target
        self.train = train

    def __len__(self):
        return len(self.feature)

    def __getitem__(self, idx):
        if self.train == True:
            item = self.feature[idx]
            label = self.target[idx]

            return item, label

        else:
            item = self.feature[idx]

            return item


def add_remaining_useful_life(df):
    grouped_by_unit = df.groupby(by="unit_nr")
    max_cycle = grouped_by_unit["time_cycles"].max()

    result_frame = df.merge(max_cycle.to_frame(
        name='max_cycle'), left_on='unit_nr', right_index=True)

    remaining_useful_life = result_frame["max_cycle"] - \
        result_frame["time_cycles"]
    result_frame["RUL"] = remaining_useful_life

    result_frame = result_frame.drop("max_cycle", axis=1)
    return result_frame


def add_operating_condition(df):
    df_op_cond = df.copy()

    df_op_cond['setting_1'] = abs(df_op_cond['setting_1'].round())
    df_op_cond['setting_2'] = abs(df_op_cond['setting_2'].round(decimals=2))
    df_op_cond['op_cond'] = df_op_cond['setting_1'].astype(str) + '_' + \
        df_op_cond['setting_2'].astype(str) + '_' + \
        df_op_cond['setting_3'].astype(str)

    return df_op_cond


def condition_scaler(df_train, df_test, sensor_names):
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']
                   == condition, sensor_names])
        df_train.loc[df_train['op_cond'] == condition, sensor_names] = scaler.transform(
            df_train.loc[df_train['op_cond'] == condition, sensor_names])
        df_test.loc[df_test['op_cond'] == condition, sensor_names] = scaler.transform(
            df_test.loc[df_test['op_cond'] == condition, sensor_names])
    return df_train, df_test


def exponential_smoothing(df, sensors, n_samples, alpha=0.4):
    df = df.copy()
    df[sensors] = df.groupby('unit_nr')[sensors].apply(
        lambda x: x.ewm(alpha=alpha).mean()).reset_index(level=0, drop=True)

    def create_mask(data, samples):
        result = np.ones_like(data)
        result[0:samples] = 0
        return result

    mask = df.groupby('unit_nr')['unit_nr'].transform(
        create_mask, samples=n_samples).astype(bool)
    df = df[mask]

    return df


def gen_train_data(df, sequence_length, columns):
    data = df[columns].values
    num_elements = data.shape[0]
    for start, stop in zip(range(0, num_elements-(sequence_length-1)), range(sequence_length, num_elements+1)):
        yield data[start:stop, :]


def gen_data_wrapper(df, sequence_length, columns, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    data_gen = (list(gen_train_data(df[df['unit_nr'] == unit_nr], sequence_length, columns))
                for unit_nr in unit_nrs)
    data_array = np.concatenate(list(data_gen)).astype(np.float32)
    return data_array


def gen_labels(df, sequence_length, label):
    data_matrix = df[label].values
    num_elements = data_matrix.shape[0]
    return data_matrix[sequence_length-1:num_elements, :]


def gen_label_wrapper(df, sequence_length, label, unit_nrs=np.array([])):
    if unit_nrs.size <= 0:
        unit_nrs = df['unit_nr'].unique()

    label_gen = [gen_labels(df[df['unit_nr'] == unit_nr], sequence_length, label)
                 for unit_nr in unit_nrs]
    label_array = np.concatenate(label_gen).astype(np.float32)
    return label_array


def gen_test_data(df, sequence_length, columns, mask_value):
    if df.shape[0] < sequence_length:
        data_matrix = np.full(shape=(sequence_length, len(
            columns)), fill_value=mask_value)  
        idx = data_matrix.shape[0] - df.shape[0]
        data_matrix[idx:, :] = df[columns].values 
    else:
        data_matrix = df[columns].values


    stop = data_matrix.shape[0]
    start = stop - sequence_length
    for i in list(range(1)):
        yield data_matrix[start:stop, :]


def get_data(dataset, sensors, sequence_length, alpha, threshold, wp):
    dir_path = './dataset/CMAPSSData/'
    train_file = 'train_'+dataset+'.txt'
    test_file = 'test_'+dataset+'.txt'

    index_names = ['unit_nr', 'time_cycles']
    setting_names = ['setting_1', 'setting_2', 'setting_3']
    sensor_names = ['s_{}'.format(i+1) for i in range(0, 21)]
    col_names = index_names + setting_names + sensor_names

    train = pd.read_csv((dir_path+train_file), sep=r'\s+', header=None,
                        names=col_names)
    test = pd.read_csv((dir_path+test_file), sep=r'\s+', header=None,
                       names=col_names)
    y_test = pd.read_csv((dir_path+'RUL_'+dataset+'.txt'), sep=r'\s+', header=None,
                         names=['RemainingUsefulLife'])

    train = add_remaining_useful_life(train)
    train['RUL'].clip(upper=threshold, inplace=True)

    drop_sensors = [
        element for element in sensor_names if element not in sensors]

    X_train_pre = add_operating_condition(train.drop(drop_sensors, axis=1))

    X_test_pre = add_operating_condition(test.drop(drop_sensors, axis=1))
    X_train_pre, X_test_pre = condition_scaler(
        X_train_pre, X_test_pre, sensors)

    X_train_pre = exponential_smoothing(X_train_pre, sensors, 0, alpha)
    X_test_pre = exponential_smoothing(X_test_pre, sensors, 0, alpha)

    gss = GroupShuffleSplit(n_splits=1, train_size=0.8, random_state=42)

    for train_unit, val_unit in gss.split(X_train_pre['unit_nr'].unique(), groups=X_train_pre['unit_nr'].unique()):
        train_unit = X_train_pre['unit_nr'].unique()[train_unit]
        val_unit = X_train_pre['unit_nr'].unique()[val_unit]

        x_train = gen_data_wrapper(
            X_train_pre, sequence_length, sensors, train_unit)
        y_train = gen_label_wrapper(
            X_train_pre, sequence_length, ['RUL'], train_unit)

        x_val = gen_data_wrapper(
            X_train_pre, sequence_length, sensors, val_unit)
        y_val = gen_label_wrapper(
            X_train_pre, sequence_length, ['RUL'], val_unit)
    test_gen = (list(gen_test_data(X_test_pre[X_test_pre['unit_nr'] == unit_nr], sequence_length, sensors, -99.))
                for unit_nr in X_test_pre['unit_nr'].unique())
    x_test = np.concatenate(list(test_gen)).astype(np.float32)

    if wp == True:
        wp_train = pywt.WaveletPacket(data=x_train, wavelet='db1', mode='symmetric', maxlevel=3)
        wp_val = pywt.WaveletPacket(data=x_val, wavelet='db1', mode='symmetric', maxlevel=3)
        wp_test = pywt.WaveletPacket(data=x_test, wavelet='db1', mode='symmetric', maxlevel=3)

        Wp_train = np.zeros((x_train.shape[0],x_train.shape[1],24))
        Wp_val = np.zeros((x_val.shape[0],x_val.shape[1],24))
        Wp_test = np.zeros((x_test.shape[0],x_test.shape[1],24))

        for j, t in enumerate([node.path for node in wp_train.get_level(3, 'natural')]):

            Wp_train[:, :, 3*j:3*(j+1)] = wp_train[t].data
            Wp_val[:, :, 3*j:3*(j+1)] = wp_val[t].data
            Wp_test[:, :, 3*j:3*(j+1)] = wp_test[t].data

        x_train = np.concatenate((x_train, Wp_train), axis=2)
        x_val = np.concatenate((x_val, Wp_val), axis=2)
        x_test = np.concatenate((x_test, Wp_test), axis=2)


    return x_train, y_train, x_val, y_val, x_test, y_test['RemainingUsefulLife']



def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{:.2f}, R2:{:.2f}'.format(label, rmse, variance))

    return rmse

def score(y_true, y_hat):
  res = 0
  for true, hat in zip(y_true, y_hat):
    subs = hat - true
    if subs < 0:
      res = res + np.exp(-subs/10)[0]-1
    else:
      res = res + np.exp(subs/13)[0]-1
  print("score: ", res)

  return res


def CosineSimilarity(p, z, version='simplified'):  
    if version == 'original':
        z = z.detach()  
        p = F.normalize(p, dim=1)  
        z = F.normalize(z, dim=1) 
        return -(p*z).sum(dim=1).mean()

    elif version == 'simplified':  
        return -1 *  F.cosine_similarity(p, z.detach(), dim=-1).mean()
    else:
        raise Exception

def ThreeBrenchLoss(z1, z2, z3, p1, p2, p3):

    
    loss1 = CosineSimilarity(p1, z2)/2 + CosineSimilarity(p2, z1)/2

    loss2 = CosineSimilarity(p2, z3)/2 + CosineSimilarity(p3, z2)/2

    loss3 = CosineSimilarity(p1, z3)/2 + CosineSimilarity(p3, z1)/2

    return (loss1+loss2+loss3)/3

def TwoBrenchLoss(z1, z2, p1, p2):
    
    loss1 = CosineSimilarity(p1, z2)/2 + CosineSimilarity(p2, z1)/2

    return loss1

 
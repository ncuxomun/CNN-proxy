#%%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
import torch
from torch import nn
import pytorch_lightning as pl
import torchmetrics as tm
from sklearn.metrics import r2_score, mean_squared_error
from adabelief_pytorch import AdaBelief
import torch.nn.functional as F
import os
import gc
from torchsummary import summary

seed = 999

# seeds and flags
np.random.seed(seed)
torch.manual_seed(seed)
pl.seed_everything(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.determinstic = True

sns.set_style("dark")
path = os.getcwd()

gc.collect()
torch.cuda.empty_cache()

# %%
# data module
class DataModule(pl.LightningDataModule):
    def __init__(self, dataset, batch_size=64, split=None, seed=0):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self.split = split

        if torch.cuda.is_available():
            self.pin = True
        else:
            self.pin = False

    def prepare_data(self):
        # download only
        self.dataset = self.dataset

    def setup(self, stage=None):
        # train/valid/test split
        # and assign to use in dataloaders via self
        train_set, valid_set, test_set = torch.utils.data.random_split(self.dataset, self.split, generator=torch.Generator().manual_seed(self.seed))

        if stage == 'fit' or stage is None:
            self.train_set = train_set
            self.valid_set = valid_set

        if stage == 'test' or stage is None:
            self.test_set = test_set

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size, pin_memory=self.pin, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.valid_set, batch_size=self.batch_size, pin_memory=self.pin, shuffle=False)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_set, batch_size=self.batch_size, pin_memory=self.pin, shuffle=False)

#%%
class REG(pl.LightningModule):
    def __init__(self, d_dim=100):
        super().__init__()
        self.activ = nn.SELU()
        inter_dim = 512
        conv_out = 1024
        inter_ch = 16

        # encoder model
        self.proxy = nn.Sequential(
            nn.Conv2d(1, inter_ch, 4, 2, 0),
            self.activ,
            nn.Conv2d(inter_ch, inter_ch*2, 3, 2, 0),
            self.activ,
            nn.Conv2d(inter_ch*2, inter_ch*4, 3, 2, 0),
            self.activ,
            nn.Conv2d(inter_ch*4, inter_ch*8, 2, 2, 0),
            self.activ,
            nn.Conv2d(inter_ch*8, inter_ch*16, 2, 1, 0),
            self.activ,
            nn.Flatten(),
            nn.Dropout(0.20),
            nn.Linear(conv_out, inter_dim),
            nn.BatchNorm1d(inter_dim),
            self.activ,
            nn.Linear(inter_dim, inter_dim//2),
            nn.BatchNorm1d(inter_dim//2),
            self.activ,
            nn.Dropout(0.20),
            nn.Linear(inter_dim//2, d_dim)
        )

    def forward(self, data):
        data_hat = self.proxy(data)
        return data_hat

# dummy = torch.ones((10, 1, 64, 64))

# reg = REG(150)
# _ = reg(dummy)
# print(_.shape)

#%%
class LitReg(pl.LightningModule):
    def __init__(self, d_dim, lr=2e-4):
        super().__init__()

        self.lr = lr
        self.mse = nn.MSELoss()

        # enhacer model
        self.reg = REG(d_dim)
        self.reg.apply(self.weights_init)

    def forward(self, maps):
        data_out = self.reg(maps)
        return data_out

    def training_step(self, batch, batch_idx):
        
        x, y = batch
        pred_y = self(x)
        train_loss = self.mse(pred_y, y)

        self.log('train_loss', train_loss, prog_bar=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'val')

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, 'test')

    def _shared_eval(self, batch, batch_idx, prefix):
        x, y = batch
        pred_y = self(x)
        loss = self.mse(pred_y, y)
        self.log(f'{prefix}_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), self.lr)
        optimizer = AdaBelief(self.parameters(), lr=self.lr, eps=1e-16, betas=(0.9, 0.999), weight_decouple=True, rectify=True)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.65, patience=30, verbose=True) #0.5 works
        # return optimizer
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0)
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, 0.0, 0.02)
            torch.nn.init.constant_(m.bias, 0.0)

#%%
# Dataset preparation
SLICE = 2000
MAPS = np.load('channels_uncond_10k.npy').T
MAPS = MAPS[:SLICE, :]
H = W = 64
MAPS = MAPS.reshape(-1, 1, H, W)

r = 50
WOPR = np.load(path + '/WOPR_unc.npy')[:, :r, 0]
WWCT = np.load(path + '/WWCT_unc.npy')[:, :r, 0]
WBHP = np.load(path + '/WBHP_unc.npy')[:, r:, 0]

training_data = np.hstack((WOPR, WWCT, WBHP)) # N x M
training_data = training_data[:SLICE, :]

max_array = training_data.max(axis=0)

normed_data = np.nan_to_num(training_data / max_array)
dataset = torch.utils.data.TensorDataset(torch.FloatTensor(MAPS), torch.FloatTensor(normed_data))

#%%
batch_size = 100
d_dim = normed_data.shape[-1]
lr = 1e-3#2e-4
epochs = 200
in_size = dataset[0][0].size(0)
#
X = int(len(dataset) * 0.8)
Y = int(len(dataset) - X)
Z = 0
split = [X, Y, Z]

# data model
dm = DataModule(dataset, batch_size, split, seed)

# %%
if torch.cuda.is_available():
    precision = 16
    gpu = 1
else:
    precision = 32
    gpu = 0

train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(dataset, split, generator=torch.Generator().manual_seed(seed))

early_stopping = pl.callbacks.EarlyStopping('val_loss', patience=75) #10 for mine
check_point = pl.callbacks.ModelCheckpoint(dirpath=path+'//lightning_logs', monitor='val_loss')

reg_model = LitReg(d_dim, lr)
# summary(d_ae_model, in_size)

CHECK = False

if CHECK:
    trainer = pl.Trainer(fast_dev_run=True)
else:
    trainer = pl.Trainer(max_epochs=epochs, progress_bar_refresh_rate=50, stochastic_weight_avg=True,
                         callbacks=[early_stopping], precision=precision, gpus=gpu)

# trainer.fit(reg_model, datamodule=dm)

# torch.save(reg_model.state_dict(), "reg_model")

# %%
def data_access(folder=None):
    dim = in_size
    if folder == 'train':
        x, y = train_dataset[:]
    elif folder == 'val':
        x, y = valid_dataset[:]
    elif folder == 'test':
        x, y = test_dataset[:]
    else:
        ValueError('Wrong loading')

    return x, y

folder = "val"  # "train", "val", "test"

x, y = data_access(folder)
# %%
reg_model.load_state_dict(torch.load('reg_model'))

reg_model.eval()
with torch.no_grad():
    data_hat = reg_model(x)
 
x, data_in, data_hat = x.numpy(), y.numpy(), data_hat.numpy()

# %%
def plots(x, data_in, data_hat):
    for i in range(9):
        plt.figure(44, figsize=(8, 8))
        plt.subplot(3, 3, i+1)
        plt.imshow(x[i, 0], interpolation='none', cmap='jet')
        plt.suptitle('Maps')
        plt.axis('off')

        # plt.tight_layout()

        plt.figure(2, figsize=(8, 8))
        plt.subplot(3, 3, i+1)
        plt.plot(data_in[i, :r], 'r--')
        plt.plot(data_hat[i, :r], 'b-')
        plt.suptitle('Production rate')
        plt.ylim(data_in[:, :r].min(), data_in[:, :r].max())
        # plt.tight_layout()

        plt.figure(3, figsize=(8, 8))
        plt.subplot(3, 3, i+1)
        plt.plot(data_in[i, r:r*2] * max_array[r:r*2], 'r--')
        plt.plot(data_hat[i, r:r*2] * max_array[r:r*2], 'b-')
        plt.suptitle('Watercut')
        plt.ylim(data_in[:, r:r*2].min(), data_in[:, r:r*2].max())
        # plt.tight_layout()

        plt.figure(4, figsize=(8, 8))
        plt.subplot(3, 3, i+1)
        plt.plot(data_in[i, r*2:], 'r--')
        plt.plot(data_hat[i, r*2:], 'b-')
        plt.suptitle('Injection pressure')
        plt.ylim(data_in[:, r*2:].min(), data_in[:, r*2:].max())
        # plt.tight_layout()

    plt.show()

    plt.close(2); plt.close(3); plt.close(4); plt.close(44)

plots(x, data_in, data_hat)

print(f'RMSE, {folder}: ', mean_squared_error(data_in.flatten(), data_hat.flatten()))
#%%

plt.figure(figsize=(4, 4))
plt.scatter(data_in.flatten(), data_hat.flatten(), alpha=0.5)
plt.plot([0, 1], 'r', alpha=0.5)
plt.xlabel('True')
plt.ylabel('Predictions')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.tight_layout()
plt.show()

# %%
gc.collect()
torch.cuda.empty_cache()

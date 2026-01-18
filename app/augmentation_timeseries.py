from typing import Optional, Dict, Union
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy import signal
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
import os
import warnings
import random

warnings.filterwarnings("ignore")

class AutoAugmentationTimeseries:
    
    def __init__(self, df_or_path, freq='D'):
        self.selected_features: list = []
        self.freq = freq
        self.df_input = self._load_and_standardize(df_or_path) 
        self.df_input = self.df_input.apply(pd.to_numeric, errors="coerce")
        self.stats: Optional[Dict] = None
        self.n_missing_total: Optional[int] = None
        self.df_updated: Optional[pd.DataFrame] = None
        self.model = None
        self.pred_len = 80
        self.timegan_model = None
        self.scaler = None
        self.seq_len = 24
        self.hidden_dim = 24
        self.z_dim = self.df_input.shape[0]
        self.num_layers = 3
        self.epochs = 350
        self.batch_size = 128
        self.alpha_teacher = 0.7
        self.beta1 = 0.9
        self.lr = 0.001
        self.w_gamma = 1
        self.w_es = 0.1
        self.w_e0 = 10
        self.w_g = 100
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.direction=None
        torch.manual_seed(42)
        np.random.seed(42)
        self.df_input.columns = pd.to_datetime(self.df_input.columns)
        self.df_updated = self.df_input
        self.counter = 0
        self.resume = ''
    
    def _load_and_standardize(self, df_or_path) -> pd.DataFrame:
       
        if isinstance(df_or_path, pd.DataFrame):
            return self._standardize_dataframe(df_or_path)

        if isinstance(df_or_path, str):
            if not os.path.exists(df_or_path):
                raise FileNotFoundError(f"Файл не найден: {df_or_path}")

            # пробуем читать CSV с разными разделителями
            df = self._try_load_csv(df_or_path)

            # приводим к стандарту
            return self._standardize_dataframe(df)

        raise TypeError("df_or_path должен быть DataFrame или строкой пути к CSV.")

    def _try_load_csv(self, path: str) -> pd.DataFrame:
       
        for sep in [',', ';', '\t']:
            try:
                df = pd.read_csv(path, sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass

        raise ValueError(f"Не удалось прочитать CSV: {path}")

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        # Формат "factor, date, value" (long)
        if {'factor', 'date', 'value'}.issubset(df.columns):
            df['date'] = pd.to_datetime(df['date'])
            pivoted = df.pivot(index='factor', columns='date', values='value')
            return pivoted.sort_index(axis=1)

        # Проверяем первый столбец
        first_col = df.columns[0]
        if isinstance(first_col, str):
            first_col_lower = first_col.lower()
        else:
            first_col_lower = ""

        # Формат "date, f1, f2, ..." (wide)
        if 'date' in first_col_lower or 'time' in first_col_lower:
            df[df.columns[0]] = pd.to_datetime(df.iloc[:, 0])
            df = df.set_index(df.columns[0]).T  # строки = факторы
            return df

        # Если колонки уже в datetime
        try:
            df.columns = pd.to_datetime(df.columns)
            return df
        except Exception:
            pass

        # Если индекс в datetime
        try:
            df.index = pd.to_datetime(df.index)
            df = df.T
            return df
        except Exception:
            pass

        raise ValueError(
            "Не удалось распознать формат данных. "
            "Поддерживаются форматы:\n"
            "1) factor, date, value (long)\n"
            "2) date, f1, f2, ... (wide)\n"
            "3) FACTORS в index, ДАТЫ в столбцах."
        )

    def fit_timegan(self):
        df = self.df_updated.T
        df.index = pd.to_datetime(df.index)
        numerator = df.values - np.min(df.values, axis=0)
        denominator = np.max(df.values, axis=0) - np.min(df.values, axis=0)
        norm_data = numerator / (denominator + 1e-7)
        temp_data = []
        seq_len = self.seq_len
    
        for i in range(0, len(norm_data) - seq_len):
            sequence = norm_data[i:i + seq_len]
            temp_data.append(sequence)
        
        idx = np.random.permutation(len(temp_data))
        ori_data = []
        for i in range(len(temp_data)):
            ori_data.append(temp_data[idx[i]])
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df.values)
        self.data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(self.device)
        self.data_tensor = self.data_tensor
        self.n_features = df.shape[1]
        z_dim = self.z_dim
        hidden_dim = self.hidden_dim
        num_layer = self.num_layers
        batch_size = self.batch_size
        iteration = self.epochs
        beta1 = self.beta1
        lr = self.lr
        resume = self.resume
        w_gamma= self.w_gamma
        w_es = self.w_es
        w_e0 = self.w_e0
        w_g = self.w_g
        isTrain = True
        device = self.device

        # Создание окон
        def create_sequences(data, seq_len):
            X = []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len])
            return torch.stack(X)
        
        def _weights_init(m):
            classname = m.__class__.__name__
            if isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0)
            elif classname.find('Conv') != -1:
                m.weight.data.normal_(0.0, 0.02)
            elif classname.find('Norm') != -1:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)
            elif classname.find("GRU") != -1:
                for name,param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)

        def batch_generator(data, time, batch_size):
            no = len(data)
            idx = np.random.permutation(no)
            train_idx = idx[:batch_size]
            X_mb = list(data[i] for i in train_idx)
            T_mb = list(time[i] for i in train_idx)
            return X_mb, T_mb

        def extract_time (data):
            time = list()
            max_seq_len = 0
            for i in range(len(data)):
                max_seq_len = max(max_seq_len, len(data[i][:,0]))
                time.append(len(data[i][:,0]))
                
            return time, max_seq_len

        def random_generator (batch_size, z_dim, T_mb, max_seq_len):
            Z_mb = list()
            for i in range(batch_size):
                temp = np.zeros([max_seq_len, z_dim])
                temp_Z = np.random.uniform(0., 1, [T_mb[i], z_dim])
                temp[:T_mb[i],:] = temp_Z
                Z_mb.append(temp_Z)

            return Z_mb

        def NormMinMax(data):
            min_val = np.min(np.min(data, axis=0), axis=0)
            data = data - min_val
            max_val = np.max(np.max(data, axis=0), axis=0)
            norm_data = data / (max_val + 1e-7)
            return norm_data, min_val, max_val

        class Encoder(nn.Module):
            def __init__(self):
                super(Encoder, self).__init__()
                self.rnn = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layer)
              # self.norm = nn.BatchNorm1d(hidden_dim)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
                self.sigmoid = nn.Sigmoid()
                self.apply(_weights_init)

            def forward(self, input, sigmoid=True):
                e_outputs, _ = self.rnn(input)
                H = self.fc(e_outputs)
                if sigmoid:
                    H = self.sigmoid(H)
                return H

        class Recovery(nn.Module):
            def __init__(self):
                super(Recovery, self).__init__()
                self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=z_dim, num_layers=num_layer)
             #  self.norm = nn.BatchNorm1d(z_dim)
                self.fc = nn.Linear(z_dim, z_dim)
                self.sigmoid = nn.Sigmoid()
                self.apply(_weights_init)

            def forward(self, input, sigmoid=True):
                r_outputs, _ = self.rnn(input)
                X_tilde = self.fc(r_outputs)
                if sigmoid:
                    X_tilde = self.sigmoid(X_tilde)
                return X_tilde
        
        class Supervisor(nn.Module):
            def __init__(self):
                super(Supervisor, self).__init__()
                self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layer)
            #   self.norm = nn.LayerNorm(hidden_dim)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
                self.sigmoid = nn.Sigmoid()
                self.apply(_weights_init)

            def forward(self, input, sigmoid=True):
                s_outputs, _ = self.rnn(input)
            #   s_outputs = self.norm(s_outputs)
                S = self.fc(s_outputs)
                if sigmoid:
                    S = self.sigmoid(S)
                return S

        class Generator(nn.Module):
            def __init__(self):
                super(Generator, self).__init__()
                self.rnn = nn.GRU(input_size=z_dim, hidden_size=hidden_dim, num_layers=num_layer)
            #   self.norm = nn.LayerNorm(hidden_dim)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
                self.sigmoid = nn.Sigmoid()
                self.apply(_weights_init)

            def forward(self, input, sigmoid=True):
                g_outputs, _ = self.rnn(input)
            #   g_outputs = self.norm(g_outputs)
                E = self.fc(g_outputs)
                if sigmoid:
                    E = self.sigmoid(E)
                return E
        
        class Discriminator(nn.Module):
            def __init__(self):
                super(Discriminator, self).__init__()
                self.rnn = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layer)
            #   self.norm = nn.LayerNorm(hidden_dim)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
                self.sigmoid = nn.Sigmoid()
                self.apply(_weights_init)

            def forward(self, input, sigmoid=True):
                d_outputs, _ = self.rnn(input)
                Y_hat = self.fc(d_outputs)
                if sigmoid:
                    Y_hat = self.sigmoid(Y_hat)
                return Y_hat
        
        class BaseModel():
            def __init__(self, ori_data):
                self.seed(-1)
                self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
                self.ori_time, self.max_seq_len = extract_time(ori_data)
                self.data_num, _, _ = np.asarray(ori_data).shape  
                self.trn_dir = os.path.join('./output', 'experiment_name', 'train')
                self.tst_dir = os.path.join('./output' 'experiment_name', 'test')
                self.device = torch.device("cpu")

            def seed(self, seed_value):
                if seed_value == -1:
                    return

                random.seed(seed_value)
                torch.manual_seed(seed_value)
                torch.cuda.manual_seed_all(seed_value)
                np.random.seed(seed_value)
                torch.backends.cudnn.deterministic = True

            def save_weights(self, epoch):
                weight_dir = os.path.join('./output', 'experiment_name', 'train', 'weights')
                if not os.path.exists(weight_dir): 
                    os.makedirs(weight_dir)

                torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
                            '%s/netE.pth' % (weight_dir))
                torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
                            '%s/netR.pth' % (weight_dir))
                torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                            '%s/netG.pth' % (weight_dir))
                torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                            '%s/netD.pth' % (weight_dir))
                torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
                            '%s/netS.pth' % (weight_dir))

            def train_one_iter_er(self):
                self.nete.train()
                self.netr.train()
                # set mini-batch
                self.X0, self.T = batch_generator(self.ori_data, self.ori_time, batch_size)
                self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
                # train encoder & decoder
                self.optimize_params_er()

            def train_one_iter_er_(self):
                self.nete.train()
                self.netr.train()
                # set mini-batch
                self.X0, self.T = batch_generator(self.ori_data, self.ori_time, batch_size)
                self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
                # train encoder & decoder
                self.optimize_params_er_()
            
            def train_one_iter_s(self):
                #self.nete.eval()
                self.nets.train()
                # set mini-batch
                self.X0, self.T = batch_generator(self.ori_data, self.ori_time, batch_size)
                self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
                # train superviser
                self.optimize_params_s()

            def train_one_iter_g(self):
                self.netg.train()
                # set mini-batch
                self.X0, self.T = batch_generator(self.ori_data, self.ori_time, batch_size)
                self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
                self.Z = random_generator(batch_size, z_dim, self.T, self.max_seq_len)
                # train superviser
                self.optimize_params_g()

            def train_one_iter_d(self):
                self.netd.train()
                # set mini-batch
                self.X0, self.T = batch_generator(self.ori_data, self.ori_time, batch_size)
                self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
                self.Z = random_generator(batch_size, z_dim, self.T, self.max_seq_len)
                # train superviser
                self.optimize_params_d()

            def train(self):
                for iter in range(iteration):
                # Train for one iter
                    self.train_one_iter_er()
                    print('Encoder training step: '+ str(iter) + '/' + str(iteration))

                for iter in range(iteration):
                # Train for one iter
                    self.train_one_iter_s()
                    print('Superviser training step: '+ str(iter) + '/' + str(iteration))

                for iter in range(iteration):
                # Train for one iter
                    for kk in range(2):
                        self.train_one_iter_g()
                        self.train_one_iter_er_()

                    self.train_one_iter_d()
                    print('Superviser training step: '+ str(iter) + '/' + str(iteration))

                self.save_weights(iteration)
                self.generated_data = self.generation(batch_size)
                print('Finish Synthetic Data Generation')

            def generation(self, num_samples, mean = 0.0, std = 1.0):
                if num_samples == 0:
                    return None, None
                ## Synthetic data generation
                self.X0, self.T = batch_generator(self.ori_data, self.ori_time, batch_size)
                self.Z = random_generator(num_samples, z_dim, self.T, self.max_seq_len)
                self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
                self.E_hat = self.netg(self.Z)    
                self.H_hat = self.nets(self.E_hat)  
                generated_data_curr = self.netr(self.H_hat).cpu().detach().numpy()  
                generated_data = list()
                for i in range(num_samples):
                    temp = generated_data_curr[i, :self.ori_time[i], :]
                    generated_data.append(temp)
                
                # Renormalization
                generated_data = generated_data * self.max_val
                generated_data = generated_data + self.min_val
                return generated_data
            
        class TimeGAN(BaseModel):
            @property
            def name(self):
                return 'TimeGAN'

            def __init__(self, ori_data):
                super(TimeGAN, self).__init__(ori_data)
                # -- Misc attributes
                self.epoch = 0
                self.times = []
                self.total_steps = 0
                # Create and initialize networks.
                self.nete = Encoder().to(device)
                self.netr = Recovery().to(device)
                self.netg = Generator().to(device)
                self.netd = Discriminator().to(device)
                self.nets = Supervisor().to(device)

                if resume != '':
                    print("\nLoading pre-trained networks.")
                    self.iter = torch.load(os.path.join(resume, 'netG.pth'))['epoch']
                    self.nete.load_state_dict(torch.load(os.path.join(resume, 'netE.pth'))['state_dict'])
                    self.netr.load_state_dict(torch.load(os.path.join(resume, 'netR.pth'))['state_dict'])
                    self.netg.load_state_dict(torch.load(os.path.join(resume, 'netG.pth'))['state_dict'])
                    self.netd.load_state_dict(torch.load(os.path.join(resume, 'netD.pth'))['state_dict'])
                    self.nets.load_state_dict(torch.load(os.path.join(resume, 'netS.pth'))['state_dict'])
                    print("\tDone.\n")

               # loss
                self.l_mse = nn.MSELoss()
                self.l_r = nn.L1Loss()
                self.l_bce = nn.BCELoss()

                # Setup optimizer
                if isTrain:
                    self.nete.train()
                    self.netr.train()
                    self.netg.train()
                    self.netd.train()
                    self.nets.train()
                    self.optimizer_e = optim.Adam(self.nete.parameters(), lr=lr, betas=(beta1, 0.999))
                    self.optimizer_r = optim.Adam(self.netr.parameters(), lr=lr, betas=(beta1, 0.999))
                    self.optimizer_g = optim.Adam(self.netg.parameters(), lr=lr, betas=(beta1, 0.999))
                    self.optimizer_d = optim.Adam(self.netd.parameters(), lr=lr, betas=(beta1, 0.999))
                    self.optimizer_s = optim.Adam(self.nets.parameters(), lr=lr, betas=(beta1, 0.999))

            def forward_e(self):
                self.H = self.nete(self.X)

            def forward_er(self):
                self.H = self.nete(self.X)
                self.X_tilde = self.netr(self.H)

            def forward_g(self):
                self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
                self.E_hat = self.netg(self.Z)

            def forward_dg(self):
                self.Y_fake = self.netd(self.H_hat)
                self.Y_fake_e = self.netd(self.E_hat)

            def forward_rg(self):
                self.X_hat = self.netr(self.H_hat)

            def forward_s(self):
                self.H_supervise = self.nets(self.H)

            def forward_sg(self):
                self.H_hat = self.nets(self.E_hat)

            def forward_d(self):
              
                self.Y_real = self.netd(self.H)
                self.Y_fake = self.netd(self.H_hat)
                self.Y_fake_e = self.netd(self.E_hat)

            def backward_er(self):
                self.err_er = self.l_mse(self.X_tilde, self.X)
                self.err_er.backward(retain_graph=True)
                print("Loss: ", self.err_er)

            def backward_er_(self):
                self.err_er_ = self.l_mse(self.X_tilde, self.X) 
                self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
                self.err_er = 10 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
                self.err_er.backward(retain_graph=True)

            def backward_g(self):
                self.err_g_U = self.l_bce(self.Y_fake, torch.ones_like(self.Y_fake))
                self.err_g_U_e = self.l_bce(self.Y_fake_e, torch.ones_like(self.Y_fake_e))
                self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(torch.std(self.X_hat,[0])[1] + 1e-6) - torch.sqrt(torch.std(self.X,[0])[1] + 1e-6)))   # |a^2 - b^2|
                self.err_g_V2 = torch.mean(torch.abs((torch.mean(self.X_hat,[0])[0]) - (torch.mean(self.X,[0])[0])))  # |a - b|
                self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
                self.err_g = self.err_g_U + \
                            self.err_g_U_e * w_gamma + \
                            self.err_g_V1 * w_g + \
                            self.err_g_V2 * w_g + \
                            torch.sqrt(self.err_s) 
                self.err_g.backward(retain_graph=True)
                print("Loss G: ", self.err_g)

            def backward_s(self):
                self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
                self.err_s.backward(retain_graph=True)
                print("Loss S: ", self.err_s)

            def backward_d(self):
                self.err_d_real = self.l_bce(self.Y_real, torch.ones_like(self.Y_real))
                self.err_d_fake = self.l_bce(self.Y_fake, torch.zeros_like(self.Y_fake))
                self.err_d_fake_e = self.l_bce(self.Y_fake_e, torch.zeros_like(self.Y_fake_e))
                self.err_d = self.err_d_real + \
                            self.err_d_fake + \
                            self.err_d_fake_e * w_gamma
                if self.err_d > 0.15:
                    self.err_d.backward(retain_graph=True)

            def optimize_params_er(self):
                # Forward-pass
                self.forward_er()
                # Backward-pass
                # nete & netr
                self.optimizer_e.zero_grad()
                self.optimizer_r.zero_grad()
                self.backward_er()
                self.optimizer_e.step()
                self.optimizer_r.step()

            def optimize_params_er_(self):
                # Forward-pass
                self.forward_er()
                self.forward_s()
                # Backward-pass
                # nete & netr
                self.optimizer_e.zero_grad()
                self.optimizer_r.zero_grad()
                self.backward_er_()
                self.optimizer_e.step()
                self.optimizer_r.step()

            def optimize_params_s(self):
                # Forward-pass
                self.forward_e()
                self.forward_s()
                # Backward-pass
                # nets
                self.optimizer_s.zero_grad()
                self.backward_s()
                self.optimizer_s.step()

            def optimize_params_g(self):
                # Forward-pass
                self.forward_e()
                self.forward_s()
                self.forward_g()
                self.forward_sg()
                self.forward_rg()
                self.forward_dg()

                # Backward-pass
                # nets
                self.optimizer_g.zero_grad()
                self.optimizer_s.zero_grad()
                self.backward_g()
                self.optimizer_g.step()
                self.optimizer_s.step()

            def optimize_params_d(self):
                # Forward-pass
                self.forward_e()
                self.forward_g()
                self.forward_sg()
                self.forward_d()
                self.forward_dg()

                # Backward-pass
                # nets
                self.optimizer_d.zero_grad()
                self.backward_d()
                self.optimizer_d.step()
        
        self.model = TimeGAN(ori_data)
        self.model.train()   
    #    self.model = (embedder, recovery, generator, supervisor, discriminator)
    
    def _extend_datetime_index(self):
        if not isinstance(self.df_updated.T.index, pd.DatetimeIndex) or len(self.df_updated.T.index) < 2:
            raise ValueError("Индекс должен быть DatetimeIndex с >=2 элементов")

        df_extended = self.df_updated.T.copy()
        step = self.df_updated.T.index[1] - self.df_updated.T.index[0]
        
        # Используем self.direction для определения количества шагов
        direction = getattr(self, 'direction', 'forward')
        pred_len = getattr(self, 'pred_len', 10)  # значение по умолчанию
        
        # Forward индексы
        forward_idx = []
        if direction in ["forward", "both"]:
            last_idx = df_extended.index[-1]
            n_forward = pred_len if direction == "forward" else pred_len // 2 + pred_len % 2
            forward_idx = [last_idx + step*(i+1) for i in range(int(n_forward))]
        
        # Backward индексы
        backward_idx = []
        if direction in ["backward", "both"]:
            first_idx = df_extended.index[0]
            n_backward = pred_len if direction == "backward" else pred_len // 2
            backward_idx = [first_idx - step*(i+1) for i in reversed(range(int(n_backward)))]

        # Расширяем df, чтобы можно было вставить прогнозы
        for idx in backward_idx + forward_idx:
            if idx not in df_extended.index:
                df_extended.loc[idx] = np.nan

        return df_extended, forward_idx, backward_idx

    def predict_timegan(self, **kwargs):
        if self.model is None:
            raise RuntimeError("Модель TimeGAN не обучена. Сначала вызови fit_timegan()")

        if not hasattr(self, 'scaler') or self.scaler is None:
            raise RuntimeError("Скалер не инициализирован. Проверьте fit_timegan()")

        # Получаем направление прогноза
        direction = kwargs.get('direction', getattr(self, 'direction', 'forward'))
        
        model = self.model
        
        # Переводим все подсети в eval режим
        model.nete.eval()
        model.netr.eval()
        model.netg.eval()
        model.nets.eval()
        
        # Получаем расширенные индексы
        df_extended, forward_idx, backward_idx = self._extend_datetime_index()
        n_features = self.n_features
        
        with torch.no_grad():
            # Получаем данные из модели
            real_data = self.model.ori_data  # это может быть list или numpy array
            
            # Проверяем, что данные есть (правильная проверка для разных типов)
            if real_data is None:
                raise ValueError("Нет данных для прогнозирования: ori_data is None")
            
            # Преобразуем в список, если это numpy array
            if isinstance(real_data, np.ndarray):
                # Если real_data 3D array (num_samples, seq_len, n_features)
                real_data = [real_data[i] for i in range(len(real_data))]
            elif not isinstance(real_data, list):
                raise TypeError(f"Неожиданный тип данных: {type(real_data)}")
            
            # Проверяем, что список не пустой
            if len(real_data) == 0:
                raise ValueError("Нет данных для прогнозирования: список последовательностей пуст")
            
            # Берем последнюю последовательность как контекст
            context_seq = real_data[-1]  # shape: (seq_len, n_features)
            
            # Подготавливаем для каждого направления
            backward_arr = []
            forward_arr = []
            
            # Прогноз назад (если нужно)
            if direction in ["backward", "both"] and backward_idx:
                context_tensor = torch.tensor(context_seq, dtype=torch.float32).to(self.device).unsqueeze(0)
                
                # Для backward прогноза
                H_context = model.nete(context_tensor)
                next_hidden = H_context[:, :1, :]  # первое скрытое состояние
                
                for _ in range(len(backward_idx)):
                    z = torch.randn(1, 1, self.z_dim, device=self.device)
                    h_next = model.netg(z)
                    next_hidden = self.alpha_teacher * next_hidden + (1 - self.alpha_teacher) * h_next
                    x_next = model.netr(next_hidden)
                    backward_arr.append(x_next.squeeze().cpu().numpy())
                
                # Переворачиваем для правильного порядка времени
                if backward_arr:
                    backward_arr = np.stack(backward_arr[::-1], axis=0)
                else:
                    backward_arr = np.empty((0, n_features))
            
            # Прогноз вперед (если нужно)
            if direction in ["forward", "both"] and forward_idx:
                context_tensor = torch.tensor(context_seq, dtype=torch.float32).to(self.device).unsqueeze(0)
                
                # Для forward прогноза
                H_context = model.nete(context_tensor)
                next_hidden = H_context[:, -1:, :]  # последнее скрытое состояние
                
                for _ in range(len(forward_idx)):
                    z = torch.randn(1, 1, self.z_dim, device=self.device)
                    h_next = model.netg(z)
                    next_hidden = self.alpha_teacher * next_hidden + (1 - self.alpha_teacher) * h_next
                    x_next = model.netr(next_hidden)
                    forward_arr.append(x_next.squeeze().cpu().numpy())
                
                if forward_arr:
                    forward_arr = np.stack(forward_arr, axis=0)
                else:
                    forward_arr = np.empty((0, n_features))
            
            # Объединяем прогнозы в правильном порядке
            if direction == "both":
                # backward + forward
                generated = np.vstack([backward_arr, forward_arr])
                all_idx = backward_idx + forward_idx
            elif direction == "backward":
                generated = backward_arr
                all_idx = backward_idx
            else:  # forward
                generated = forward_arr
                all_idx = forward_idx
            
            # Проверяем, что есть данные для денормализации
            if len(generated) > 0:
                generated_original = self.scaler.inverse_transform(generated)
            else:
                generated_original = np.empty((0, n_features))
        
        # Обновляем DataFrame с прогнозами
        for i, date in enumerate(all_idx):
            if date in df_extended.index and i < len(generated_original):
                df_extended.loc[date] = generated_original[i]
        
        # Создаем DataFrame с прогнозами для возврата
        if len(generated_original) > 0:
            extrapolated_df = pd.DataFrame(
                generated_original, 
                index=all_idx,
                columns=[f'feature_{i}' for i in range(n_features)]
            ).T
        else:
            extrapolated_df = pd.DataFrame(columns=all_idx)
        
        # Если у нас есть названия признаков из оригинального DataFrame
        if hasattr(self, 'df_updated') and self.df_updated is not None:
            # Берем имена строк из оригинального DataFrame
            feature_names = self.df_updated.index.tolist()
            if len(feature_names) == n_features and len(extrapolated_df) > 0:
                extrapolated_df.index = feature_names
        
        # Фильтрация по selected_features
        if hasattr(self, 'selected_features') and self.selected_features and len(extrapolated_df) > 0:
            # Убедимся, что selected_features соответствуют индексам
            valid_features = [f for f in self.selected_features if f in extrapolated_df.index]
            if valid_features:
                extrapolated_df = extrapolated_df.loc[valid_features]
            else:
                # Если нет совпадений, используем числовые индексы
                try:
                    valid_features = [f for f in self.selected_features if int(f) < n_features]
                    if valid_features:
                        extrapolated_df = extrapolated_df.iloc[valid_features]
                except (ValueError, TypeError):
                    pass
        
        # Сохраняем индексы
        self.forward_idx = forward_idx
        self.backward_idx = backward_idx
        
        return extrapolated_df
   
    @staticmethod
    def _check_stationarity(series: pd.Series) -> Dict[str, Optional[Union[str, float]]]:
        series_clean = series.dropna()
        if len(series_clean) < 5:
            return {"Категория стационарности": "Ошибка", "Стационарность": "Недостаточно данных",
                    "ADF p-value": None, "ADF Statistic": None}
        if series_clean.nunique() == 1:
            return {"Категория стационарности": "Стационарный", "Стационарность": "Постоянные значения",
                    "ADF p-value": 0.0, "ADF Statistic": float('inf')}
        try:
            adf_stat, pvalue, *_ = adfuller(series_clean)
            return {
                "Категория стационарности": "Стационарный" if pvalue <= 0.05 else "Нестационарный",
                "Стационарность": f"p-value={round(pvalue,4)}",
                "ADF p-value": round(pvalue, 4),
                "ADF Statistic": round(adf_stat, 4)
            }
        except Exception as e:
            return {"Категория стационарности": "Ошибка",
                    "Стационарность": f"Ошибка: {e}",
                    "ADF p-value": None, "ADF Statistic": None}

    def calculate_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        if df is None:
            df = self.df_input
        stats_all = {}
        total_missing = 0
        for idx, row in df.iterrows():
            series = pd.Series(row)
            stat_station = self._check_stationarity(series)
            count_missing = int(series.isna().sum())
            s_clean = series.dropna()
            diff1 = np.diff(s_clean) if len(s_clean) > 1 else np.array([0.0])
            diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0.0])
            stats_all[idx] = {
                "Длина ряда": int(len(series)),
                "Количество пустных значений": count_missing,
                "Доля пустных значений, %": round(count_missing / len(series) * 100, 2),
                "Среднее": round(series.mean(skipna=True), 4) if not series.dropna().empty else None,
                "Медиана": round(series.median(skipna=True), 4) if not series.dropna().empty else None,
                "Мин": round(series.min(skipna=True), 4) if not series.dropna().empty else None,
                "Макс": round(series.max(skipna=True), 4) if not series.dropna().empty else None,
                "Ст. откл.": round(series.std(skipna=True), 4) if not series.dropna().empty else None,
                "Категория стационарности": stat_station["Категория стационарности"],
                "ADF p-value": stat_station["ADF p-value"],
                "ADF Statistic": stat_station["ADF Statistic"],
                "Локальная вариативность": float(np.mean(diff1**2)) if len(diff1) > 0 else 0.0,
                "Локальная кривизна": float(np.mean(diff2**2)) if len(diff2) > 0 else 0.0
            }
            total_missing += count_missing
        self.stats = stats_all
        self.n_missing_total = total_missing
        return stats_all

    def _rowwise_interpolate(self, df: pd.DataFrame, method: str, order: int = None) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        df_filled = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        
        for idx, row in df_num.iterrows():
            s = row.copy()
            mask_notna = s.notna()
            
            if mask_notna.sum() < 2:
                df_filled.loc[idx] = s.values
                continue
                
            try:
                # Интерполируем только пропуски, сохраняя известные значения
                if method in ["polynomial", "spline"]:
                    if mask_notna.sum() >= (order if order else 2):
                        # interpolate() НЕ меняет не-NaN значения
                        filled_row = s.interpolate(method=method, order=order, limit_area='inside')
                    else:
                        filled_row = s.interpolate(method='linear', limit_area='inside')
                else:
                    filled_row = s.interpolate(method=method, limit_area='inside')
                
                df_filled.loc[idx] = filled_row.values
            except Exception:
                try:
                    filled_row = s.interpolate(method='linear', limit_area='inside')
                    df_filled.loc[idx] = filled_row.values
                except Exception:
                    df_filled.loc[idx] = s.values
        
        df_filled.columns = df.columns
        return df_filled

    def interpolate_linear(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._rowwise_interpolate(df, method="linear")

    def interpolate_nearest(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._rowwise_interpolate(df, method="nearest")

    def interpolate_quadratic(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._rowwise_interpolate(df, method="polynomial", order=2)

    def interpolate_cubic(self, df: pd.DataFrame) -> pd.DataFrame:
        return self._rowwise_interpolate(df, method="polynomial", order=3)

    def interpolate_spline2(self, df: pd.DataFrame, order: int = 2) -> pd.DataFrame:
        return self._rowwise_interpolate(df, method="spline", order=order)

    def interpolate_spline3(self, df: pd.DataFrame, order: int = 3) -> pd.DataFrame:
        return self._rowwise_interpolate(df, method="spline", order=order)

    def interpolate_pchip(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        ncols = df_num.shape[1]
        idx_num = np.arange(ncols)
        filled = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)

        for r_idx, row in df_num.iterrows():
            s = row.values.copy()
            mask_notna = ~np.isnan(s)
            
            if mask_notna.sum() < 2:
                filled.loc[r_idx] = s
                continue

            # Сохраняем исходные значения
            original_values = s[mask_notna]
            original_indices = np.where(mask_notna)[0]
            
            x_known = idx_num[mask_notna]
            y_known = s[mask_notna]

            # Интерполируем только между известными точками
            first_known_idx = x_known[0]
            last_known_idx = x_known[-1]
            
            # Создаем PCHIP интерполятор
            try:
                interpolator = PchipInterpolator(x_known, y_known)
                
                # Интерполируем только точки между известными
                for idx in range(first_known_idx + 1, last_known_idx):
                    if np.isnan(s[idx]):  # Только NaN точки между известными
                        s[idx] = interpolator(idx)
                
                # Восстанавливаем исходные значения
                for idx, val in zip(original_indices, original_values):
                    s[idx] = val
                
                filled.loc[r_idx] = s
            except Exception:
                # Если PCHIP не сработал, оставляем как есть
                filled.loc[r_idx] = s

        filled.columns = df.columns
        return filled

    def interpolate_akima(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        ncols = df_num.shape[1]
        idx_num = np.arange(ncols)
        filled = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        
        for r_idx, row in df_num.iterrows():
            s = row.values
            mask_notna = ~np.isnan(s)
            
            if mask_notna.sum() < 2:
                filled.loc[r_idx] = s
                continue
            
            # Сохраняем исходные значения
            original_values = s[mask_notna]
            original_indices = np.where(mask_notna)[0]
            
            x_known = idx_num[mask_notna]
            y_known = s[mask_notna]
            
            try:
                interpolator = Akima1DInterpolator(x_known, y_known)
                
                # Интерполируем только точки между известными
                first_known_idx = x_known[0]
                last_known_idx = x_known[-1]
                
                for idx in range(first_known_idx + 1, last_known_idx):
                    if np.isnan(s[idx]):
                        s[idx] = interpolator(idx)
                
                # Восстанавливаем исходные значения
                for idx, val in zip(original_indices, original_values):
                    s[idx] = val
                
                filled.loc[r_idx] = s
            except Exception:
                # Если Akima не сработал, оставляем как есть
                filled.loc[r_idx] = s
        
        filled.columns = df.columns
        return filled

    def interpolate_auto(self, df: pd.DataFrame) -> pd.DataFrame:
        methods = [
            self.interpolate_linear,
            self.interpolate_quadratic,
            self.interpolate_cubic,
            self.interpolate_nearest,
            lambda d: self.interpolate_spline2(d, order=2),
            lambda d: self.interpolate_spline3(d, order=3),
            self.interpolate_pchip,
            self.interpolate_akima
        ]

        method_names = [
            "linear", "quadratic", "cubic", "nearest",
            "spline2", "spline3", "pchip", "akima"
        ]

        df_num = df.copy()
        df_result = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        self.best_methods_interpolation = {}
        
        for idx in df_num.index:
            row = df_num.loc[idx].copy()
            mask_notna = row.notna()
            
            # Если нет пропусков - возвращаем как есть
            if not row.isna().any():
                df_result.loc[idx] = row.values
                self.best_methods_interpolation[idx] = "none"
                continue
            
            # Проверяем достаточно ли точек
            if mask_notna.sum() < 2:
                # Меньше 2 точек - простое заполнение
                s_filled = row.ffill().bfill()
                df_result.loc[idx] = s_filled.values
                self.best_methods_interpolation[idx] = "ffill_bfill"
                continue
            
            # Находим реальные NaN позиции (которые нужно интерполировать)
            real_nan_mask = row.isna()
            
            # Для валидации выбираем только часть известных точек
            valid_indices = np.where(mask_notna)[0]
            valid_values = row[valid_indices].values
            
            if len(valid_indices) < 4:
                # Мало данных для валидации - используем linear
                try:
                    interpolated = self.interpolate_linear(pd.DataFrame([row]))
                    df_result.loc[idx] = interpolated.iloc[0].values
                    self.best_methods_interpolation[idx] = "linear"
                except:
                    df_result.loc[idx] = row.values
                    self.best_methods_interpolation[idx] = "none"
                continue
            
            # Выбираем тестовые точки из не Nan
            # Исключаем первую и последнюю известную точку
            if len(valid_indices) > 3:
                candidate_test_indices = valid_indices[1:-1]  # без первой и последней
                candidate_test_values = valid_values[1:-1]
            else:
                candidate_test_indices = valid_indices
                candidate_test_values = valid_values
            
            # Выбираем 30% для тестирования (минимум 2)
            n_test = max(2, int(len(candidate_test_indices) * 0.3))
            if n_test < len(candidate_test_indices):
                test_idx_positions = np.random.choice(
                    range(len(candidate_test_indices)), 
                    size=n_test, 
                    replace=False
                )
                test_positions = candidate_test_indices[test_idx_positions]
                true_test_values = candidate_test_values[test_idx_positions]
            else:
                test_positions = candidate_test_indices
                true_test_values = candidate_test_values
            
            # Создаем тестовый ряд
            test_row = row.copy()
            test_row.iloc[test_positions] = np.nan  # скрываем выбранные точки
            
            best_score = float('inf')
            best_method = "linear"
            
            # Тестируем методы на искусственных пропусках
            for method_func, method_name in zip(methods, method_names):
                try:
                    # Проверяем требования метода
                    available_points = mask_notna.sum() - len(test_positions)
                    
                    if method_name in ['quadratic', 'cubic', 'spline2', 'spline3']:
                        order_needed = 3 if method_name in ['cubic', 'spline3'] else 2
                        if available_points < order_needed + 1:
                            continue
                    
                    # Применяем метод
                    test_df = pd.DataFrame([test_row])
                    interpolated_test = method_func(test_df)
                    
                    if interpolated_test is None or interpolated_test.empty:
                        continue
                    
                    # Извлекаем предсказания для скрытых точек
                    pred_values = []
                    valid_predictions = True
                    
                    for pos in test_positions:
                        pred = interpolated_test.iloc[0, pos]
                        if pd.isna(pred):
                            valid_predictions = False
                            break
                        pred_values.append(pred)
                    
                    if not valid_predictions or len(pred_values) != len(true_test_values):
                        continue
                    
                    # Вычисляем метрики
                    mae = np.mean(np.abs(np.array(pred_values) - np.array(true_test_values)))
                    data_range = np.max(true_test_values) - np.min(true_test_values)
                    
                    if data_range > 0:
                        mae_normalized = mae / data_range
                    else:
                        mae_normalized = mae
                    
                    if len(pred_values) > 1 and len(true_test_values) > 1:
                        pred_diffs = np.diff(pred_values)
                        true_diffs = np.diff(true_test_values)
                        sign_match = np.mean(np.sign(pred_diffs) == np.sign(true_diffs))
                    else:
                        sign_match = 1.0
                    
                    score = mae_normalized + (1 - sign_match) * 0.5
                    
                    # Дополнительные проверки
                    if method_name == 'nearest' and len(pred_values) > 1 and len(true_test_values) > 1:
                        pred_var = np.var(pred_values)
                        true_var = np.var(true_test_values)
                        if true_var > 0 and pred_var > true_var * 2:
                            score *= 1.2
                    
                    if score < best_score:
                        best_score = score
                        best_method = method_name
                        
                except Exception:
                    continue
            
            try:
                row_to_interpolate = row.copy()
                
                row_df = pd.DataFrame([row_to_interpolate])
                
                if best_method == "linear":
                    interpolated_row = self.interpolate_linear(row_df)
                elif best_method == "quadratic":
                    interpolated_row = self.interpolate_quadratic(row_df)
                elif best_method == "cubic":
                    interpolated_row = self.interpolate_cubic(row_df)
                elif best_method == "nearest":
                    interpolated_row = self.interpolate_nearest(row_df)
                elif best_method == "spline2":
                    interpolated_row = self.interpolate_spline2(row_df, order=2)
                elif best_method == "spline3":
                    interpolated_row = self.interpolate_spline3(row_df, order=3)
                elif best_method == "pchip":
                    interpolated_row = self.interpolate_pchip(row_df)
                else:
                    interpolated_row = self.interpolate_akima(row_df)
                 
                result_values = interpolated_row.iloc[0].values.copy()
                
                for i in valid_indices:
                    result_values[i] = row.iloc[i]
                
                df_result.loc[idx] = result_values
                self.best_methods_interpolation[idx] = best_method
                
            except Exception:

                fallback_success = False
                for fallback_method in ['linear', 'pchip', 'nearest']:
                    try:
                        row_df = pd.DataFrame([row])
                        if fallback_method == 'linear':
                            interpolated = self.interpolate_linear(row_df)
                        elif fallback_method == 'pchip':
                            interpolated = self.interpolate_pchip(row_df)
                        else:
                            interpolated = self.interpolate_nearest(row_df)
                        
                        if interpolated is not None:

                            result_values = interpolated.iloc[0].values.copy()
                            for i in valid_indices:
                                result_values[i] = row.iloc[i]
                            
                            df_result.loc[idx] = result_values
                            self.best_methods_interpolation[idx] = fallback_method
                            fallback_success = True
                            break
                    except:
                        continue
                
                if not fallback_success:
                    df_result.loc[idx] = row.values
                    self.best_methods_interpolation[idx] = "none"
        
        self.df_updated = df_result.copy()
        return df_result

    def _rowwise_apply_noise(self, df: pd.DataFrame, fn_row) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        out = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        for idx, row in df_num.iterrows():
            try:
                s = pd.Series(row.values, index=df_num.columns, dtype=float)
                mask = ~s.isna()  # добавляем шум только к существующим значениям
                s_noised = s.copy()
                s_noised[mask] = fn_row(s[mask]).values
                out.loc[idx] = s_noised.values
            except Exception: 
                out.loc[idx] = row.values
        out.columns = df.columns
        return out

    def _noise_jitter_gauss_row(self, s: pd.Series, noise_level: Optional[float] = None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        std = s.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0
        scale = (noise_level if noise_level is not None else 0.05) * std
        noise = rng.normal(0, scale, size=len(s))
        res = s.copy()
        res = res + noise
        return res

    def _noise_jitter_white_row(self, s: pd.Series, noise_level: Optional[float] = None, rng=None):

        if rng is None:
            rng = np.random.default_rng()
        std = s.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0
        a = (noise_level if noise_level is not None else 0.03) * std
        noise = rng.uniform(-a, a, size=len(s))
        res = s.copy()
        res = res + noise
        return res

    def _noise_multiplicative_row(self, s: pd.Series, noise_level: Optional[float] = None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        std = s.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0
        scale = (noise_level if noise_level is not None else 0.02)
        noise = 1.0 + rng.normal(0, scale, size=len(s))
        res = s.copy()
        res = s * noise
        return res

    def _noise_outliers_row(self, s: pd.Series, outlier_fraction: float = 0.01, outlier_magnitude: float = 5.0, rng=None):
        res = s.copy()
        
        if len(res) == 0 or res.isna().all():
            return res
        
        std = np.nanstd(res)
        if std == 0 or np.isnan(std):
            return 1e-8  # чтобы не было деления на 0 и нулевой амплитуды
        
        # Количество выбросов
        n_outliers = max(1, int(len(res) * getattr(self, "outlier_fraction", 0.2)))
        idx = np.random.choice(np.arange(len(res)), n_outliers, replace=False)
        
        # Амплитуда выброса
        magnitude = getattr(self, "outlier_magnitude", 5) * std
        noise = np.random.choice([-1, 1], n_outliers) * magnitude
        res.iloc[idx] += noise
        return res

    def _noise_markovian_row(self, s: pd.Series, transition_prob: float = 0.5, noise_std: float = 0.1, rng=None) -> pd.Series:

        result = s.copy()
        if rng is None:
            rng = np.random.default_rng()
        
        if len(result) == 0 or result.isna().all():
            return result
        
        data = result.values
        noised_data = np.zeros_like(data)
        
        # Первое значение сохраняем или добавляем шум
        if rng.random() < transition_prob:
            noise = rng.normal(scale=noise_std)
            noised_data[0] = data[0] + noise
        else:
            noised_data[0] = data[0]
        
        # Обрабатываем остальные значения
        for i in range(1, len(data)):
            if rng.random() < transition_prob:
                noise = rng.normal(scale=noise_std)
                noised_data[i] = noised_data[i-1] + noise
            else:
                noised_data[i] = data[i]
        
        result.iloc[:] = noised_data
        return result

    def _noise_smoothed_row(self, s: pd.Series, smoothing_factor: float = 0.5, noise_std: float = 0.1, rng=None) -> pd.Series:

        result = s.copy()
        if rng is None:
            rng = np.random.default_rng()
        
        if len(result) == 0 or result.isna().all():
            return result
        
        data = result.values
        
        # Используем скользящее среднее для сглаживания
        window_length = max(3, int(len(data) * smoothing_factor))
        if window_length % 2 == 0:
            window_length += 1  # Делаем нечетным для savgol_filter
        
        window_length = min(window_length, len(data))
        if window_length < 3:
            # Слишком короткий ряд для сглаживания
            smoothed_data = data
        else:
            try:
                smoothed_data = signal.savgol_filter(data, window_length=window_length, polyorder=2)
            except:
                smoothed_data = data
        
        # Добавляем шум к сглаженным данным
        noise = rng.normal(scale=noise_std, size=len(smoothed_data))
        noised_data = smoothed_data + noise
        
        result.iloc[:] = noised_data
        return result

    def add_noise(self, df: pd.DataFrame, noise_type: str = "jitter", noise_level: Optional[float] = None,
              outlier_fraction: float = 0.01,
              outlier_magnitude: float = 5.0,
              rng_seed: Optional[int] = None) -> pd.DataFrame:
  
        rng = np.random.default_rng(rng_seed)

        methods_map = {
        "jitter_gauss": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_jitter_gauss_row(s, noise_level, rng=rng)),
        "jitter_white": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_jitter_white_row(s, noise_level, rng=rng)),
        "jitter_multiplicative": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_multiplicative_row(s, noise_level, rng=rng)),
        "jitter_outliers": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_outliers_row(s, outlier_fraction, outlier_magnitude, rng=rng)),
        "markovian": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_markovian_row(s, transition_prob=0.5, noise_std=0.1, rng=rng)),
        "smoothed": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_smoothed_row(s, smoothing_factor=0.5, noise_std=0.1, rng=rng))
        }

        # если указан конкретный метод — оставить поведение прежним
        if noise_type in methods_map and noise_type != "jitter":
            df_noised = methods_map[noise_type](df.copy())
            self.df_updated = df_noised.copy()
            return df_noised

        # автоматический построчный выбор (noise_type == "jitter" или "auto")
        df_num = df.copy()
        df_result = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        candidates = list(methods_map.keys())
        self.best_methods_noise = {}

        for idx in df_num.index:
            row_df = df_num.loc[[idx]].copy()
            if not row_df.isna().any(axis=None):
                
                df_result.loc[idx] = row_df.iloc[0].values
                self.best_methods_noise[idx] = "none"
                continue

            try:
               
                stats_orig = self.calculate_statistics(row_df)
            except Exception:
                
                df_result.loc[idx] = row_df.iloc[0].values
                self.best_methods_noise[idx] = "none"
                continue

            best_score = float("inf")
            best_candidate_row = row_df.copy()
            best_method_name = "none"
            for cand in candidates:
                try:

                    candidate_df = methods_map[cand](row_df.copy())

                    if candidate_df is None or not isinstance(candidate_df, pd.DataFrame):
                        
                        raise ValueError(f"{cand} вернул некорректный результат")

                    if candidate_df.shape != row_df.shape:
                        
                        candidate_df = candidate_df.reindex_like(row_df)

                    stats_new = self.calculate_statistics_modified(candidate_df)
                    if not stats_new:
                        raise ValueError(f"Не удалось вычислить статистику для метода {cand}")

                    total_score = 0.0
                    eps = 1e-8
                    for r_key in stats_orig.keys():
                        orig = stats_orig[r_key]
                        filled = stats_new.get(r_key, {})

                        mean_diff = abs((filled.get("Среднее", 0) or 0) - (orig.get("Среднее", 0) or 0)) / (abs(orig.get("Среднее", 0) or 0) + eps)
                        std_diff = abs((filled.get("Ст. откл.", 0) or 0) - (orig.get("Ст. откл.", 0) or 0)) / (abs(orig.get("Ст. откл.", 0) or 0) + eps)
                        smooth_diff = abs((filled.get("Локальная вариативность", 0) or 0) - (orig.get("Локальная вариативность", 0) or 0)) / ((orig.get("Локальная вариативность", eps) or eps) + eps)
                        curvature_diff = abs((filled.get("Локальная кривизна", 0) or 0) - (orig.get("Локальная кривизна", 0) or 0)) / ((orig.get("Локальная кривизна", eps) or eps) + eps)

                        total_score += mean_diff + std_diff + smooth_diff + curvature_diff

                    total_score /= max(1, len(stats_orig))

                    if total_score < best_score:
                        best_score = total_score
                        best_candidate_row = candidate_df.copy()
                        best_method_name = cand

                except Exception as e:
                    continue

            try:
                df_result.loc[idx] = best_candidate_row.iloc[0].values
            except Exception:
                df_result.loc[idx] = df_num.loc[idx].values

            self.best_methods_noise[idx] = best_method_name

        self.df_updated = df_result.copy()
        return df_result


    def calculate_statistics_modified(self, df_modified: pd.DataFrame) -> Dict:
     
        stats_all = {}
        for idx, row in df_modified.iterrows():
            series = pd.Series(row)
            stat_station = self._check_stationarity(series)
            count_missing = int(series.isna().sum())
            s_clean = series.dropna()
            diff1 = np.diff(s_clean) if len(s_clean) > 1 else np.array([0.0])
            diff2 = np.diff(diff1) if len(diff1) > 1 else np.array([0.0])
            stats_all[idx] = {
                "Длина ряда": int(len(series)),
                "Количество пустных значений": count_missing,
                "Доля пустных значений, %": round(count_missing / len(series) * 100, 2),
                "Среднее": round(series.mean(skipna=True), 4) if not series.dropna().empty else None,
                "Медиана": round(series.median(skipna=True), 4) if not series.dropna().empty else None,
                "Мин": round(series.min(skipna=True), 4) if not series.dropna().empty else None,
                "Макс": round(series.max(skipna=True), 4) if not series.dropna().empty else None,
                "Ст. откл.": round(series.std(skipna=True), 4) if not series.dropna().empty else None,
                "Категория стационарности": stat_station["Категория стационарности"],
                "ADF p-value": stat_station["ADF p-value"],
                "ADF Statistic": stat_station["ADF Statistic"],
                "Локальная вариативность": float(np.mean(diff1**2)) if len(diff1) > 0 else 0.0,
                "Локальная кривизна": float(np.mean(diff2**2)) if len(diff2) > 0 else 0.0
            }
        return stats_all

    def apply_action(self, df_input: pd.DataFrame, df_modified: pd.DataFrame, action: str, method: str, **kwargs):
   
        result_html = {}

        def _filter_selected(df: pd.DataFrame) -> pd.DataFrame:
            if self.selected_features:
                selected = [s for s in self.selected_features if s in df.index]
                return df.loc[selected]
            return df
        
        if action == "interpolate":
            methods_map = {
                "linear": self.interpolate_linear,
                "quadratic": self.interpolate_quadratic,
                "cubic": self.interpolate_cubic,
                "nearest": self.interpolate_nearest,
                "spline2": lambda d: self.interpolate_spline2(d, order=2),
                "spline3": lambda d: self.interpolate_spline3(d, order=3),
                "pchip": self.interpolate_pchip,
                "akima": self.interpolate_akima,
                "auto": self.interpolate_auto
            }
            fn = methods_map.get(method)
            if fn is None:
                raise ValueError(f"Неизвестный метод интерполяции: {method}")
          
            df_selected = _filter_selected(df_modified.copy())
            df_filled = fn(df_selected)
            df_modified.update(df_filled)
            self.df_updated = df_modified.copy()

        elif action == "jitter":
            
            noise_type = method if method is not None else "jitter"
            df_selected = _filter_selected(df_modified.copy())
            df_noised = self.add_noise(df_selected.copy(), noise_type=noise_type)
            df_modified.update(df_noised)
            self.df_updated = df_modified.copy()

        elif action == "extrapolate":
            if self.counter == 0:
                
                self.counter += 1
                self.fit_timegan(**kwargs)
                self.direction = method
                all_dates = pd.to_datetime(df_modified.columns)
                if method in ["forward", "both"]:

                    last_date = all_dates.max()
                    existing_future_dates = [d for d in all_dates if d > last_date]
                    if existing_future_dates:

                        df_selected = _filter_selected(df_modified.copy())
                        for idx in df_selected.index:
                            row_series = df_selected.loc[idx]
                            nan_in_extrapolated = row_series[existing_future_dates].isna()
                            if nan_in_extrapolated.any():
                
                                last_valid_idx = row_series[:last_date].last_valid_index()
                                if last_valid_idx:
                                    last_valid_value = row_series[last_valid_idx]
                                    df_modified.loc[idx, nan_in_extrapolated[nan_in_extrapolated].index] = last_valid_value
                
                if method in ["backward", "both"]:
                    first_date = all_dates.min()
                    existing_past_dates = [d for d in all_dates if d < first_date]
                    
                    if existing_past_dates:
                        df_selected = _filter_selected(df_modified.copy())
                        for idx in df_selected.index:
                            row_series = df_selected.loc[idx]
                            nan_in_extrapolated = row_series[existing_past_dates].isna()
                            
                            if nan_in_extrapolated.any():
                                first_valid_idx = row_series[first_date:].first_valid_index()
                                if first_valid_idx:
                                    first_valid_value = row_series[first_valid_idx]
                                    df_modified.loc[idx, nan_in_extrapolated[nan_in_extrapolated].index] = first_valid_value
                
                df_extrapolated = self.predict_timegan(**kwargs)
                
                for idx in df_extrapolated.index:
                    for date in df_extrapolated.columns:
                        if date in df_modified.columns:
                            if pd.isna(df_modified.at[idx, date]):
                                df_modified.at[idx, date] = df_extrapolated.at[idx, date]
                            elif idx in (self.selected_features or []):
                                df_modified.at[idx, date] = df_extrapolated.at[idx, date]
                        else:
                            if date not in df_modified.columns:
                                df_modified[date] = np.nan
                            df_modified.at[idx, date] = df_extrapolated.at[idx, date]
            
                df_modified = df_modified.reindex(sorted(df_modified.columns), axis=1)
                self.df_updated = df_modified.copy()

        stats_df = pd.DataFrame.from_dict(self.calculate_statistics_modified(df_modified), orient='index')
        result_html["df_modified_html"] = df_modified.to_html(classes="dataframe table table-sm", border=0)
        result_html["stats_modified_html"] = stats_df.to_html(classes="dataframe table table-sm", border=0)

        return df_modified, result_html
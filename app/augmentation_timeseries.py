from typing import Optional, Dict, Union
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from scipy import signal
from statsmodels.tsa.stattools import adfuller
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler
import os
import warnings

warnings.filterwarnings("ignore")

class AutoAugmentationTimeseries:
    
    def __init__(self, df_or_path, freq='D'):
        self.selected_features: list = []
        self.freq = freq
        self.df_input = self._load_and_standardize(df_or_path)   # ← ключевой вызов
        self.df_input = self.df_input.apply(pd.to_numeric, errors="coerce")
        self.stats: Optional[Dict] = None
        self.n_missing_total: Optional[int] = None
        self.df_updated: Optional[pd.DataFrame] = None
        self.model = None
        self.pred_len = 30
        self.timegan_model = None
        self.scaler = None
        self.seq_len = 30
        self.hidden_dim = 64
        self.num_layers = 2
        self.epochs_auto = 500
        self.epochs_super = 500
        self.epochs_gan = 500
        self.batch_size = 32
        self.alpha_teacher = 0.7
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.direction=None
        torch.manual_seed(42)
        np.random.seed(42)
        self.df_input.columns = pd.to_datetime(self.df_input.columns)
        self.df_updated = self.df_input
        self.counter = 0
    
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

        # Масштабирование
        self.scaler = MinMaxScaler()
        data_scaled = self.scaler.fit_transform(df.values)
        self.data_tensor = torch.tensor(data_scaled, dtype=torch.float32).to(self.device)
        self.data_tensor = self.data_tensor
        self.n_features = df.shape[1]

        # Создание окон
        def create_sequences(data, seq_len):
            X = []
            for i in range(len(data) - seq_len):
                X.append(data[i:i + seq_len])
            return torch.stack(X)

        train_data = create_sequences(self.data_tensor, self.seq_len)

        # Сборка модели
        class GRUWithLN(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers, dropout=0.1):
                super().__init__()
                self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
                self.ln = nn.LayerNorm(hidden_dim)
            def forward(self, x):
                h, _ = self.rnn(x)
                return self.ln(h)

        class Embedder(nn.Module):
            def __init__(self, input_dim, hidden_dim, num_layers):
                super().__init__()
                self.gru_ln = GRUWithLN(input_dim, hidden_dim, num_layers)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
            def forward(self, x):
                return torch.tanh(self.fc(self.gru_ln(x)))

        class Recovery(nn.Module):
            def __init__(self, hidden_dim, output_dim):
                super().__init__()
                self.fc = nn.Linear(hidden_dim, output_dim)
            def forward(self, h):
                return torch.sigmoid(self.fc(h))

        class Generator(nn.Module):
            def __init__(self, z_dim, hidden_dim, num_layers):
                super().__init__()
                self.gru_ln = GRUWithLN(z_dim, hidden_dim, num_layers)
                self.fc = nn.Linear(hidden_dim, hidden_dim)
            def forward(self, z):
                return torch.tanh(self.fc(self.gru_ln(z)))

        class Supervisor(nn.Module):
            def __init__(self, hidden_dim, num_layers):
                super().__init__()
                self.gru_ln = GRUWithLN(hidden_dim, hidden_dim, num_layers)
            def forward(self, h):
                return self.gru_ln(h)

        class Discriminator(nn.Module):
            def __init__(self, hidden_dim, num_layers):
                super().__init__()
                self.gru_ln = GRUWithLN(hidden_dim, hidden_dim, num_layers)
                self.fc = nn.Linear(hidden_dim, 1)
            def forward(self, h):
                return torch.sigmoid(self.fc(self.gru_ln(h)))

        # Объявление 
        embedder = Embedder(self.n_features, self.hidden_dim, self.num_layers).to(self.device)
        recovery = Recovery(self.hidden_dim, self.n_features).to(self.device)
        generator = Generator(self.hidden_dim, self.hidden_dim, self.num_layers).to(self.device)
        supervisor = Supervisor(self.hidden_dim, self.num_layers).to(self.device)
        discriminator = Discriminator(self.hidden_dim, self.num_layers).to(self.device)

        loss_fn = nn.MSELoss()

        opt_auto = torch.optim.Adam(list(embedder.parameters()) + list(recovery.parameters()), lr=1e-3)
        opt_super = torch.optim.Adam(list(supervisor.parameters()) + list(embedder.parameters()), lr=1e-3)
        opt_gen = torch.optim.Adam(list(generator.parameters()) + list(supervisor.parameters()), lr=1e-3)
        opt_disc = torch.optim.Adam(discriminator.parameters(), lr=1e-4)

        # Обучение автоэнкодера
        for epoch in range(self.epochs_auto):
            idx = torch.randint(0, len(train_data), (self.batch_size,))
            X = train_data[idx]

            opt_auto.zero_grad()
            H = embedder(X)
            X_tilde = recovery(H)
            loss_auto = loss_fn(X_tilde, X)
            loss_auto.backward()
            opt_auto.step()

            if epoch % 100 == 0:
                print(f"Auto Epoch {epoch:04d}: recon_loss={loss_auto.item():.6f}")

        # Обучение супервизора
        for epoch in range(self.epochs_super):
            idx = torch.randint(0, len(train_data), (self.batch_size,))
            X = train_data[idx]
            opt_super.zero_grad()
            H = embedder(X).detach()
            H_hat = supervisor(H)
            loss_sup = loss_fn(H_hat[:, :-1, :], H[:, 1:, :])
            loss_sup.backward()
            opt_super.step()

            if epoch % 100 == 0:
                print(f"Super Epoch {epoch:04d}: sup_loss={loss_sup.item():.6f}")

        # Обучение генератора и дискриминатор
        for epoch in range(self.epochs_gan):
            w_super = min(1.0, 0.01 + epoch / self.epochs_gan) # Динамическое изменение весов похожее на оригинальном исполнеии (но очень примитивное)
            w_adv = min(1.0, 0.01 + epoch / self.epochs_gan)

            idx = torch.randint(0, len(train_data), (self.batch_size,))
            X = train_data[idx]
            H_real = embedder(X).detach()

            opt_gen.zero_grad()
            Z = torch.randn_like(H_real)
            H_fake = generator(Z)
            H_fake_s = supervisor(H_fake)
            Y_fake = discriminator(H_fake)
            loss_gan = torch.mean((Y_fake - 1) ** 2)
            loss_sup_gen = loss_fn(H_fake_s[:, :-1, :], H_fake[:, 1:, :])
            total_gen_loss = w_adv * loss_gan + w_super * loss_sup_gen
            total_gen_loss.backward()
            opt_gen.step()

            opt_disc.zero_grad()
            Y_real = discriminator(H_real)
            Y_fake_det = discriminator(H_fake.detach())
            loss_disc = torch.mean((Y_real - 1) ** 2) + torch.mean(Y_fake_det ** 2)
            loss_disc.backward()
            opt_disc.step()

            if epoch % 100 == 0:
                print(f"GAN Epoch {epoch:04d}: gen_loss={total_gen_loss.item():.6f}, disc_loss={loss_disc.item():.6f}")

        self.model = (embedder, recovery, generator, supervisor, discriminator)
    
    def _extend_datetime_index(self):
        if not isinstance(self.df_updated.T.index, pd.DatetimeIndex) or len(self.df_updated.T.index) < 2:
            raise ValueError("Индекс должен быть DatetimeIndex с >=2 элементов")

        df_extended = self.df_updated.T.copy()
        step = self.df_updated.T.index[1] - self.df_updated.T.index[0]

        # Forward индексы
        if self.direction in ["forward", "both"]:
            last_idx = df_extended.index[-1]
            self.n_forward = self.pred_len if self.direction == "forward" else self.pred_len // 2 + self.pred_len % 2
            self.forward_idx = [last_idx + step*(i+1) for i in range(int(self.n_forward))]
        else:
            self.forward_idx = []

        # Backward индексы
        if self.direction in ["backward", "both"]:
            first_idx = df_extended.index[0]
            self.n_backward = self.pred_len if self.direction == "backward" else self.pred_len // 2
            self.backward_idx = [first_idx - step*(i+1) for i in reversed(range(int(self.n_backward)))]
        else:
            self.backward_idx = []

        # Расширяем df, чтобы можно было вставить прогнозы
        for idx in self.backward_idx + self.forward_idx:
            if idx not in df_extended.index:
                df_extended.loc[idx] = np.nan

        return df_extended, self.forward_idx, self.backward_idx

    def predict_timegan(self, **kwargs):
        if self.model is None:
            raise RuntimeError("Модель TimeGAN не обучена. Сначала вызови fit_timegan()")

        embedder, recovery, generator, supervisor, _ = self.model
        embedder.eval(); recovery.eval(); generator.eval(); supervisor.eval()
        df_extended, self.forward_idx, self.backward_idx = self._extend_datetime_index()
        n_features = self.data_tensor.shape[1]

        with torch.no_grad():
            # Прогноз назад
            backward_arr = []
            if self.backward_idx:
                last_seq = self.data_tensor[:self.seq_len].unsqueeze(0)
                H_last = embedder(last_seq)
                next_hidden = H_last[:, :1, :]
                for _ in range(len(self.backward_idx)):
                    z = torch.randn(next_hidden.shape, device=self.device)
                    h_next = generator(z)
                    next_hidden = self.alpha_teacher * next_hidden + (1 - self.alpha_teacher) * h_next
                    x_next = recovery(next_hidden)
                    backward_arr.append(x_next.squeeze(0).squeeze(0).cpu().numpy())
                backward_arr = np.stack(backward_arr[::-1], axis=0)
            else:
                backward_arr = np.empty((0, n_features))

            # Прогноз вперёд
            forward_arr = []
            if self.forward_idx:
                last_seq = self.data_tensor[-self.seq_len:].unsqueeze(0)
                H_last = embedder(last_seq)
                next_hidden = H_last[:, -1:, :]
                for _ in range(len(self.forward_idx)):
                    z = torch.randn(next_hidden.shape, device=self.device)
                    h_next = generator(z)
                    next_hidden = self.alpha_teacher * next_hidden + (1 - self.alpha_teacher) * h_next
                    x_next = recovery(next_hidden)
                    forward_arr.append(x_next.squeeze(0).squeeze(0).cpu().numpy())
                forward_arr = np.stack(forward_arr, axis=0)
            else:
                forward_arr = np.empty((0, n_features))

            # Объединяем
            generated = np.vstack([backward_arr, forward_arr])
            generated_original = self.scaler.inverse_transform(generated)

        df_extended.loc[self.backward_idx + self.forward_idx] = generated_original
        full_df = df_extended.sort_index().T.copy()
     #   self.df_updated = full_df.copy() 
        df_scaled_new = self.scaler.fit_transform(full_df.T.values) 
        self.data_tensor = torch.tensor(df_scaled_new, dtype=torch.float32).to(self.device)
        extrapolated_df = pd.DataFrame(generated_original, index=self.backward_idx + self.forward_idx).T
        if hasattr(self, 'selected_features') and self.selected_features:
            valid_features = [f for f in self.selected_features if f in extrapolated_df.index]
            if valid_features:
                extrapolated_df = extrapolated_df.loc[valid_features]
        
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
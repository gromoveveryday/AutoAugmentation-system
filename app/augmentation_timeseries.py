from typing import Optional, Dict, Union
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
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
    def __init__(self, df_or_path: Union[pd.DataFrame, str]):
        if isinstance(df_or_path, pd.DataFrame):
            self.df_input = df_or_path.copy()
        elif isinstance(df_or_path, str):
            if not os.path.exists(df_or_path):
                raise FileNotFoundError(f"Файл {df_or_path} не найден.")
            df = pd.read_csv(df_or_path)
            if df.iloc[:, 0].dtype == object and df.iloc[:, 0].is_unique:
                df = df.set_index(df.columns[0])
            self.df_input = df
        else:
            raise ValueError("df_or_path должен быть DataFrame или путь к CSV")

        self.df_input = self.df_input.apply(pd.to_numeric, errors="coerce")
        self.stats: Optional[Dict] = None
        self.n_missing_total: Optional[int] = None
        self.df_updated: Optional[pd.DataFrame] = None
        self.model = None
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
        self.direction='forward'
        torch.manual_seed(42)
        np.random.seed(42)
        self.df_input.columns = pd.to_datetime(self.df_input.columns)
        self.df_updated = self.df_input
        self.counter = 0

    def fit_timegan(self, pred_len=30):
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
    
    def _extend_datetime_index(self, pred_len=30):
        if not isinstance(self.df_updated.T.index, pd.DatetimeIndex) or len(self.df_updated.T.index) < 2:
            raise ValueError("Индекс должен быть DatetimeIndex с >=2 элементов")

        df_extended = self.df_updated.T.copy()
        step = self.df_updated.T.index[1] - self.df_updated.T.index[0]

        # Forward индексы
        if self.direction in ["forward", "both"]:
            last_idx = df_extended.index[-1]
            n_forward = pred_len if self.direction == "forward" else pred_len // 2 + pred_len % 2
            forward_idx = [last_idx + step*(i+1) for i in range(n_forward)]
        else:
            forward_idx = []

        # Backward индексы
        if self.direction in ["backward", "both"]:
            first_idx = df_extended.index[0]
            n_backward = pred_len if self.direction == "backward" else pred_len // 2
            backward_idx = [first_idx - step*(i+1) for i in reversed(range(n_backward))]
        else:
            backward_idx = []

        # Расширяем df, чтобы можно было вставить прогнозы
        for idx in backward_idx + forward_idx:
            if idx not in df_extended.index:
                df_extended.loc[idx] = np.nan

        return df_extended, forward_idx, backward_idx

    def predict_timegan(self, **kwargs):
        if self.model is None:
            raise RuntimeError("Модель TimeGAN не обучена. Сначала вызови fit_timegan()")

        embedder, recovery, generator, supervisor, _ = self.model
        embedder.eval(); recovery.eval(); generator.eval(); supervisor.eval()

        df_extended, forward_idx, backward_idx = self._extend_datetime_index()
        n_features = self.data_tensor.shape[1]

        with torch.no_grad():
            # Прогноз назад
            backward_arr = []
            if backward_idx:
                last_seq = self.data_tensor[:self.seq_len].unsqueeze(0)
                H_last = embedder(last_seq)
                next_hidden = H_last[:, :1, :]
                for _ in range(len(backward_idx)):
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
            if forward_idx:
                last_seq = self.data_tensor[-self.seq_len:].unsqueeze(0)
                H_last = embedder(last_seq)
                next_hidden = H_last[:, -1:, :]
                for _ in range(len(forward_idx)):
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

        # Заполняем расширенный df
        df_extended.loc[backward_idx + forward_idx] = generated_original
        self.df_updated = df_extended.T.copy()
        return self.df_updated
  
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

    @staticmethod
    def kalman_manual(series: pd.Series, A=1, H=1, Q=1e-5, R=1e-2) -> pd.Series:
        n = len(series)
        if n == 0:
            return series
        x_est = np.zeros(n, dtype=float)
        P = np.zeros(n, dtype=float)
        s_clean = pd.to_numeric(series, errors="coerce").dropna()
        if s_clean.empty:
            return pd.Series([np.nan] * n, index=series.index)
        x_est[0] = float(s_clean.iloc[0])
        P[0] = 1.0
        for t in range(1, n):
            x_pred = A * x_est[t - 1]
            P_pred = A * P[t - 1] * A + Q
            if pd.isna(series.iloc[t]):
                x_est[t] = x_pred
                P[t] = P_pred
            else:
                z = float(series.iloc[t])
                K = P_pred * H / (H * P_pred * H + R)
                x_est[t] = x_pred + K * (z - H * x_pred)
                P[t] = (1 - K * H) * P_pred
        return pd.Series(x_est, index=series.index)

    def _rowwise_interpolate(self, df: pd.DataFrame, method: str, order: int = None) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        df_filled = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        for idx, row in df_num.iterrows():
            s = row.copy()
            mask_notna = s.notna()
            if mask_notna.sum() < 2:  # мало данных для интерполяции
                df_filled.loc[idx] = s.values
                continue
            first, last = mask_notna.idxmax(), mask_notna[::-1].idxmax()  # границы
            s_trunc = s.loc[first:last]
            try:
                if method in ["polynomial", "spline"]:
                    filled_row = s_trunc.interpolate(method=method, order=order, limit_direction='both')
                else:
                    filled_row = s_trunc.interpolate(method=method, limit_direction='both')

                s.loc[first:last] = filled_row
            except Exception:
                pass
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

            first_idx = np.where(mask_notna)[0][0]
            last_idx = np.where(mask_notna)[0][-1]

            # точки для интерполяции внутри first..last
            x_known = idx_num[first_idx:last_idx+1][mask_notna[first_idx:last_idx+1]]
            y_known = s[first_idx:last_idx+1][mask_notna[first_idx:last_idx+1]]

            if len(x_known) < 2:
                # интерполировать нечего
                filled.loc[r_idx] = s
                continue

            try:
                interpolator = PchipInterpolator(x_known, y_known)
                s[first_idx:last_idx+1] = interpolator(idx_num[first_idx:last_idx+1])
            except Exception:
                pass

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
            mask = ~np.isnan(s)
            if mask.sum() == 0:
                filled.loc[r_idx] = [np.nan] * ncols
            elif mask.sum() == 1:
                filled.loc[r_idx] = [float(s[mask][0])] * ncols
            else:
                try:
                    filled_vals = Akima1DInterpolator(idx_num[mask], s[mask])(idx_num)
                    filled.loc[r_idx] = filled_vals
                except Exception:
                    filled.loc[r_idx] = s
        filled.columns = df.columns
        return filled

    def interpolate_kalman(self, df: pd.DataFrame) -> pd.DataFrame:
        df_num = df.apply(pd.to_numeric, errors="coerce")
        filled = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        for r_idx, row in df_num.iterrows():
            s = pd.Series(row.values, index=df_num.columns)
            mask_notna = s.notna()
            if mask_notna.sum() < 2:
                filled.loc[r_idx] = s.values
                continue

            first, last = mask_notna.idxmax(), mask_notna[::-1].idxmax()
            s_trunc = s.loc[first:last]

            s_trunc_filled = self.kalman_manual(s_trunc)
            s.loc[first:last] = s_trunc_filled

            filled.loc[r_idx] = s.values

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
            self.interpolate_akima,
            self.interpolate_kalman
        ]

        method_names = [
            "linear", "quadratic", "cubic", "nearest",
            "spline2", "spline3", "pchip", "akima", "kalman"
        ]

        df_num = df.copy()
        df_result = pd.DataFrame(index=df_num.index, columns=df_num.columns, dtype=float)
        self.best_methods_interpolation = {}

        for idx in df_num.index:
            row_df = df_num.loc[[idx]].copy()  # DataFrame с одной строкой
            # если в строке нет пропусков — ничего не делаем
            if not row_df.isna().any(axis=None):
                df_result.loc[idx] = row_df.iloc[0].values
                self.best_methods_interpolation[idx] = "none"
                continue

            try:
                stats_orig = self.calculate_statistics(row_df)
            except Exception:
                # если не удалось — оставляем как есть
                df_result.loc[idx] = row_df.iloc[0].values
                self.best_methods_interpolation[idx] = "none"
                continue

            best_score = float("inf")
            best_candidate_row = row_df.copy()
            best_method_name = "none"

            for fn, name in zip(methods, method_names):
                try:
                    candidate = fn(row_df.copy())  # метод возвращает DataFrame
                    # валидация результата
                    if candidate is None or not isinstance(candidate, pd.DataFrame):
                        raise ValueError(f"{name} вернул некорректный результат")

                    # привести форму к исходной строке
                    if candidate.shape != row_df.shape:
                        candidate = candidate.reindex_like(row_df)

                    stats_new = self.calculate_statistics_modified(candidate)
                    if not stats_new:
                        raise ValueError(f"Не удалось вычислить статистику для метода {name}")

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
                        best_candidate_row = candidate.copy()
                        best_method_name = name

                except Exception as e:
                    print(f"⚠️ Метод '{name}' пропущен для строки {idx}: {e}")
                    continue

            try:
                df_result.loc[idx] = best_candidate_row.iloc[0].values
            except Exception:
                df_result.loc[idx] = df_num.loc[idx].values

            self.best_methods_interpolation[idx] = best_method_name

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

    def _noise_jitter_normal_row(self, s: pd.Series, noise_level: Optional[float] = None, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        std = s.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0
        scale = (noise_level if noise_level is not None else 0.02) * std
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

    def add_noise(self, df: pd.DataFrame, noise_type: str = "jitter", noise_level: Optional[float] = None,
              outlier_fraction: float = 0.01,
              outlier_magnitude: float = 5.0,
              rng_seed: Optional[int] = None) -> pd.DataFrame:
  
        rng = np.random.default_rng(rng_seed)

        methods_map = {
            "jitter_gauss": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_jitter_gauss_row(s, noise_level, rng)),
            "jitter_normal": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_jitter_normal_row(s, noise_level, rng)),
            "jitter_white": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_jitter_white_row(s, noise_level, rng)),
            "jitter_multiplicative": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_multiplicative_row(s, noise_level, rng)),
            "jitter_outliers": lambda d: self._rowwise_apply_noise(d, lambda s: self._noise_outliers_row(s, outlier_fraction, outlier_magnitude, rng))
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
            # если в строке нет значений (всё NaN) — просто копируем
            if not row_df.isna().any(axis=None):
                # строка без пропусков — можно всё равно добавить шум, но по логике оригинала пропустим
                # чтобы сохранить прежнее поведение, не меняем строку
                df_result.loc[idx] = row_df.iloc[0].values
                self.best_methods_noise[idx] = "none"
                continue

            # статистики до добавления шума
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
                    # применяем метод к одной строке (функции принимают DataFrame)
                    candidate_df = methods_map[cand](row_df.copy())

                    if candidate_df is None or not isinstance(candidate_df, pd.DataFrame):
                        raise ValueError(f"{cand} вернул некорректный результат")

                    if candidate_df.shape != row_df.shape:
                        candidate_df = candidate_df.reindex_like(row_df)

                    stats_new = self.calculate_statistics_modified(candidate_df)
                    if not stats_new:
                        raise ValueError(f"Не удалось вычислить статистику для метода {cand}")

                    # вычисляем score
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
                "kalman": self.interpolate_kalman,
                "auto": self.interpolate_auto
            }
            fn = methods_map.get(method)
            if fn is None:
                raise ValueError(f"Неизвестный метод интерполяции: {method}")
            df_modified  = fn(df_modified.copy())
            self.df_updated = df_modified.copy()

        elif action == "jitter":
            noise_type = method if method is not None else "jitter"
            df_modified = self.add_noise(df_modified.copy(), noise_type=noise_type)
            self.df_updated = df_modified.copy()

        elif action == "extrapolate":
            if self.counter == 0:
                self.counter += 1
                self.fit_timegan(**kwargs)
            
            df_modified = self.predict_timegan(**kwargs)
            self.df_updated = df_modified.copy()

        stats_df = pd.DataFrame.from_dict(self.calculate_statistics_modified(df_modified), orient='index')
        result_html["df_modified_html"] = df_modified.to_html(classes="dataframe table table-sm", border=0)
        result_html["stats_modified_html"] = stats_df.to_html(classes="dataframe table table-sm", border=0)

        return df_modified, result_html
    fit_timegan
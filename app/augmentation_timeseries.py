from typing import Optional, Dict, Union, Tuple
import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator, Akima1DInterpolator
from statsmodels.tsa.stattools import adfuller
import torch
from torch import nn
import warnings
import os
import time

warnings.filterwarnings("ignore")

class TimeGAN(nn.Module):
    def __init__(self, input_dim, hidden_dim=24, num_layers=3, device=None):
        super(TimeGAN, self).__init__()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.embedder = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.recovery = nn.LSTM(hidden_dim, input_dim, num_layers, batch_first=True)

        self.generator = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.supervisor = nn.LSTM(hidden_dim, hidden_dim, num_layers - 1, batch_first=True)

        self.discriminator_lstm = nn.LSTM(hidden_dim, hidden_dim, num_layers, batch_first=True)
        self.discriminator_linear = nn.Linear(hidden_dim, 1)
        self.discriminator_sigmoid = nn.Sigmoid()
        
        self.hidden_dim = hidden_dim
        self.to(self.device)

    def embed(self, x):
        h, _ = self.embedder(x)
        return h

    def recover(self, h):
        x_tilde, _ = self.recovery(h)
        return x_tilde
    
    def discriminator(self, h):
        lstm_out, _ = self.discriminator_lstm(h)  
        linear_out = self.discriminator_linear(lstm_out)
        return self.discriminator_sigmoid(linear_out)

    def generate(self, n_series=1, seq_len=10):
        z = torch.randn((n_series, seq_len, self.hidden_dim)).to(self.device)
        g_h, _ = self.generator(z)
        s_h, _ = self.supervisor(g_h)
        x_fake, _ = self.recovery(s_h)
        return x_fake.detach().cpu().numpy()

    def fit(self, data, epochs=500, batch_size=32, pretrain_steps=200):
        data = torch.tensor(data, dtype=torch.float32, device=self.device)

        optimizer_auto = torch.optim.Adam(list(self.embedder.parameters()) + list(self.recovery.parameters()), lr=1e-3)
        optimizer_gan = torch.optim.Adam(self.parameters(), lr=1e-3)

        mse = nn.MSELoss()
        bce = nn.BCELoss()

        for step in range(pretrain_steps):
            optimizer_auto.zero_grad()
            h = self.embed(data)
            x_tilde = self.recover(h)
            loss = mse(x_tilde, data)
            loss.backward()
            optimizer_auto.step()
            if step % 50 == 0:
                print(f"Шаг предобучения {step}/{pretrain_steps}, Лоссы={loss.item():.4f}")
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
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def fit_timegan(self, df, hidden_dim=24, num_layers=3, batch_size=32, epochs=500, pretrain_steps=200):
        df_clean = df.dropna().copy()
        if df_clean.empty:
            raise ValueError("После удаления NaN нет данных для обучения TimeGAN.")
        
        x = df_clean.values.reshape(1, df_clean.shape[0], df_clean.shape[1])
        x_norm = self._normalize(x)

        self.model = TimeGAN(
            input_dim=df_clean.shape[1],
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            device=self.device
        )
        self.model.fit(x_norm, epochs=epochs, batch_size=batch_size, pretrain_steps=pretrain_steps)
    
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        x_min = np.nanmin(x, axis=1, keepdims=True)
        x_max = np.nanmax(x, axis=1, keepdims=True)
        diff = x_max - x_min
        diff[diff == 0] = 1e-8  # защита от деления на 0
        x_norm = (x - x_min) / diff
        x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)
        return x_norm

    def _denormalize(self, x_norm: np.ndarray, x_original: np.ndarray = None) -> np.ndarray:
        if x_original is None:
            x_original = self.df_input.values.reshape(1, self.df_input.shape[0], self.df_input.shape[1])
        x_min = np.nanmin(x_original, axis=1, keepdims=True)
        x_max = np.nanmax(x_original, axis=1, keepdims=True)
        x = x_norm * (x_max - x_min + 1e-8) + x_min
        return x
    
    def extrapolate_timegan(self, periods=10, mode="forward"):
        model = self.model
        model.eval()

        x = self.df_input.values
        x = x.reshape(1, x.shape[0], x.shape[1])
        x_norm = torch.tensor(self._normalize(x), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            h = model.embed(x_norm)

            if mode == "forward":
                h_last = h[:, -1:, :]
                z_future = h_last.repeat(1, periods, 1) + 0.01 * torch.randn((1, periods, h.shape[-1]), device=self.device)
                h_future = model.supervisor(model.generator(z_future)[0])[0]  # генератор + супервизор
                x_future = model.recover(h_future)
                x_future = self._denormalize(x_future.cpu().numpy())[0]
                df_future = pd.DataFrame(x_future, columns=self.df_input.columns)
                df_update = pd.concat([pd.DataFrame(x[0], columns=self.df_input.columns), df_future], ignore_index=True)

            elif mode == "backward":
                h_first = h[:, :1, :]
                z_past = h_first.repeat(1, periods, 1) + 0.01 * torch.randn((1, periods, h.shape[-1]), device=self.device)
                h_past = model.supervisor(model.generator(z_past)[0])[0]
                x_past = model.recover(h_past)
                x_past = self._denormalize(x_past.cpu().numpy())[0]
                df_past = pd.DataFrame(x_past, columns=self.df_input.columns)
                df_update = pd.concat([df_past, pd.DataFrame(x[0], columns=self.df_input.columns)], ignore_index=True)

            elif mode == "both":
                half = periods // 2
                h_first = h[:, :1, :]
                z_past = h_first.repeat(1, half, 1) + 0.01 * torch.randn((1, half, h.shape[-1]), device=self.device)
                h_past = model.supervisor(model.generator(z_past)[0])[0]
                x_past = model.recover(h_past)
                x_past = self._denormalize(x_past.cpu().numpy())[0]
                df_past = pd.DataFrame(x_past, columns=self.df_input.columns)

                h_last = h[:, -1:, :]
                z_future = h_last.repeat(1, periods - half, 1) + 0.01 * torch.randn((1, periods - half, h.shape[-1]), device=self.device)
                h_future = model.supervisor(model.generator(z_future)[0])[0]
                x_future = model.recover(h_future)
                x_future = self._denormalize(x_future.cpu().numpy())[0]
                df_future = pd.DataFrame(x_future, columns=self.df_input.columns)

                df_update = pd.concat([df_past, pd.DataFrame(x[0], columns=self.df_input.columns), df_future], ignore_index=True)

        self.df_updated = df_update
        return df_update
    
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
        if rng is None:
            rng = np.random.default_rng()
        res = s.copy()
        n = len(s)
        k = max(1, int(round(n * outlier_fraction)))
        idxs = rng.choice(n, size=k, replace=False)
        std = s.std(skipna=True)
        if pd.isna(std) or std == 0:
            std = 1.0
        for i in idxs:
            sign = rng.choice([-1, 1])
            res.iloc[i] = res.iloc[i] + sign * outlier_magnitude * std
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

    def apply_action(self, df_input: pd.DataFrame, df_modified: pd.DataFrame, action: str, method: str, periods=None):
   
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
            df_modified = fn(df_modified.copy())
            self.df_updated = df_modified.copy()

        elif action == "jitter":
            noise_type = method if method is not None else "jitter"
            df_modified = self.add_noise(df_modified.copy(), noise_type=noise_type)
            self.df_updated = df_modified.copy()

        elif action == "extrapolate":
            mode_map = {
            "Экстраполяция": "forward",
            "Ретрополяция": "backward",
            "Комбинированная экстраполяция": "both"
            }
            mode = mode_map.get(method, "forward")

            if self.model is None:
                self.fit_timegan(df_input)
            df_modified = self.extrapolate_timegan(periods=periods, mode=mode)
            self.df_updated = df_modified.copy()

        stats_df = pd.DataFrame.from_dict(self.calculate_statistics_modified(df_modified), orient='index')
        result_html["df_modified_html"] = df_modified.to_html(classes="dataframe table table-sm", border=0)
        result_html["stats_modified_html"] = stats_df.to_html(classes="dataframe table table-sm", border=0)

        return df_modified, result_html

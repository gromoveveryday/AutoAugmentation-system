import os
import uuid
import random
import pandas as pd
import numpy as np
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import pairwise_distances
import cv2
from skimage.metrics import structural_similarity as ssim
import torch
import torchvision.transforms as T
from piq import brisque # из cv2 не запускается у меня почему-то
from sklearn.cluster import KMeans
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing import image
from itertools import cycle
import uuid

class autoaugmentation_images():
    
    # константы для расчета метрик - сколяюсь к тому, что пользователю они не будут доступы 
    bins = 32 # Для расчета цветовой гистограммы, может быть меньше/больше, нужно подумать про динамический расчет  
    epsilon = 1e-6 # малое эпсилон для избежания деления на 0 в цветовой метрике
    min_n_clusters = 5 # Минимальное количество кластеров/референсов для расчета микшированной метрики
    min_n_references = min_n_clusters
    model = MobileNetV2(weights='imagenet', # предобученная модель на нахождение объектов
                               include_top=False, # убирает полносвязный слой с классами обученных данных
                               pooling='avg', # Чтобы вернуть ближайшее расстояние 
                               input_shape=(128, 128, 3)) # Модель для нахождения эмбеддингов
    
    # Константы для аугументации - тут возможен вывод для пользователя 

    # Параметры геометрической аугументации (верх/низ диапазонов)
    min_crop_ratio = 0.8 # Нижняя граница случайного среза - 20% от изображения  
    max_crop_ratio = 0.95 # Верхняя граница случайного среза - 5 % от изображения
    min_zoom_factor = 1.05 # Приближение 1.1
    max_zoom_factor = 1.4  # Приближение X 1.5

    # Параметры цветовой аугументации (верх/низ диапазонов)

    min_h_shift = 5 # Изменение оттенка на 5% 
    max_h_shift = 30 # Изменение оттенка на 30% 
    min_s_scale = 0.8 # Уменьшение насыщенности на 20%
    max_s_scale = 1.5  # Увеличение насыщенности на 50%
    min_v_scale = 0.7 # Уменьшение яркости на 30%
    max_v_scale = 1.4 # Увеличение яркости на 40%
    min_bits = 3 # # Минимальное количество бит чтобы изменить их количество на пастеризацию изображений
    max_bits = 6 # Максимальное количество бит чтобы изменить их количество на пастеризацию изображений
    base_bits = 8 # Начальное количество бит для адекватной пастеризации изображения

    # Параметры шумовой аугументации

    salt_vs_pepper = 0.5 # Баланс белого/черного шума в аугументации Salt и Papper (ближе к 1 - больше белого шума)
    mean_gaussian_noise = 0 # Среднее значение для распределения N гауссовского шума
    std_gaussian_noise = 25 # Стандартное отклонение значение для распределения N гауссовского шума
    min_amount = 0.01  # Минимальное количество % пикселей в изображении, которое заменится на шум (1%)
    max_amount = 0.15 # Максимальное количество % пикселей в изображении, которое заменится на шум (15%)

    # Параметры микшированной аугументации
    min_alpha = 0.3 # Минимальный размер альфа для альфа блидинга
    max_alpha = 0.7 # Максимальный размер альфа для альфа блидинга
    alpha_for_cutmix = 0.5 # Значение альфа для для cutmix 

    def __init__(self, images_dir, labels_dir, augumentation_dir, n_samples_to_augument,
                  max_nfeatures_to_orb, dir_images_with_labels, dir_with_new_labels, is_labeled):
        
        self.images_dir = images_dir  # Папка с изображениями
        self.augumentation_dir = augumentation_dir # Куда нужно сохранить
        self.labels_dir = labels_dir # Где лежат разметки
        self.dir_images_with_labels = dir_images_with_labels # Директория с разметкой Bbox
        self.dir_with_new_labels = dir_with_new_labels # Новые координаты аугументированных данных Bbox
        self.n_samples_to_augument = n_samples_to_augument  # Сколько аугментировать
        self.max_nfeatures_to_orb = max_nfeatures_to_orb # Максимально допустимое количество ключевых точек для расчета ORB (геометрики)
        self.filenames = [
            f for f in os.listdir(images_dir)
            if os.path.splitext(f)[1].lower() in ('.jpg', '.jpeg', '.png')] # Сразу берет только изображения 
        self.labels = [x.split('.')[0] for x in self.filenames]  # Метки (если имя файла = метка)
        self.metadata = pd.DataFrame({'filename': self.filenames, 'label': self.labels})
        self.is_error = 0 # Флаг ошибки 0 в начале
        self.is_labeled = is_labeled # Есть ли разметка у изображений

        if self.n_samples_to_augument < 4:
            self.is_error = 1

        try:
            if isinstance(self.max_nfeatures_to_orb, str) and self.max_nfeatures_to_orb == 'По умолчанию':
                self.max_nfeatures_to_orb = 5000  # Продвинутый параметр, чем выше разрешение, тем больше максимальное количество ключевых точек на нем
            elif self.max_nfeatures_to_orb.is_integer():
                self.max_nfeatures_to_orb = int(self.max_nfeatures_to_orb)
        except AttributeError:
            # Обработка случая, когда max_nfeatures_to_orb не имеет метода is_integer()
            pass
    
    def check_augmentation_label_consistency(self):
        images_dir = os.path.join(self.images_dir)
        labels_dir = os.path.join(self.labels_dir)
        
        image_files = set(f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png')))
        label_files = set(f for f in os.listdir(labels_dir) if f.lower().endswith('.txt'))
        
        image_basenames = {os.path.splitext(f)[0] for f in image_files}
        label_basenames = {os.path.splitext(f)[0] for f in label_files}
        
        missing_labels = image_basenames - label_basenames
        missing_images = label_basenames - image_basenames
        
        if missing_labels | missing_images:
            self.is_error = 1

    def calculate_normalized_images_size(self):
        widths, heights = [], [] # Пустые списки широт и высот

        if not self.filenames:
            self.is_error = 1
            return

        for img_name in self.filenames:
            img_path = os.path.join(self.images_dir, img_name)

            try:
                with Image.open(img_path) as img:
                    w, h = img.size
                    widths.append(w)
                    heights.append(h)
            except UnidentifiedImageError:
                print(f"Файл {img_name} не является изображением") # Можно убрать, если не нужно
            except Exception as e:
                print(f"Ошибка при открытии {img_name}: {e}")

        
            self.median_width = int(np.median(widths)) # Медианная ширина
            self.median_height = int(np.median(heights)) # Медианная высота
            self.max_width = int(np.max(widths)) # Максимально возможная ширина изображений для расчета nfeatures для ORB
            self.max_height = int(np.max(heights)) # Максимально возможная высота изображений для расчета nfeatures для ORB

    def calculate_geometric_metric(self): # Расчет геометрической монотопнности 
        def estimate_orb_nfeatures(img_w, img_h): # Динамический расчет среднего количества 
            # ключевых точек для ORB по медианным высотам и широтам
            base_area = self.max_width * self.max_height # Начальные средние размеры для расчета нормализованного размера
            img_area = img_w * img_h
            base_features = 500 # nfeatures по умолчанию в ORB из CV2
            scale_factor = img_area / base_area
            nfeatures = int(base_features * scale_factor)
            return max(500, min(self.max_nfeatures_to_orb, nfeatures))
        
        img_w, img_h = self.median_width, self.median_height
        nfeatures = estimate_orb_nfeatures(img_w, img_h)
        orb = cv2.ORB_create(nfeatures=nfeatures)
        
        results = []
        all_mean_coords = []

        # Первый проход — сбор средних координат
        for filename in self.filenames:
            img_path = os.path.join(self.images_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            
            if img is None:
                continue
            
            img = cv2.resize(img, (img_w, img_h))
            keypoints = orb.detect(img, None)
            if keypoints:
                coords = np.array([kp.pt for kp in keypoints])
                mean_coords = coords.mean(axis=0)
                all_mean_coords.append(mean_coords)

        # Расчёт метрики
        if len(all_mean_coords) >= 2: # Для расчетов геометрической и микшированной метрики прежполагается наличие 2 и 
            # более сэмплов, иначе расчеты некорректные 
            all_mean_coords = np.array(all_mean_coords)
            var_x = np.var(all_mean_coords[:, 0])
            var_y = np.var(all_mean_coords[:, 1])
            max_disp_x = (img_w ** 2) / 12
            max_disp_y = (img_h ** 2) / 12
            norm_var_x = var_x / max_disp_x
            norm_var_y = var_y / max_disp_y
            avg_norm_var = (norm_var_x + norm_var_y) / 2
            self.geometric_metric = round(float(max(0.0, min(100.0, 100 * (1 - avg_norm_var)))), 2)
        
        else:
            print("Недостаточно фотографий для расчета геометрической метрики. Добавьте еще фотографий")
            self.geometric_metric = 0 # брейк если мало фотографий
            return

        # Второй проход — сбор метаданных по каждому изображению
        for filename in self.filenames:
            img_path = os.path.join(self.images_dir, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            record = {
                'filename': filename,
                'keypoints_by_orb': None,
                'kp_count': 0,
                'mean_x_from_keypoints': np.nan,
                'mean_y_from_keypoints': np.nan,
                'local_variance_from_keypoints': np.nan,
                'normalized_variance_from_keypoints': np.nan
            }
            
            if img is None: # если пусто то пустые строки в метаданных, возможно это излишне
                results.append(record)
                continue
            
            img = cv2.resize(img, (img_w, img_h))
            keypoints = orb.detect(img, None)
            if not keypoints:
                results.append(record)
                continue
            
            coords = np.array([kp.pt for kp in keypoints])
            mean_coords = coords.mean(axis=0)
            local_var = coords.var(axis=0).sum()
            var_x = np.var(coords[:, 0])
            var_y = np.var(coords[:, 1])
            norm_var_x = var_x / max_disp_x
            norm_var_y = var_y / max_disp_y
            norm_var = (norm_var_x + norm_var_y) / 2
            
            kps_data = [{'x': kp.pt[0], 'y': kp.pt[1], 'size': kp.size} for kp in keypoints]
            
            record.update({
                'keypoints_by_orb': kps_data,
                'kp_count': len(keypoints),
                'mean_x_from_keypoints': mean_coords[0],
                'mean_y_from_keypoints': mean_coords[1],
                'local_variance_from_keypoints': local_var,
                'normalized_variance_from_keypoints': norm_var
            })
            
            results.append(record)

        columns_to_drop = [
        'keypoints_by_orb',
        'kp_count',
        'mean_x_from_keypoints',
        'mean_y_from_keypoints',
        'local_variance_from_keypoints',
        'normalized_variance_from_keypoints'
         ]
        self.metadata = self.metadata.drop(columns=[col for col in columns_to_drop if col in self.metadata.columns]) 
        
        # Объединение метаданных (один раз после цикла)
        df_results = pd.DataFrame(results)
        self.metadata = self.metadata.merge(df_results, on='filename', how='left')

    def calculate_color_metric(self): # 0 - 100, 100 - одинаковые цвета
        histograms = [] # Лист цветовых гистограмм
        metadata = [] # Лист метаданных
            
        for filename in self.filenames:
            path = os.path.join(self.images_dir, filename)
            img = cv2.imread(path)
            if img is None: # если пусто то пустые строки в метаданных, возможно это излишне
                histograms.append(np.zeros(self.bins * 3))
                metadata.append({
                    'filename': filename,
                    'mean_r': np.nan,
                    'mean_g': np.nan,
                    'mean_b': np.nan,
                    'avg_cosine_dist': np.nan})
                continue

            img = cv2.resize(img, (self.median_width, self.median_height)) # Нормализация изображений
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # На вход нужны изображения RGB

            hist_r = cv2.calcHist([img], [0], None, [self.bins], [0, 256]).flatten() # Гистограмма цветов по каналу r
            hist_g = cv2.calcHist([img], [1], None, [self.bins], [0, 256]).flatten() # Гистограмма цветов по каналу g
            hist_b = cv2.calcHist([img], [2], None, [self.bins], [0, 256]).flatten() # Гистограмма цветов по каналу b
        
            # Нормализация гистограммы
            hist_r /= (hist_r.sum() + self.epsilon) # 1e-6 - эмперическая константа, можно без нее
            hist_g /= (hist_g.sum() + self.epsilon)
            hist_b /= (hist_b.sum() + self.epsilon)
        
            # Средние цвета (центры масс)
            mean_r = np.sum(hist_r * np.arange(len(hist_r))) 
            mean_g = np.sum(hist_g * np.arange(len(hist_g)))
            mean_b = np.sum(hist_b * np.arange(len(hist_b)))

            hist = np.concatenate([hist_r, hist_g, hist_b]) # Нормализация полной гистограммы
            hist = hist / (hist.sum() + self.epsilon) 

            histograms.append(hist)

            # Добавление в metadata
            metadata.append({
                'filename': filename,
                'mean_r': mean_r,
                'mean_g': mean_g,
                'mean_b': mean_b,
                'avg_cosine_dist': np.nan})

        histograms = np.array(histograms)
        distances = pairwise_distances(histograms, metric='cosine') # Расчет косинусного расстояния по каждой из гистограмм [0; 1]

        for i in range(len(metadata)): # Для каждого изображения считаем среднее расстояние до других
            mask = np.ones(len(metadata), dtype=bool)
            mask[i] = False
            avg_dist = distances[i, mask].mean()
            metadata[i]['avg_cosine_dist'] = avg_dist

        mask = ~np.eye(len(distances), dtype=bool)
        avg_distance = distances[mask].mean()
        self.color_metric = round(float(100 * (1 - avg_distance)), 2) # Чем ниже разнообразие — тем выше метрика
        df_results = pd.DataFrame(metadata)
        self.metadata = self.metadata.merge(df_results, on='filename', how='left') # Обновление метаданных

    def calculate_noise_metric(self):
        transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor() 
        ]) # Модель SVD предобученна на данных размерах изображений

        brisque_scores = [] # значение Brisque по каждой из фотографий
        metadata = []

        for filename in self.filenames:
                path = os.path.join(self.images_dir, filename)
                img_bgr = cv2.imread(path)
                if img_bgr is None:
                    continue

                # BRISQUE требует перевод RGB-изображения в тензор
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img_rgb)
                tensor_img = transform(pil_img).unsqueeze(0)  # [1, C, H, W] # 
                    
                with torch.no_grad():
                    score = brisque(tensor_img) # Расчет метрики BRISQUE
                    brisque_scores.append(score.item()) 

                metadata.append({ # Добавляем метаданные
                    'filename': filename,
                    'brisque_score': score.item()
                    })

        # Нормализация BRISQUE в шкалу 0–100
        scores_np = np.array(brisque_scores)
        self.noise_metric = round(float((100 - np.clip(scores_np, 0, 100)).mean()), 2) # Расчет шумовой метрики
        df_results = pd.DataFrame(metadata)
        self.metadata = self.metadata.merge(df_results, on='filename', how='left') # Обновление метаданных

    def calculate_uniformity_metric(self):
        resize_shape = (128, 128) # Модель предобучена на таких размерах 

        def load_and_preprocess_image(path):
            img = image.load_img(path, target_size=resize_shape)
            img_array = image.img_to_array(img)
            return preprocess_input(img_array)

        if len(self.filenames) < 2: # Хотя бы 2 фотографии для расчета
            print("Недостаточно фотографий для расчета микшированной метрики. Добавьте еще фотографий")
            self.uniformity_metric = 0 # Брейк если меньше 2
            return

        features, original_images = [], []
        for fname in self.filenames:
            path = os.path.join(self.images_dir, fname)
            try:
                img_array = load_and_preprocess_image(path)
                img_batch = np.expand_dims(img_array, axis=0)
                feat = self.model.predict(img_batch, verbose=0)
                features.append(feat.squeeze())
                original_images.append(cv2.imread(path))
            except Exception as e:
                print(f"Ошибка при обработке {fname}: {e}") # может быть излишним

        features = np.array(features)
 
        if len(features) < 2: # для корректных расчетов нужно все-таки больше 2 референсов, 
            # можно попробовать динамизироваться
            print("Недостаточно изображений для расчета кластеров.")
            self.uniformity_metric = 0 # брейк если меньше 2 
            return

        # Корректируем количество кластеров и референсов
        n_clusters = min(self.min_n_clusters, len(features)) # Эмпрерически - 5 кластеров для поиска референсов

        # Кластеризация и выбор референсов
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(features)
        centers = kmeans.cluster_centers_

        ssim_scores = []

        for cluster_id in range(n_clusters):
            # Индексы картинок этого кластера
            cluster_indices = np.where(labels == cluster_id)[0]
            if len(cluster_indices) == 0:
                continue

            # Считаем расстояние до центра
            cluster_feats = features[cluster_indices]
            dists = np.linalg.norm(cluster_feats - centers[cluster_id], axis=1)

            # Находим эталон (ближайший к центру)
            ref_idx = cluster_indices[np.argmin(dists)]
            ref_img = cv2.resize(original_images[ref_idx], resize_shape)
            ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

            # Сравниваем все изображения в кластере с эталоном
            for idx in cluster_indices:
                try:
                    img_resized = cv2.resize(original_images[idx], resize_shape)
                    gray_img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
                    score = ssim(gray_img, ref_gray, data_range=255)
                    ssim_scores.append({
                        'filename': self.filenames[idx],
                        'cluster_id': cluster_id,
                        'ssim_score': score
                    })
                except Exception as e:
                    print(f"Ошибка при расчете SSIM для {self.filenames[idx]}: {e}")
                    ssim_scores.append({
                        'filename': self.filenames[idx],
                        'cluster_id': cluster_id,
                        'ssim_score': np.nan
                    })
        
        df_results = pd.DataFrame(ssim_scores)
        self.metadata = self.metadata.merge(df_results, on='filename', how='left')
        self.ssim_score = ssim_scores
        scores = [d['ssim_score'] for d in ssim_scores]  # список float
        self.uniformity_metric = round(float(np.nanmean(scores)) * 100, 2)
    
    def count_by_metrics(self): # Простое округление часто будет давать ошибку на 1 от общего 
        # количества, поэтому только так:
        metrics = {
           'geometric': self.geometric_metric,
            'color': self.color_metric,
            'noise': self.noise_metric,
            'uniformity': self.uniformity_metric
        }

        total = sum(metrics.values())
        if total == 0:
            # Если все метрики нулевые — делим поровну
            base = self.n_samples_to_augument // 4
            remainder = self.n_samples_to_augument % 4
            self.n_by_geometric_metric = base + (1 if remainder > 0 else 0)
            self.n_by_color_metric = base + (1 if remainder > 1 else 0)
            self.n_by_noise_metric = base + (1 if remainder > 2 else 0)
            self.n_by_uniformity_metric = base
            return

        # Вычисляем доли и остатки
        shares = {k: (v / total * self.n_samples_to_augument) for k, v in metrics.items()}
        floors = {k: int(np.floor(val)) for k, val in shares.items()}
        remainders = {k: shares[k] - floors[k] for k in metrics}

        # Сколько еще нужно добавить единиц
        total_assigned = sum(floors.values())
        remaining = self.n_samples_to_augument - total_assigned

        # Добавим оставшиеся единицы к самым большим остаткам
        sorted_keys = sorted(remainders, key=remainders.get, reverse=True)
        for i in range(remaining):
            floors[sorted_keys[i]] += 1

        # Присваиваем результаты
        self.n_by_geometric_metric = floors['geometric']
        self.n_by_color_metric = floors['color']
        self.n_by_noise_metric = floors['noise']
        self.n_by_uniformity_metric = floors['uniformity']
    
    def geometric_augmentation(self): 
        os.makedirs(self.augumentation_dir, exist_ok=True)
        
        if getattr(self, 'is_labeled', False):
            os.makedirs(self.dir_images_with_labels, exist_ok=True)
            os.makedirs(self.dir_with_new_labels, exist_ok=True)
        
        target_total = self.n_by_geometric_metric
        
        valid_metadata = self.metadata[self.metadata['normalized_variance_from_keypoints'].notnull()].copy()
        valid_metadata.sort_values('normalized_variance_from_keypoints', ascending=True, inplace=True)
        sorted_filenames = valid_metadata['filename'].tolist() 
        
 
        # Прочитать bbox из файла в формате YOLO: class x_center y_center width height (в относительных координатах() 
        def read_bbox(txt_path):
            bboxes = []
            if not os.path.isfile(txt_path):
                return bboxes
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_c, y_c, w, h = parts
                        bboxes.append([cls, float(x_c), float(y_c), float(w), float(h)])
            return bboxes

        # Сохранить bbox в файл (YOLO формат)
        def save_bbox(txt_path, bboxes):
            with open(txt_path, 'w') as f:
                for bbox in bboxes:
                    cls, x_c, y_c, w, h = bbox
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        # Нарисовать bbox на изображении (прямоугольники) для визуализации
        def draw_bboxes(img, bboxes):
            h, w = img.shape[:2]
            for bbox in bboxes:
                _, x_c, y_c, bw, bh = bbox
                x1 = int((x_c - bw / 2) * w)
                y1 = int((y_c - bh / 2) * h)
                x2 = int((x_c + bw / 2) * w)
                y2 = int((y_c + bh / 2) * h)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Масштабирование bbox при zoom
        def zoom_bboxes(bboxes, zoom_factor):
            bboxes_new = []
            # Центрирование: координаты смещаются в зависимости от zoom, т.к. zoom по центру
            # В данном случае при zoom >1 bbox относительно центра уменьшается, при zoom <1 bbox увеличивается
            for bbox in bboxes:
                cls, x_c, y_c, w, h = bbox
                x_c_new = 0.5 + (x_c - 0.5) * zoom_factor
                y_c_new = 0.5 + (y_c - 0.5) * zoom_factor
                w_new = w * zoom_factor
                h_new = h * zoom_factor
                # Обрезаем bbox по границам [0,1]
                x_c_new = max(0.0, min(1.0, x_c_new))
                y_c_new = max(0.0, min(1.0, y_c_new))
                w_new = max(0.0, min(1.0, w_new))
                h_new = max(0.0, min(1.0, h_new))
                bboxes_new.append([cls, x_c_new, y_c_new, w_new, h_new])
            return bboxes_new

        # Смещение bbox при crop и масштабирование обратно к размеру исходного изображения
        def crop_bboxes(bboxes, top, left, ch, cw, orig_h, orig_w):
            bboxes_new = []
            for bbox in bboxes:
                cls, x_c, y_c, w, h = bbox
                # Переводим в пиксели
                x_center_pix = x_c * orig_w
                y_center_pix = y_c * orig_h
                w_pix = w * orig_w
                h_pix = h * orig_h

                x1 = x_center_pix - w_pix / 2
                y1 = y_center_pix - h_pix / 2
                x2 = x_center_pix + w_pix / 2
                y2 = y_center_pix + h_pix / 2

                # Смещение относительно кропа
                x1_new = x1 - left
                y1_new = y1 - top
                x2_new = x2 - left
                y2_new = y2 - top

                # Ограничение по размеру кропа
                x1_new = max(0, min(cw, x1_new))
                y1_new = max(0, min(ch, y1_new))
                x2_new = max(0, min(cw, x2_new))
                y2_new = max(0, min(ch, y2_new))

                # Проверка валидности bbox после кропа
                w_new_pix = x2_new - x1_new
                h_new_pix = y2_new - y1_new
                if w_new_pix <= 0 or h_new_pix <= 0:
                    continue  # bbox полностью вырезался

                # Переводим обратно в относительные координаты (после ресайза кропа обратно к orig_w, orig_h)
                x_c_new = (x1_new + w_new_pix / 2) / cw
                y_c_new = (y1_new + h_new_pix / 2) / ch
                w_new = w_new_pix / cw
                h_new = h_new_pix / ch

                # Так как изображение ресайзится обратно к исходному размеру (orig_w, orig_h),
                # bbox остаётся в относительных координатах по ресайзеному изображению
                bboxes_new.append([cls, x_c_new, y_c_new, w_new, h_new])

            return bboxes_new

        def zoom_transform(img, zoom_factor):
            h, w = img.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 0, zoom_factor)
            zoomed = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            return zoomed
        
        def random_crop(img):
            h, w, _ = img.shape
            
            # Случайно выбираем размер кропа в допустимом диапазоне
            crop_size_ratio = random.uniform(self.min_crop_ratio, self.max_crop_ratio)
            ch, cw = int(h * crop_size_ratio), int(w * crop_size_ratio)
            
            # Если кроп равен размеру — возвращаем оригинал
            if ch == h or cw == w:
                return img, 0, 0, h, w
            
            # Выбираем случайное смещение кропа
            top = random.randint(0, h - ch)
            left = random.randint(0, w - cw)
            
            # Обрезка
            cropped = img[top:top + ch, left:left + cw]
            
            # Возвращаем к исходному размеру, чтобы совместимость осталась
            resized = cv2.resize(cropped, (w, h))
            
            return resized, top, left, ch, cw
        
        def flip_transform(img):
            return cv2.flip(img, 1)  # горизонтальный флип
        
        def flip_bboxes(bboxes):
            bboxes_new = []
            for bbox in bboxes:
                 cls, x_c, y_c, w, h = bbox
                 x_c_new = 1.0 - x_c   # только центр по X отражается
                 bboxes_new.append([cls, x_c_new, y_c, w, h])
            return bboxes_new

        top_filenames = sorted_filenames
        final_filenames = []
        idx = 0
        while len(final_filenames) < target_total:
            final_filenames.append(top_filenames[idx % len(top_filenames)])
            idx += 1

        aug_types = ['zoom', 'crop', 'flip']
        method_to_filenames = {'zoom': [], 'crop': [], 'flip':[]}
        for i, filename in enumerate(final_filenames):
            aug_type = aug_types[i % 3]
            method_to_filenames[aug_type].append(filename)

        for aug_type in aug_types:
            filenames = method_to_filenames[aug_type]

            for filename in filenames:
                img_path = os.path.join(self.images_dir, filename)
                img = cv2.imread(img_path)
                if img is None or img.size == 0:
                    continue

                try:
                    if aug_type == 'zoom':
                        zoom_factor = random.uniform(self.min_zoom_factor, self.max_zoom_factor)
                        augmented = zoom_transform(img, zoom_factor)
                    elif aug_type == 'crop':
                        augmented, top, left, ch, cw = random_crop(img)
                    elif aug_type == 'flip':
                        augmented = flip_transform(img)
                    else:
                        continue

                    base_name = os.path.splitext(filename)[0]
                    new_name = f"{base_name}_{aug_type}_{uuid.uuid4().hex[:6]}.jpg"

                    if getattr(self, 'is_labeled', False):
                        txt_path = os.path.join(self.labels_dir, base_name + '.txt')
                        bboxes = read_bbox(txt_path)
 
                        if aug_type == 'zoom':
                            bboxes_new = zoom_bboxes(bboxes, zoom_factor)
                        elif aug_type == 'crop':
                            orig_h, orig_w = img.shape[:2]
                            bboxes_new = crop_bboxes(bboxes, top, left, ch, cw, orig_h, orig_w)
                        elif aug_type == 'flip':
                            bboxes_new = flip_bboxes(bboxes)
                        else:
                            bboxes_new = bboxes

                        img_for_draw = augmented.copy()
                        draw_bboxes(img_for_draw, bboxes_new)
                        cv2.imwrite(os.path.join(self.dir_images_with_labels, new_name), img_for_draw)

                        new_txt_path = os.path.join(self.dir_with_new_labels, os.path.splitext(new_name)[0] + '.txt')
                        save_bbox(new_txt_path, bboxes_new)
                        cv2.imwrite(os.path.join(self.augumentation_dir, new_name), augmented)
                    else:
                        cv2.imwrite(os.path.join(self.augumentation_dir, new_name), augmented)

                except Exception as e:
                    print(f"Ошибка при {aug_type}-аугументации {filename}: {e}")
                    continue

    def color_augmentation(self):
        os.makedirs(self.augumentation_dir, exist_ok=True)
        target_total = self.n_by_color_metric

        # Фильтруем метаданные и сортируем по убыванию hist_entropy, используя таким образом в первую 
        # очередь фотографии с многообразными цветами 
        valid_metadata = self.metadata[self.metadata['avg_cosine_dist'].notnull()].copy()
        valid_metadata.sort_values('avg_cosine_dist', ascending=True, inplace=True)

        # Фильтрация по фактическому наличию изображений (возможно это лишнее)
        sorted_filenames = []
        for filename in valid_metadata['filename']:
            img_path = os.path.join(self.images_dir, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    sorted_filenames.append(filename)

        def apply_color_jitter(img): # №1 джиттеринг - случайное изменение цветов изображения
            h_shift = random.randint(self.min_h_shift, self.max_h_shift)   # Случайный сдвиг hue
            s_scale = random.uniform(self.min_s_scale, self.max_s_scale)  # Насыщенность
            v_scale = random.uniform(self.min_v_scale, self.max_v_scale)  # Яркость
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32) 
            h, s, v = cv2.split(hsv)
            h = (h + random.uniform(-h_shift, h_shift)) % 180
            s *= random.uniform(1.0, s_scale)
            v *= random.uniform(1.0, v_scale)
            s = np.clip(s, 0, 255)
            v = np.clip(v, 0, 255)
            hsv_aug = cv2.merge([h, s, v]).astype(np.uint8)
            return cv2.cvtColor(hsv_aug, cv2.COLOR_HSV2BGR)

        def apply_posterize_effect(img): # №2 - Пастеризация изображений - более резким/размытым 
            # по цветовой палитре
            # bits ∈ [1, self.base_bits]; чем меньше, тем сильнее эффект

            bits = random.randint(self.min_bits, self.max_bits)  # Случайное значение для каждого вызова
            shift = self.base_bits - bits
            return ((img >> shift) << shift).astype(np.uint8)

        def apply_channel_shuffle(img): # №3 - Перемешивание цветов изображения (Chanel Shuffle)
            # img: H x W x 3
            channels = cv2.split(img)  # на H, S, V
            indices = np.random.permutation(3)  # перемешиваем [0, 1, 2]
            shuffled = [channels[i] for i in indices] 
            return cv2.merge(shuffled) # изображение с перемешанными цветами

        aug_methods = [
            ('jitter', apply_color_jitter),
            ('poster', apply_posterize_effect),
            ('shuffle', apply_channel_shuffle)
        ]

        # Создаём список файлов с нужной длиной (с повторами, если нужно)
        filenames_for_aug = []
        idx = 0
        while len(filenames_for_aug) < target_total:
            filenames_for_aug.append(sorted_filenames[idx % len(sorted_filenames)])
            idx += 1

        # Распределяем циклически по типам аугментации
        method_to_filenames = {method: [] for method, _ in aug_methods}
        for i, filename in enumerate(filenames_for_aug):
            aug_type = aug_methods[i % 3][0]
            method_to_filenames[aug_type].append(filename)

        for aug_type, aug_fn in aug_methods:
            for filename in method_to_filenames[aug_type]:
                img_path = os.path.join(self.images_dir, filename)
                img = cv2.imread(img_path)

                try:
                    augmented = aug_fn(img)
                    base_name = os.path.splitext(filename)[0]
                    new_name = f"{base_name}_{aug_type}_{uuid.uuid4().hex[:6]}.jpg"
                    cv2.imwrite(os.path.join(self.augumentation_dir, new_name), augmented)

                    # Если разметка включена
                    if self.is_labeled:
                        label_file = os.path.join(self.labels_dir, f"{base_name}.txt")
                        if os.path.exists(label_file):
                            with open(label_file, 'r') as lf:
                                lines = lf.readlines()

                            # Копируем разметку под новым именем
                            new_label_file = os.path.join(self.dir_with_new_labels, f"{os.path.splitext(new_name)[0]}.txt")
                            with open(new_label_file, 'w') as nlf:
                                nlf.writelines(lines)

                            # Рисуем bbox
                            img_with_bbox = augmented.copy()
                            h_img, w_img = img_with_bbox.shape[:2]
                            for line in lines:
                                cls, x_center, y_center, width, height = map(float, line.strip().split())
                                x1 = int((x_center - width / 2) * w_img)
                                y1 = int((y_center - height / 2) * h_img)
                                x2 = int((x_center + width / 2) * w_img)
                                y2 = int((y_center + height / 2) * h_img)
                                cv2.rectangle(img_with_bbox, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(img_with_bbox, str(int(cls)), (x1, y1 - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                            cv2.imwrite(os.path.join(self.dir_images_with_labels, new_name), img_with_bbox)

                except Exception as e:
                    print(f"Ошибка при {aug_type}-аугументации {filename}: {e}")
                    continue
        
    def noise_augmentation(self):
        os.makedirs(self.augumentation_dir, exist_ok=True)
        target_total = self.n_by_noise_metric

        # Сортировка по качеству — от лучшего к худшему (низкий brisque = высокое качество, 
        # обратная ситуация прошлым аугументациям)
        metadata_sorted = self.metadata.dropna(subset=['brisque_score']).sort_values(by='brisque_score', 
                                                                                     ascending=True)
        valid_filenames = []

        # Фильтрация по фактическому наличию изображений
        for filename in metadata_sorted['filename']: 
            img_path = os.path.join(self.images_dir, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    valid_filenames.append(filename)

        # Шум соль/перец
        def apply_salt_pepper_noise(img):
            amount = random.uniform(self.min_amount, self.max_amount) # Нужно подумать, можно ли это вытащить глобально
            noisy = img.copy()
            num_salt = np.ceil(amount * img.size * self.salt_vs_pepper).astype(int)
            num_pepper = np.ceil(amount * img.size * (1.0 - self.salt_vs_pepper)).astype(int)

            coords = [np.random.randint(0, i, num_salt) for i in img.shape[:2]]
            noisy[coords[0], coords[1]] = 255
            coords = [np.random.randint(0, i, num_pepper) for i in img.shape[:2]]
            noisy[coords[0], coords[1]] = 0
            return noisy

        # Гауссовский шум
        def apply_gaussian_noise(img):
            gauss = np.random.normal(self.mean_gaussian_noise, self.std_gaussian_noise, img.shape).astype(np.float32)
            noisy = img.astype(np.float32) + gauss
            return np.clip(noisy, 0, 255).astype(np.uint8)

        # Пуассоновский шум
        def apply_poisson_noise(img):
            noisy = np.random.poisson(img.astype(np.float32)).astype(np.float32)
            return np.clip(noisy, 0, 255).astype(np.uint8)

        aug_methods = [
            ('saltpepper', apply_salt_pepper_noise),
            ('gaussian', apply_gaussian_noise),
            ('poisson', apply_poisson_noise)
        ]

        # Распределение количества на каждый метод
        counts = [target_total // len(aug_methods)] * len(aug_methods)
        for i in range(target_total % len(aug_methods)):
            counts[i] += 1

        for (aug_type, aug_fn), count_limit in zip(aug_methods, counts):
            # Берём лучшие изображения (если достаточно — сэмпл без повторов, иначе — цикл)
            if count_limit <= len(valid_filenames):
                filenames_iter = iter(random.sample(valid_filenames, count_limit))
            else:
                filenames_iter = cycle(valid_filenames)

            count = 0
            while count < count_limit:
                try:
                    filename = next(filenames_iter)
                    img_path = os.path.join(self.images_dir, filename)
                    img = cv2.imread(img_path)

                    augmented = aug_fn(img)
                    base_name = os.path.splitext(filename)[0]
                    new_name = f"{base_name}_{aug_type}_{uuid.uuid4().hex[:6]}.jpg"
                    cv2.imwrite(os.path.join(self.augumentation_dir, new_name), augmented)

                    # Если есть разметка
                    if self.is_labeled:
                        label_file = os.path.join(self.labels_dir, base_name + '.txt')
                        if os.path.exists(label_file):
                            # Копируем разметку
                            with open(label_file, 'r', encoding='utf-8') as f:
                                label_data = f.read()

                            # Рисуем bbox
                            labeled_img = augmented.copy()
                            for line in label_data.strip().split('\n'):
                                cls, x_center, y_center, w, h = map(float, line.split())
                                img_h, img_w = labeled_img.shape[:2]
                                x_center_abs = int(x_center * img_w)
                                y_center_abs = int(y_center * img_h)
                                w_abs = int(w * img_w)
                                h_abs = int(h * img_h)
                                x1 = int(x_center_abs - w_abs / 2)
                                y1 = int(y_center_abs - h_abs / 2)
                                x2 = int(x_center_abs + w_abs / 2)
                                y2 = int(y_center_abs + h_abs / 2)
                                cv2.rectangle(labeled_img, (x1, y1), (x2, y2), (0, 255, 0), 2)

                            # Сохраняем размеченное изображение
                            labeled_img_path = os.path.join(self.dir_images_with_labels, new_name)
                            cv2.imwrite(labeled_img_path, labeled_img)

                            # Дублируем txt с тем же содержимым под новым именем
                            new_label_file = os.path.join(self.dir_with_new_labels,
                                                      os.path.splitext(new_name)[0] + '.txt')
                            with open(new_label_file, 'w', encoding='utf-8') as f:
                                f.write(label_data)

                    count += 1

                except Exception as e:
                    print(f"Ошибка при {aug_type}-аугментации {filename}: {e}")
                    continue
    
    def uniformity_augmentation(self):

        os.makedirs(self.augumentation_dir, exist_ok=True)

        if getattr(self, 'is_labeled', False):
            os.makedirs(self.dir_images_with_labels, exist_ok=True)
            os.makedirs(self.dir_with_new_labels, exist_ok=True)

        target_total = self.n_by_uniformity_metric

        # Отбор по ssim_score (чем больше - тем лучше, идут в первую очередь)
        valid_metadata = self.metadata[self.metadata['ssim_score'].notnull()].copy()
        valid_metadata.sort_values('ssim_score', ascending=False, inplace=True)

        # Фильтрация существующих изображений
        valid_filenames = []
        for filename in valid_metadata['filename']:
            img_path = os.path.join(self.images_dir, filename)
            if os.path.isfile(img_path):
                img = cv2.imread(img_path)
                if img is not None and img.size > 0:
                    valid_filenames.append(filename)

        def mix_alpha_blend(img1, img2, alpha=None): # №1 - Alpha Blending - смешивание одного вторым на долю по альфа
            if alpha is None:
                alpha = random.uniform(self.min_alpha, self.max_alpha)
            blended = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)
            return blended

        def mix_channel_blend(img1, img2): # №2 - Channel Blending
            # Составление нового изображения из каналов разных изображений
            img1 = img1.copy()
            img2 = img2.copy()
            r = img1[:, :, 2]
            g = img2[:, :, 1]
            b = img2[:, :, 0]
            return cv2.merge([b, g, r])
        
        def mix_cutmix(img1, img2, bboxes1=None, bboxes2=None): # №3 - Cutmix
            h, w, _ = img1.shape
    
            # Размер вставки
            cut_w = int(w * self.alpha_for_cutmix)
            cut_h = int(h * self.alpha_for_cutmix)

            # Случайная позиция вставки
            x1 = random.randint(0, w - cut_w)
            y1 = random.randint(0, h - cut_h)

            # Вырезаем кусок из img2
            img2_resized = cv2.resize(img2, (w, h))
            patch = img2_resized[y1:y1 + cut_h, x1:x1 + cut_w]

            # Вставляем в img1
            mixed = img1.copy()
            mixed[y1:y1 + cut_h, x1:x1 + cut_w] = patch

            # Если разметка не передана → вернуть только картинку
            if bboxes1 is None or bboxes2 is None:
                return mixed

            # --- ниже только если работаем с bbox ---
            new_bboxes1 = []
            for cls, x_c, y_c, bw, bh in bboxes1:
                xc_pix = x_c * w
                yc_pix = y_c * h
                bw_pix = bw * w
                bh_pix = bh * h
                x0 = xc_pix - bw_pix / 2
                y0 = yc_pix - bh_pix / 2
                x1b = xc_pix + bw_pix / 2
                y1b = yc_pix + bh_pix / 2
                
                # Проверка пересечения с вставкой (x1, y1, cut_w, cut_h)
                if x1b <= x1 or x0 >= x1 + cut_w or y1b <= y1 or y0 >= y1 + cut_h:
                    # bbox полностью вне вставки → оставляем как есть
                    new_bboxes1.append([cls, x_c, y_c, bw, bh])
                else:
                    # Если bbox частично пересекается вставку → корректируем
                    x0_new = max(x0, 0)
                    y0_new = max(y0, 0)
                    x1_new = min(x1b, w)
                    y1_new = min(y1b, h)
                    bw_new = (x1_new - x0_new) / w
                    bh_new = (y1_new - y0_new) / h
                    xc_new = (x0_new + x1_new) / 2 / w
                    yc_new = (y0_new + y1_new) / 2 / h
                    new_bboxes1.append([cls, xc_new, yc_new, bw_new, bh_new])
            
            new_bboxes = []
            for cls, x_c, y_c, bw, bh in bboxes2:
                # перевод в пиксели
                xc_pix = x_c * w
                yc_pix = y_c * h
                bw_pix = bw * w
                bh_pix = bh * h
                x0 = xc_pix - bw_pix / 2
                y0 = yc_pix - bh_pix / 2
                x1b = xc_pix + bw_pix / 2
                y1b = yc_pix + bh_pix / 2

                # Проверка пересечения с вставкой
                if x1b < x1 or y1b < y1 or x0 > x1 + cut_w or y0 > y1 + cut_h:
                    continue

                # Обрезка bbox
                x0_new = max(x0, x1)
                y0_new = max(y0, y1)
                x1_new = min(x1b, x1 + cut_w)
                y1_new = min(y1b, y1 + cut_h)

                # назад в нормализованные
                bw_new = (x1_new - x0_new) / w
                bh_new = (y1_new - y0_new) / h
                xc_new = (x0_new + x1_new) / 2 / w
                yc_new = (y0_new + y1_new) / 2 / h

                new_bboxes.append([cls, xc_new, yc_new, bw_new, bh_new])

            return mixed, bboxes1 + new_bboxes


        mix_methods = [
            ('blend', mix_alpha_blend),
            ('channels', mix_channel_blend),
            ('cutmix', lambda img1, img2, b1=None, b2=None: mix_cutmix(img1, img2, b1, b2))
        ]

        # Подготовка пар изображений
        mixed_pairs = []
        idx = 0
        while len(mixed_pairs) < target_total:
            img1_name = valid_filenames[idx % len(valid_filenames)]
            img2_name = valid_filenames[(idx + 1) % len(valid_filenames)]
            mixed_pairs.append((img1_name, img2_name))
            idx += 1

        method_to_pairs = {name: [] for name, _ in mix_methods}
        for i, pair in enumerate(mixed_pairs):
            method = mix_methods[i % len(mix_methods)][0]
            method_to_pairs[method].append(pair)

        # Чтение и сохранение bbox
        def read_bbox(txt_path):
            bboxes = []
            if not os.path.isfile(txt_path):
                return bboxes
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cls, x_c, y_c, w, h = parts
                        bboxes.append([cls, float(x_c), float(y_c), float(w), float(h)])
            return bboxes

        def save_bbox(txt_path, bboxes):
            with open(txt_path, 'w') as f:
                for bbox in bboxes:
                    cls, x_c, y_c, w, h = bbox
                    f.write(f"{cls} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")

        for method_name, mix_fn in mix_methods:
            for fname1, fname2 in method_to_pairs[method_name]:
                img1_path = os.path.join(self.images_dir, fname1)
                img2_path = os.path.join(self.images_dir, fname2)
                img1 = cv2.imread(img1_path)
                img2 = cv2.imread(img2_path)
            
                if img1 is None or img2 is None:
                    continue
                if img1.shape != img2.shape:
                    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

                base_name1 = os.path.splitext(fname1)[0]
                base_name2 = os.path.splitext(fname2)[0]
                # Генерация нового имени
                new_name = f"{base_name1}_{base_name2}_{method_name}_{uuid.uuid4().hex[:6]}.jpg"
                
                try:
                    if method_name == 'cutmix' and getattr(self, 'is_labeled', False):
                        bboxes1 = read_bbox(os.path.join(self.labels_dir, base_name1 + '.txt'))
                        bboxes2 = read_bbox(os.path.join(self.labels_dir, base_name2 + '.txt'))
                        mixed, all_bboxes = mix_cutmix(img1, img2, bboxes1, bboxes2)
                    elif getattr(self, 'is_labeled', False):
                        # для Blend и Channels просто берем исходные bbox
                        bboxes1 = read_bbox(os.path.join(self.labels_dir, base_name1 + '.txt'))
                        bboxes2 = read_bbox(os.path.join(self.labels_dir, base_name2 + '.txt'))
                        mixed = mix_fn(img1, img2)
                        all_bboxes = bboxes1 + bboxes2
                    else:
                        # если нет разметки
                        mixed = mix_fn(img1, img2)
                        all_bboxes = None
                    
                    # Сохраняем изображение в общий каталог
                    cv2.imwrite(os.path.join(self.augumentation_dir, new_name), mixed)
                    
                    # Если есть разметка — сохраняем изображение с bbox и txt
                    
                    if getattr(self, 'is_labeled', False):
                        img_for_draw = mixed.copy()
                        h, w = img_for_draw.shape[:2]
                        for bbox in all_bboxes:
                            _, x_c, y_c, bw, bh = bbox
                            x1 = int((x_c - bw / 2) * w)
                            y1 = int((y_c - bh / 2) * h)
                            x2 = int((x_c + bw / 2) * w)
                            y2 = int((y_c + bh / 2) * h)
                            cv2.rectangle(img_for_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.imwrite(os.path.join(self.dir_images_with_labels, new_name), img_for_draw)
                        
                        new_txt_path = os.path.join(self.dir_with_new_labels, os.path.splitext(new_name)[0] + '.txt')
                        save_bbox(new_txt_path, all_bboxes)
                    
                except Exception as e:
                    print(f"Ошибка при {method_name}-аугументации {fname1}, {fname2}: {e}")
    
    def run_pipeline(self): # Как выглядиь пайплайн, можно переделать наверное как-то, у меня просто последовательно выполняется
        
        if self.is_error == 1:
            self.output_name = 'Указано меньшее допустимого количества изображений на аугументацию'
            return
        
        self.calculate_normalized_images_size()

        if self.is_error == 1:
            self.output_name = 'В директории нет поддерживаемых изображений'
            return
        
        if self.is_labeled:
            self.check_augmentation_label_consistency()
            
        if self.is_error == 1:
            self.output_name = 'Не хватает изображений или разметки, проверьте указанные директории'
            return
            
        self.calculate_geometric_metric()
        self.calculate_color_metric()
        self.calculate_noise_metric()
        self.calculate_uniformity_metric()
        self.count_by_metrics()
                
        if self.n_by_geometric_metric > 0:
            self.geometric_augmentation()
        if self.n_by_color_metric > 0:
            self.color_augmentation()
        if self.n_by_noise_metric > 0:
            self.noise_augmentation()
        if self.n_by_uniformity_metric > 0:
            self.uniformity_augmentation()
        self.output_name = 'Аугментация завершена. Проверьте указанную папку'
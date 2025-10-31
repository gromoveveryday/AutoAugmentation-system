from flask import Flask, render_template, request, send_file
from augmentation_images import autoaugmentation_images
from augmentation_timeseries import AutoAugmentationTimeseries
import pandas as pd
import os
import tempfile

app = Flask(__name__)

timegan_augmenter = None  # сохранённый объект AutoAugmentationTimeseries

TEMP_INPUT_PATH = os.path.join(tempfile.gettempdir(), "timeseries_input.json")
TEMP_MODIFIED_PATH = os.path.join(tempfile.gettempdir(), "timeseries_modified.json")


def _df_to_index_lists(df: pd.DataFrame):
    out = {}
    for idx in df.index:
        # приводим к списку и заменяем NaN на None
        vals = [None if pd.isna(x) else (float(x) if (pd.notna(x) and not isinstance(x, str)) else x) for x in df.loc[idx].tolist()]
        out[str(idx)] = vals
    cols = [str(c) for c in df.columns.tolist()]
    idxs = [str(i) for i in df.index.tolist()]
    return out, cols, idxs


def save_initial_state(csv_path):
    augmenter = AutoAugmentationTimeseries(csv_path)
    df_input = augmenter.df_input
    stats_html = pd.DataFrame.from_dict(augmenter.calculate_statistics(df_input), orient='index') \
        .to_html(classes="table table-sm", border=0, na_rep="NaN")

    # сохраняем во временные файлы
    df_input.to_json(TEMP_INPUT_PATH,  orient="split")
    df_input.to_json(TEMP_MODIFIED_PATH,  orient="split")

    # JSON-представление для JS (index -> list of values), колонки и список индексов
    df_json, cols, idxs = _df_to_index_lists(df_input)

    result = {
        "df_html": df_input.to_html(classes="table table-sm", border=0, na_rep="NaN"),
        "stats_html": stats_html,
        "modified_html": df_input.to_html(classes="table table-sm", border=0, na_rep="NaN"),
        "modified_stats_html": stats_html,
        "modified_method": "initial",
        "csv_path": csv_path,
        "df_json": df_json,
        "modified_json": df_json,
        "columns": cols,
        "indices": idxs
    }
    return result


@app.route("/timeseries", methods=["GET", "POST"])
def augmentation_timeseries():
    error = None
    result = {}

    if request.method == "POST":
        csv_path = request.form.get("csv_path", "").strip()
        if not csv_path:
            error = "CSV файл не указан."
        elif not os.path.exists(csv_path):
            error = f"Файл {csv_path} не найден."
        else:
            try:
                result = save_initial_state(csv_path)
            except Exception as e:
                error = str(e)

    return render_template("augmentation_timeseries.html", result=result, error=error)
@app.route("/timeseries_action", methods=["POST"])
def timeseries_action():
    global timegan_augmenter

    error = None
    result = {}
    try:
        df_input = pd.read_json(TEMP_INPUT_PATH,  orient="split")
        df_modified = pd.read_json(TEMP_MODIFIED_PATH,  orient="split")

        action = request.form.get("action")
        method = request.form.get("method", "linear")
        pred_len = int(request.form.get("pred_len", 30))


        # Если глобального объекта нет, создаём новый
        if timegan_augmenter is None:
            timegan_augmenter = AutoAugmentationTimeseries(df_input)
            timegan_augmenter.df_updated = df_modified.copy()
        else:
            # обновляем данные, но сохраняем уже обученную модель
            timegan_augmenter.df_input = df_input
            timegan_augmenter.df_updated = df_modified.copy()
            timegan_augmenter.pred_len = pred_len

        # выполняем действие (интерполяция, экстраполяция и т.д.)
        df_modified_new, html_dict = timegan_augmenter.apply_action(
            timegan_augmenter.df_input,
            timegan_augmenter.df_updated,
            action,
            method
        )
        # обновляем внутреннее состояние
        timegan_augmenter.df_updated = df_modified_new.copy()

        # сохраняем изменённый датафрейм во временный файл
        df_modified_new.to_json(TEMP_MODIFIED_PATH,  orient="split")

        # статистики
        stats_df = pd.DataFrame.from_dict(
            timegan_augmenter.calculate_statistics(timegan_augmenter.df_updated), orient='index'
        )
        html_dict["df_modified_html"] = timegan_augmenter.df_updated.to_html(
            classes="dataframe table table-sm", border=0, na_rep="NaN"
        )
        html_dict["stats_modified_html"] = stats_df.to_html(
            classes="dataframe table table-sm", border=0, na_rep="NaN"
        )

        # формируем JSON для фронтенда
        stats_html_input = pd.DataFrame.from_dict(
            timegan_augmenter.calculate_statistics(df_input), orient='index'
        ).to_html(classes="table table-sm", border=0, na_rep="NaN")

        df_json, cols, idxs = _df_to_index_lists(df_input)
        modified_json, modified_cols, modified_idxs = _df_to_index_lists(df_modified_new)

        result = {
            "df_html": df_input.to_html(classes="table table-sm", border=0, na_rep="NaN"),
            "stats_html": stats_html_input,
            "modified_html": html_dict["df_modified_html"],
            "modified_stats_html": html_dict["stats_modified_html"],
            "modified_method": action,
            "csv_path": "",
            "df_json": df_json,
            "modified_json": modified_json,
            "columns": modified_cols,
            "indices": modified_idxs
        }

    except Exception as e:
        error = str(e)

    return render_template("augmentation_timeseries.html", result=result, error=error)

@app.route("/timeseries_reset", methods=["POST"])
def reset_data():
    error = None
    result = {}
    try:
        df_input = pd.read_json(TEMP_INPUT_PATH,  orient="split")
        df_modified = df_input.copy()
        df_modified.to_json(TEMP_MODIFIED_PATH,  orient="split")

        augmenter = AutoAugmentationTimeseries(df_modified)
        stats_html = pd.DataFrame.from_dict(augmenter.calculate_statistics(df_modified), orient='index') \
            .to_html(classes="table table-sm", border=0, na_rep="NaN")

        df_json, cols, idxs = _df_to_index_lists(df_modified)

        result = {
            "df_html": df_modified.to_html(classes="table table-sm", border=0, na_rep="NaN"),
            "stats_html": stats_html,
            "modified_html": df_modified.to_html(classes="table table-sm", border=0, na_rep="NaN"),
            "modified_stats_html": stats_html,
            "modified_method": "undo",
            "df_json": df_json,
            "modified_json": df_json,
            "columns": cols,
            "indices": idxs
        }
    except Exception as e:
        error = str(e)
    return render_template("augmentation_timeseries.html", result=result, error=error)


@app.route("/")
def main():
    return render_template("main.html")


@app.route("/images", methods=["GET", "POST"])
def augmentation_images_route():
    global output_name, zip_file_path

    output_name = None
    zip_file_path = None
    form_data = request.form.to_dict() if request.method == "POST" else {}

    if request.method == "POST":
        uploaded_images = request.files.getlist("images")
        uploaded_labels = request.files.getlist("labels") if "labels" in request.files else []
        n_samples = int(form_data.get("n_samples", 0))
        max_nfeatures_to_orb = str(form_data.get("max_nfeatures_to_orb", "По умолчанию"))
        is_labeled = form_data.get("is_labeled", "Нет") == "Да"
        bins = int(form_data.get("bins", 32))
        min_n_clusters = int(form_data.get("min_n_clusters", 5))
        min_crop_ratio = float(form_data.get("min_crop_ratio", 0.8))
        max_crop_ratio = float(form_data.get("max_crop_ratio", 0.95))
        min_zoom_factor = float(form_data.get("min_zoom_factor", 1.05))
        max_zoom_factor = float(form_data.get("max_zoom_factor", 1.4))
        min_h_shift = int(form_data.get("min_h_shift", 5))
        max_h_shift = int(form_data.get("max_h_shift", 30))
        min_s_scale = float(form_data.get("min_s_scale", 0.8))
        max_s_scale = float(form_data.get("max_s_scale", 1.5))
        min_v_scale = float(form_data.get("min_v_scale", 0.7))
        max_v_scale = float(form_data.get("max_v_scale", 1.4))
        min_bits = int(form_data.get("min_bits", 3))
        max_bits = int(form_data.get("max_bits", 6))
        base_bits = int(form_data.get("base_bits", 8))
        salt_vs_pepper = float(form_data.get("salt_vs_pepper", 0.5))
        mean_gaussian_noise = int(form_data.get("mean_gaussian_noise", 0))
        std_gaussian_noise = int(form_data.get("std_gaussian_noise", 25))
        min_amount = float(form_data.get("min_amount", 0.01))
        max_amount = float(form_data.get("max_amount", 0.15))
        min_alpha = float(form_data.get("min_alpha", 0.3))
        max_alpha = float(form_data.get("max_alpha", 0.7))
        max_inter_ration = float(form_data.get("max_inter_ration", 0.4))
        alpha_for_cutmix = float(form_data.get("alpha_for_cutmix", 0.5))
        calculate_geometric = 1 if "calculate_geometric" in form_data else 0
        calculate_color = 1 if "calculate_color" in form_data else 0
        calculate_noise = 1 if "calculate_noise" in form_data else 0
        calculate_mix = 1 if "calculate_mix" in form_data else 0

        if not uploaded_images:
            output_name = "Ошибка: необходимо загрузить хотя бы одно изображение"
            return render_template("augmentation_images.html", form_data=form_data, output_name=output_name)

        augmenter = autoaugmentation_images(
            uploaded_images=uploaded_images,
            uploaded_labels=uploaded_labels if is_labeled and uploaded_labels else [],
            n_samples_to_augument=n_samples,
            max_nfeatures_to_orb=max_nfeatures_to_orb,
            is_labeled=is_labeled,
            calculate_geometric=calculate_geometric,
            calculate_color=calculate_color,
            calculate_noise=calculate_noise,
            calculate_mix=calculate_mix,
            bins=bins,
            min_n_clusters=min_n_clusters,
            min_crop_ratio=min_crop_ratio,
            max_crop_ratio=max_crop_ratio,
            min_zoom_factor=min_zoom_factor,
            max_zoom_factor=max_zoom_factor,
            min_h_shift=min_h_shift,
            max_h_shift=max_h_shift,
            min_s_scale=min_s_scale,
            max_s_scale=max_s_scale,
            min_v_scale=min_v_scale,
            max_v_scale=max_v_scale,
            min_bits=min_bits,
            max_bits=max_bits,
            base_bits=base_bits,
            salt_vs_pepper=salt_vs_pepper,
            mean_gaussian_noise=mean_gaussian_noise,
            std_gaussian_noise=std_gaussian_noise,
            min_amount=min_amount,
            max_amount=max_amount,
            min_alpha=min_alpha,
            max_alpha=max_alpha,
            max_inter_ration=max_inter_ration,
            alpha_for_cutmix=alpha_for_cutmix
        )
        
        augmenter.run_pipeline()
        output_name = augmenter.output_name
        zip_file_path = getattr(augmenter, 'zip_file_path', None)
        # return redirect(url_for("result"))
        return render_template(
            "augmentation_images.html",
            form_data=form_data,
            output_name=output_name,
            zip_file_path=zip_file_path
            )
    
    return render_template("augmentation_images.html", form_data=form_data, output_name=output_name, zip_file_path=None)

@app.route("/download_zip")
def download_zip():
    zip_filename = request.args.get('filename')
    if zip_filename:
        # Ищем файл в папке downloads
        zip_path = os.path.join(os.getcwd(), "downloads", zip_filename)
        if os.path.exists(zip_path):
            try:
                return send_file(zip_path, as_attachment=True, download_name=zip_filename)
            except Exception as e:
                return f"Ошибка при загрузке файла: {str(e)}", 500
    return "Файл не найден", 404

@app.route("/videos")
def augmentation_videos():
    return render_template("augmentation_videos.html")


@app.route("/result")
def result():
    return output_name


if __name__ == "__main__":
    app.run(debug=True)

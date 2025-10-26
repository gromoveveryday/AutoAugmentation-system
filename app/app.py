from flask import Flask, render_template, request, redirect, url_for
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
    global output_name
    if request.method == "POST":
        images_dir = request.form["images_dir"]
        augmentation_dir = request.form["augmentation_dir"]
        n_samples = int(request.form["n_samples"])
        max_nfeatures_to_orb = str(request.form["max_nfeatures_to_orb"])
        is_labeled = request.form.get("is_labeled", "Нет") == "Да"
        labels_dir = request.form.get("labels_dir", "not_specified")
        dir_images_with_labels = request.form.get("dir_images_with_labels", "not_specified")
        dir_with_new_labels = request.form.get("dir_with_new_labels", "not_specified")

        augmenter = autoaugmentation_images(
            images_dir=images_dir,
            augumentation_dir=augmentation_dir,
            labels_dir=labels_dir if is_labeled else None,
            n_samples_to_augument=n_samples,
            max_nfeatures_to_orb=max_nfeatures_to_orb,
            dir_images_with_labels=dir_images_with_labels if is_labeled else None,
            dir_with_new_labels=dir_with_new_labels if is_labeled else None,
            is_labeled=is_labeled
        )
        augmenter.run_pipeline()
        output_name = augmenter.output_name
        return redirect(url_for("result"))
    return render_template("augmentation_images.html")


@app.route("/videos")
def augmentation_videos():
    return render_template("augmentation_videos.html")


@app.route("/result")
def result():
    return output_name


if __name__ == "__main__":
    app.run(debug=True)

# app.py
from flask import Flask, render_template, request, redirect, url_for
from augmentation_images import autoaugmentation_images

app = Flask(__name__)

@app.route("/")
def main():
    return render_template("main.html")

@app.route("/images", methods=["GET", "POST"])
def augmentation_images():
    global output_name
    if request.method == "POST":
        # Получаем данные из формы
        images_dir = request.form["images_dir"]
        augmentation_dir = request.form["augmentation_dir"]
        n_samples = int(request.form["n_samples"])
        max_nfeatures_to_orb = str(request.form["max_nfeatures_to_orb"])
        is_labeled = request.form.get("is_labeled", "Нет") == "Да"

        #dir_with_new_labels = str(request.form["dir_with_new_labels"])
       # labels_dir = request.form["labels_dir"]
       # dir_images_with_labels = request.form["dir_images_with_labels"]

        # Параметры для размеченных данных (используются только если is_labeled=True)
        labels_dir = request.form.get("labels_dir", "not_specified")
        dir_images_with_labels = request.form.get("dir_images_with_labels", "not_specified")
        dir_with_new_labels = request.form.get("dir_with_new_labels", "not_specified")

        # Запускаем аугментацию
        augmenter = autoaugmentation_images(
            images_dir = images_dir,
            augumentation_dir = augmentation_dir,
            labels_dir = labels_dir if is_labeled else None,
            n_samples_to_augument = n_samples,
            max_nfeatures_to_orb = max_nfeatures_to_orb,
            dir_images_with_labels = dir_images_with_labels if is_labeled else None,
            dir_with_new_labels = dir_with_new_labels if is_labeled else None, 
            is_labeled = is_labeled
        )
        augmenter.run_pipeline()
        output_name = augmenter.output_name

        return redirect(url_for("result"))
    return render_template("augmentation_images.html")

@app.route("/videos")
def augmentation_videos():
    return render_template("augmentation_videos.html")

@app.route("/timeseries")
def augmentation_timeseries():
    return render_template("augmentation_timeseries.html")

@app.route("/result")
def result():
    return output_name

if __name__ == "__main__":
    app.run(debug=True)
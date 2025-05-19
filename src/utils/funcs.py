from Mylib import myfuncs
import time
import os
import re


def train_and_save_models(
    model_training_path,
    model_name,
    model_indices,
    train_feature_data,
    train_target_data,
    val_feature_data,
    val_target_data,
    scoring,
    plot_dir,
    num_models,
):
    print(
        f"\n========Bắt đầu train {num_models} models ở chế độ bình thường !!!!!!================\n"
    )
    start_time = time.time()  # Bắt đầu tính thời gian train model
    for model_index in model_indices:
        # Load model để train
        model = myfuncs.load_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl")
        )

        print(f"Bắt đầu train model {model_name} - {model_index}")
        model.fit(train_feature_data, train_target_data)
        print(f"Kết thúc train model {model_name} - {model_index}")

        train_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            train_feature_data,
            train_target_data,
            scoring,
        )
        val_scoring = myfuncs.evaluate_model_on_one_scoring_17(
            model,
            val_feature_data,
            val_target_data,
            scoring,
        )

        # In kết quả
        print("Kết quả của model")
        print(
            f"Model index {model_name} - {model_index}\n -> Train {scoring}: {train_scoring}, Val {scoring}: {val_scoring}\n"
        )

        # Lưu model sau khi trained
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )

        # Lưu dữ liệu để vẽ biểu đồ
        model_name_in_plot = f"{model_name}_{model_index}"

        myfuncs.save_python_object(
            os.path.join(plot_dir, f"{model_name_in_plot}.pkl"),
            (model_name_in_plot, train_scoring, val_scoring),
        )

    all_model_end_time = time.time()  # Kết thúc tính thời gian train model
    true_all_models_train_time = (all_model_end_time - start_time) / 60

    print(f"Thời gian chạy tất cả: {true_all_models_train_time} (mins)")

    print(
        f"\n========Kết thúc train {num_models} models ở chế độ bình thường !!!!!!================\n"
    )


def get_batch_size_from_model_training_name(name):
    pattern = r"batch_(\d+)"
    return re.findall(pattern, name)[0]

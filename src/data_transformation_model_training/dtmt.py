from Mylib import myfuncs
import os
import time
import re


def load_data_for_data_transformation_model_training(data_transformation_path):
    # Load các training data
    train_feature_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "train_features.pkl")
    )
    train_target_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "train_target.pkl")
    )
    val_feature_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_features.pkl")
    )
    val_target_data = myfuncs.load_python_object(
        os.path.join(data_transformation_path, "val_target.pkl")
    )

    return train_feature_data, train_target_data, val_feature_data, val_target_data


def create_and_save_models_before_training(model_training_path, model_indices, models):
    for model_index, model in zip(model_indices, models):
        model = myfuncs.convert_list_estimator_into_pipeline_59(model)
        myfuncs.save_python_object(
            os.path.join(model_training_path, f"{model_index}.pkl"), model
        )

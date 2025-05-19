from Mylib import myfuncs
import os


def save_data_for_data_correction(
    data_correction_path, transformer, df_train_transformed, feature_ordinal_dict
):
    myfuncs.save_python_object(
        os.path.join(data_correction_path, "data.pkl"), df_train_transformed
    )
    myfuncs.save_python_object(
        os.path.join(data_correction_path, "feature_ordinal_dict.pkl"),
        feature_ordinal_dict,
    )
    myfuncs.save_python_object(
        os.path.join(data_correction_path, "transformer.pkl"), transformer
    )

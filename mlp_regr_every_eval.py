import os

from tqdm import tqdm
import tensorflow as tf

import utils
import trainer

if __name__ == "__main__":
    opt = utils.Config()

    # These data are for evaluation and plotting freshness curve
    x_all_data, y_all_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        train_for_regression=False,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize=None,
        shuffle=False,
    )

    model_dir = opt.save_path[:-14]
    model_folders = os.listdir(model_dir)

    for folder in tqdm(model_folders):
        cur_model = tf.keras.models.load_model(model_dir + folder)
        cur_epoch = folder[-4:]
        preds = cur_model.predict(x_all_data)
        trainer.eval_mlp_regression(preds, opt.mushroom_class, f"epoch{cur_epoch}")
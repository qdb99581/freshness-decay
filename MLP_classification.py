# Numeric Operations
import numpy as np
# Tensorflow
from keras.callbacks import ModelCheckpoint, EarlyStopping

# Custom Modules
import utils
import trainer


if __name__ == "__main__":
    opt = utils.Config()

    x_data, y_data = utils.import_data(
        data_root_path=opt.data_root_path,
        selected_bands=opt.selected_bands,
        derivative=opt.derivative,
        mushroom_class=opt.mushroom_class,
        normalize="zscore",
        shuffle=True,
    )

    # Callbacks
    checkpoint = ModelCheckpoint(
        opt.save_path,
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
        save_weights_only=False,
        mode='auto',
        initial_value_threshold=0.0
    )

    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=500,
        restore_best_weights=True,
        verbose=0
    )

    callbacks = [checkpoint, early_stop]

    reports_list = []
    reports_list.append(f"Class {opt.mushroom_class}")
    for n_hidden in range(5, 6):

        # neurons_permutations = trainer.neuron_permutor(
        #     n_hidden=n_hidden,
        #     max_neuron=512,
        #     min_neuron=32
        # )
        
        neurons_permutations = [
            [512, 512, 128, 64, 32],
            [512, 256, 128, 64, 32]
            ]

        print("=" * 120)
        print(f"Training with {n_hidden} hidden layers.")
        for neurons_layout in neurons_permutations:
            # Training
            print()
            print("-" * 100)
            print(f"Training with {neurons_layout}...")

            report = trainer.KFold_training(
                x_data=x_data,
                y_data=y_data,
                neurons_layout=neurons_layout,
                callbacks=callbacks,
                n_splits=opt.n_KFold,
                activation=opt.activation,
                selected_bands=opt.selected_bands,
                learning_rate=opt.learning_rate,
                batch_size=opt.batch_size,
                valid_ratio=opt.valid_ratio,
                n_epochs=opt.n_epochs,
                save_path=opt.save_path
            )

            print(report)
            reports_list.append(report)
            reports_array = np.array(reports_list).T

            # Export reports
            with open(f"Reports_{opt.mushroom_class}_no_derivative.txt", "w") as text_file:
                print(reports_array, file=text_file)

        print()

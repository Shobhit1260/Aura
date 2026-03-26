from __future__ import annotations

import argparse
import json
from pathlib import Path

import tensorflow as tf


DEFAULT_NEGATIVE_CLASS = "not_chest_xray"
DEFAULT_POSITIVE_CLASS = "chest_xray"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train chest-X-ray gate model (chest_xray vs not_chest_xray) and export TFLite."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root with class subfolders")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for model artifacts")
    parser.add_argument("--model-name", type=str, default="tb_xray_gate_model", help="Artifact prefix")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs-head", type=int, default=4, help="Feature extraction epochs")
    parser.add_argument("--epochs-finetune", type=int, default=4, help="Fine-tune epochs")
    parser.add_argument("--validation-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--quantization",
        type=str,
        choices=["none", "dynamic", "float16"],
        default="dynamic",
        help="TFLite post-training quantization mode",
    )
    parser.add_argument(
        "--negative-class",
        type=str,
        default=DEFAULT_NEGATIVE_CLASS,
        help="Folder name for non-chest-X-ray images",
    )
    parser.add_argument(
        "--positive-class",
        type=str,
        default=DEFAULT_POSITIVE_CLASS,
        help="Folder name for chest-X-ray images",
    )
    return parser.parse_args()


def build_datasets(
    args: argparse.Namespace,
    class_names: list[str],
) -> tuple[tf.data.Dataset, tf.data.Dataset]:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    missing = [name for name in class_names if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folder(s): {missing}. Expected dataset structure with {class_names}."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        validation_split=args.validation_split,
        subset="training",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        color_mode="rgb",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=class_names,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        color_mode="rgb",
    )

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds


def build_model(img_size: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
        ],
        name="augmentation",
    )

    x = augmentation(inputs)
    x = tf.keras.layers.Rescaling(1.0 / 255.0, name="rescale_0_1")(x)

    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="xray_score")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="tb_xray_gate_efficientnetb0")
    return model, base_model


def set_finetune_layers(base_model: tf.keras.Model, unfreeze_layers: int = 45) -> None:
    base_model.trainable = True
    split_idx = max(len(base_model.layers) - unfreeze_layers, 0)
    for i, layer in enumerate(base_model.layers):
        if i < split_idx or isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )


def export_tflite(model: tf.keras.Model, output_path: Path, quantization: str) -> None:
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quantization == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]

    output_path.write_bytes(converter.convert())


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    class_names = [args.negative_class, args.positive_class]

    tf.keras.utils.set_random_seed(args.seed)
    train_ds, val_ds = build_datasets(args, class_names)

    model, base_model = build_model(args.img_size)
    compile_model(model, learning_rate=1e-3)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    print("\n[Stage 1/2] Training gate head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[Stage 2/2] Fine-tuning gate backbone...")
    set_finetune_layers(base_model)
    compile_model(model, learning_rate=1e-5)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head + args.epochs_finetune,
        initial_epoch=args.epochs_head,
        callbacks=callbacks,
        verbose=1,
    )

    keras_path = output_dir / f"{args.model_name}.keras"
    tflite_path = output_dir / f"{args.model_name}.tflite"
    metadata_path = output_dir / f"{args.model_name}_metadata.json"

    model.save(keras_path)
    export_tflite(model, tflite_path, args.quantization)

    metadata = {
        "model_name": args.model_name,
        "input_shape": [args.img_size, args.img_size, 3],
        "normalization": "in_graph_rescaling_1_over_255",
        "class_names": class_names,
        "positive_class": args.positive_class,
        "threshold": 0.5,
        "quantization": args.quantization,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nGate model training complete.")
    print(f"Keras model: {keras_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

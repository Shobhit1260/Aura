from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 binary classifier and export Keras + TFLite artifacts."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root with class subfolders")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for model artifacts")
    parser.add_argument("--model-name", type=str, default="model", help="Artifact prefix")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs-head", type=int, default=5, help="Feature extraction epochs")
    parser.add_argument("--epochs-finetune", type=int, default=5, help="Fine-tune epochs")
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
        "--use-class-weights",
        action="store_true",
        help="Use class weights to reduce imbalance bias",
    )
    parser.add_argument(
        "--finetune-unfreeze-layers",
        type=int,
        default=80,
        help="How many trailing base layers to unfreeze during fine-tuning",
    )
    return parser.parse_args()


def build_datasets(args: argparse.Namespace) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int, int]:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    expected_classes = ["normal", "tb"]
    missing = [name for name in expected_classes if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing class folder(s): {missing}. Expected dataset structure with 'normal/' and 'tb/'."
        )

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="binary",
        class_names=expected_classes,
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
        class_names=expected_classes,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        color_mode="rgb",
    )

    class_names = train_ds.class_names
    train_count = int(train_ds.cardinality().numpy() * args.batch_size)
    val_count = int(val_ds.cardinality().numpy() * args.batch_size)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names, train_count, val_count


def compute_class_weights(train_ds: tf.data.Dataset) -> dict[int, float]:
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(int(y.numpy()[0]))

    labels_np = np.array(labels)
    counts = np.bincount(labels_np, minlength=2)
    total = counts.sum()
    weights = {
        0: float(total / (2.0 * max(counts[0], 1))),
        1: float(total / (2.0 * max(counts[1], 1))),
    }
    return weights


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
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(img_size, img_size, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="risk_score")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="efficientnetb0_binary")
    return model, base_model


def set_finetune_layers(base_model: tf.keras.Model, unfreeze_layers: int) -> None:
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

    tflite_model = converter.convert()
    output_path.write_bytes(tflite_model)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, class_names, train_count, val_count = build_datasets(args)

    model, base_model = build_model(args.img_size)
    compile_model(model, learning_rate=1e-3)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    class_weights = compute_class_weights(train_ds) if args.use_class_weights else None

    print("\n[Stage 1/2] Training classification head...")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[Stage 2/2] Fine-tuning top EfficientNetB0 layers...")
    set_finetune_layers(base_model, args.finetune_unfreeze_layers)
    compile_model(model, learning_rate=1e-5)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head + args.epochs_finetune,
        initial_epoch=args.epochs_head,
        class_weight=class_weights,
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
        "threshold": 0.5,
        "quantization": args.quantization,
        "train_samples_estimate": train_count,
        "val_samples_estimate": val_count,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Keras model: {keras_path}")
    print(f"TFLite model: {tflite_path}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train EfficientNetB0 eye-disease multiclass classifier and export Keras + TFLite artifacts."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Dataset root with one folder per class")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for model artifacts")
    parser.add_argument("--model-name", type=str, default="eye_disease_model", help="Artifact prefix")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--epochs-head", type=int, default=6, help="Feature extraction epochs")
    parser.add_argument("--epochs-finetune", type=int, default=6, help="Fine-tune epochs")
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


def build_datasets(
    args: argparse.Namespace,
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], int, int]:
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    class_dirs = [p.name for p in data_dir.iterdir() if p.is_dir()]
    class_names = sorted(class_dirs)
    if len(class_names) < 2:
        raise ValueError("Expected at least 2 class folders under data-dir")

    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        label_mode="int",
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
        label_mode="int",
        class_names=class_names,
        validation_split=args.validation_split,
        subset="validation",
        seed=args.seed,
        image_size=(args.img_size, args.img_size),
        batch_size=args.batch_size,
        color_mode="rgb",
    )

    train_count = int(train_ds.cardinality().numpy() * args.batch_size)
    val_count = int(val_ds.cardinality().numpy() * args.batch_size)

    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(autotune)
    val_ds = val_ds.cache().prefetch(autotune)

    return train_ds, val_ds, class_names, train_count, val_count


def compute_class_weights(train_ds: tf.data.Dataset, n_classes: int) -> dict[int, float]:
    labels = []
    for _, y in train_ds.unbatch():
        labels.append(int(y.numpy()))

    counts = np.bincount(np.array(labels, dtype=np.int32), minlength=n_classes)
    total = counts.sum()
    weights = {
        idx: float(total / (max(counts[idx], 1) * n_classes))
        for idx in range(n_classes)
    }
    return weights


def build_model(img_size: int, n_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    inputs = tf.keras.Input(shape=(img_size, img_size, 3), name="image")

    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.06),
            tf.keras.layers.RandomZoom(0.12),
            tf.keras.layers.RandomContrast(0.12),
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
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(n_classes, activation="softmax", name="class_probs")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="eye_disease_efficientnetb0")
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
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name="top2_accuracy"),
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

    tf.keras.utils.set_random_seed(args.seed)

    train_ds, val_ds, class_names, train_count, val_count = build_datasets(args)
    n_classes = len(class_names)

    model, base_model = build_model(args.img_size, n_classes)
    compile_model(model, learning_rate=1e-3)

    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", mode="max", patience=4, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-6),
    ]

    class_weights = compute_class_weights(train_ds, n_classes) if args.use_class_weights else None

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

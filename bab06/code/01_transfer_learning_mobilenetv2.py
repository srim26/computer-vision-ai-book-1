"""Bab 6 Compact — Transfer Learning MobileNetV2."""
from pathlib import Path
import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT=Path(__file__).resolve().parent.parent
DATA=ROOT/"data"/"pet_subset"
OUT=ROOT/"output"
OUT.mkdir(parents=True,exist_ok=True)
IMG=(160,160); BATCH=32; SEED=42

def load(split, shuffle):
    return keras.utils.image_dataset_from_directory(
        DATA/split, image_size=IMG, batch_size=BATCH,
        shuffle=shuffle, seed=SEED if shuffle else None
    )

def main():
    tf.keras.utils.set_random_seed(SEED)
    train_ds=load("train",True)
    val_ds=load("val",False)
    test_ds=load("test",False)
    class_names=train_ds.class_names
    autotune=tf.data.AUTOTUNE
    train_ds=train_ds.prefetch(autotune); val_ds=val_ds.prefetch(autotune); test_ds=test_ds.prefetch(autotune)

    base=keras.applications.MobileNetV2(
        input_shape=(160,160,3), include_top=False, weights="imagenet"
    )
    base.trainable=False
    inputs=keras.Input((160,160,3))
    x=keras.applications.mobilenet_v2.preprocess_input(inputs)
    x=base(x,training=False)
    x=layers.GlobalAveragePooling2D()(x)
    x=layers.Dropout(0.2)(x)
    outputs=layers.Dense(len(class_names),activation="softmax")(x)
    model=keras.Model(inputs,outputs)

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    print("\n=== FEATURE EXTRACTION: 10 epoch ===")
    h1=model.fit(train_ds,validation_data=val_ds,epochs=10)

    base.trainable=True
    for layer in base.layers[:-20]:
        layer.trainable=False
    model.compile(optimizer=keras.optimizers.Adam(1e-5),
                  loss="sparse_categorical_crossentropy",metrics=["accuracy"])
    print("\n=== FINE-TUNING: 5 epoch, 20 layer akhir dibuka ===")
    h2=model.fit(train_ds,validation_data=val_ds,epochs=5)

    loss,acc=model.evaluate(test_ds,verbose=0)
    model.save(OUT/"mobilenetv2_pet.keras")
    metrics={
        "classes":class_names,
        "split":{"train":280,"validation":60,"test":60},
        "image_size":[160,160],
        "feature_extraction_epochs":10,
        "fine_tuning_epochs":5,
        "test_loss":float(loss),"test_accuracy":float(acc),
        "history_feature_extraction":{k:[float(x) for x in v] for k,v in h1.history.items()},
        "history_fine_tuning":{k:[float(x) for x in v] for k,v in h2.history.items()}
    }
    (OUT/"metrics_bab06.json").write_text(json.dumps(metrics,indent=2),encoding="utf-8")
    print(f"\nTest accuracy: {acc:.4f}")
    print("Metrics:",OUT/"metrics_bab06.json")
    print("Model  :",OUT/"mobilenetv2_pet.keras")

if __name__=="__main__":
    main()

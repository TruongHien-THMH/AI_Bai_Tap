import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# ===== CẤU HÌNH =====
IMG_SIZE = (128, 128)   # Hình nhỏ để train nhanh
BATCH_SIZE = 16
EPOCHS = 10
DATA_DIR = "dataset"

# ===== DATASET =====
# Chú ý: bạn có dataset/train/, dataset/val/, dataset/test/
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "test"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

print("✅ Classes:", train_gen.class_indices)

# ===== MODEL =====
base = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
preds = Dense(train_gen.num_classes, activation="softmax")(x)
model = Model(inputs=base.input, outputs=preds)

# Freeze base model
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])

# ===== CALLBACKS =====
early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

# ===== TRAIN =====
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ===== LƯU MODEL =====
model.save("food_model.h5")
classes = {v: k for k, v in train_gen.class_indices.items()}
with open("classes.json", "w") as f:
    json.dump(classes, f)

print("✅ Model & classes.json đã lưu thành công!")

# ===== VẼ ACCURACY =====
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig("accuracy_chart.png")
plt.show()
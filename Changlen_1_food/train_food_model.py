import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# ===== CẤU HÌNH =====
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
DATA_DIR = "dataset"

# ===== DATA AUGMENTATION =====
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# ===== MODEL =====
base = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
preds = Dense(train_gen.num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=preds)

# ===== TRAINING =====
for layer in base.layers:
    layer.trainable = False

model.compile(optimizer=Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS
)

# ===== FINE-TUNE =====
for layer in base.layers[-20:]:
    layer.trainable = True

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
history_fine = model.fit(train_gen, validation_data=val_gen, epochs=5)

# ===== LƯU MODEL =====
model.save("food_model.h5")
classes = {v: k for k, v in train_gen.class_indices.items()}
with open("classes.json", "w") as f:
    json.dump(classes, f)

print("✅ Model & classes.json đã lưu thành công!")

# ===== BIỂU ĐỒ ACCURACY =====
plt.figure(figsize=(8,5))
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy over epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
plt.show()
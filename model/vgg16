pip install opencv-python matplotlib scikit-learn pandas seaborn tqdm

# Install git (if not already available)
!apt-get install git -y   # (only for Colab/Linux; skip if on Windows Anaconda with Git installed)

# Clone only the 'data' folder using sparse checkout
!git clone --filter=blob:none --sparse https://github.com/Sam-B-Y/CV-Corn-Disease-Detection.git

%cd CV-Corn-Disease-Detection
!git sparse-checkout set data/Common_Rust data/Blight data/Gray_Leaf_Spot data/Healthy

# Now you have all 4 folders locally inside CV-Corn-Disease-Detection/data/
!ls data


import os

# Path to the data
data_path = "data"

# List disease folders
print("Class folders:", os.listdir(data_path))



!ls data



import os
print("Classes:", os.listdir("data"))
for cls in os.listdir("data"):
    n_files = len(os.listdir(f"data/{cls}"))
    print(f"{cls}: {n_files} images")



import os

DATA_DIR = "data"

for cls in os.listdir(DATA_DIR):
    cls_path = os.path.join(DATA_DIR, cls)
    print(f"{cls}: {len(os.listdir(cls_path))} images")




import os, shutil
from sklearn.model_selection import train_test_split

SRC_DIR = "data"
DST_DIR = "maize_dataset"
os.makedirs(DST_DIR, exist_ok=True)

train_ratio, val_ratio, test_ratio = 0.7, 0.2, 0.1
seed = 42

for cls in os.listdir(SRC_DIR):
    cls_path = os.path.join(SRC_DIR, cls)
    files = os.listdir(cls_path)

    train_val, test = train_test_split(files, test_size=test_ratio, random_state=seed)
    train, val = train_test_split(train_val, test_size=val_ratio/(train_ratio+val_ratio), random_state=seed)

    for split, split_files in [("train", train), ("val", val), ("test", test)]:
        split_dir = os.path.join(DST_DIR, split, cls)
        os.makedirs(split_dir, exist_ok=True)
        for f in split_files:
            shutil.copy(os.path.join(cls_path, f), os.path.join(split_dir, f))

print("✅ Dataset split into train/val/test inside 'maize_dataset/'")





from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH_SIZE = 64

train_datagen = ImageDataGenerator(    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
train_flow = train_datagen.flow_from_directory("maize_dataset/train", target_size=(224,224), batch_size=64, class_mode="categorical")

val_datagen = ImageDataGenerator(rescale=1./255)
val_flow = val_datagen.flow_from_directory("maize_dataset/val", target_size=(224,224), batch_size=64, class_mode="categorical")
test_datagen = ImageDataGenerator(rescale=1./255)

train_flow = train_datagen.flow_from_directory(
    "maize_dataset/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_flow = val_datagen.flow_from_directory(
    "maize_dataset/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_flow = test_datagen.flow_from_directory(
    "maize_dataset/test",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)




import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.5)(x)
x = Dense(256, activation="relu")(x)
x = Dropout(0.4)(x)
output = Dense(len(train_flow.class_indices), activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=output)

model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# 3. Train with class weights + callbacks
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

y_train = train_flow.classes
class_weights = dict(enumerate(compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)))

history = model.fit(
    train_flow,
    validation_data=val_flow,
    epochs=10,
    class_weight=class_weights,
)




import os
import matplotlib.pyplot as plt

# Path to dataset (update this if needed)
data_dir = "/content/CV-Corn-Disease-Detection/data" # Using train split for visualization

# Class folders
classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Count images in each class
counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

# Bar Chart
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)  # Left plot
plt.bar(counts.keys(), counts.values(), color="skyblue")
plt.title("A) Bar Chart")
plt.ylabel("Number of Images")
plt.xticks(rotation=20)

# Pie Chart
plt.subplot(1, 2, 2)  # Right plot
plt.pie(counts.values(),
        labels=counts.keys(),
        autopct="%.2f%%",
        startangle=90,
        colors=["cornflowerblue", "tomato", "lightgrey", "gold"])
plt.title("B) Pie Chart")

plt.tight_layout()
plt.show()




import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# --- 1. Get class names from training flow ---
class_names = list(train_flow.class_indices.keys())

# --- 2. Define test folder path ---
test_folder = "/content/CV-Corn-Disease-Detection/maize_dataset/test"

# --- 3. Initialize true/pred labels ---
y_true = []
y_pred = []

IMG_SIZE = (224, 224)

# --- 4. Loop through test images ---
for class_name in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.endswith(".jpg"):
                img_path = os.path.join(class_path, fname)

                # Load + preprocess image
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict
                predictions = model.predict(img_array, verbose=0)
                predicted_class = class_names[np.argmax(predictions)]

                # Append results
                y_true.append(class_name)
                y_pred.append(predicted_class)

# --- 5. Classification report ---
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# --- 6. Confusion matrix ---
cm = confusion_matrix(y_true, y_pred, labels=class_names)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.show()

# --- 7. Show sample predictions ---
plt.figure(figsize=(15, 8))
for i in range(8):
    img_path = os.path.join(test_folder, y_true[i], os.listdir(os.path.join(test_folder, y_true[i]))[0])
    img = image.load_img(img_path, target_size=IMG_SIZE)
    plt.subplot(2, 4, i+1)
    plt.imshow(img)
    plt.title(f"True: {y_true[i]}\nPred: {y_pred[i]}")
    plt.axis("off")
plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Get a batch of images and labels from the training flow
images, labels = next(train_flow)

fig, axes = plt.subplots(3, 4, figsize=(15, 7))

for i, ax in enumerate(axes.flat):
    if i < len(images):
        img = images[i]
        # Since the images are already in numpy array format from the flow,
        # we don't need PIL.Image.open and can directly display.
        ax.imshow(img)
        # Optionally, display the predicted label (if you have predictions) or true label
        # For now, just show the image
        ax.set_title(f"Label: {np.argmax(labels[i])}") # Display the index of the true label
        ax.axis("off")  # hide axes

plt.tight_layout()
plt.show()


fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Training loss
axes[0, 0].plot(history.history['loss'], label='Training Loss')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)
axes[0, 0].set_xlim([0, 5])  # set to number of epochs

# Validation loss
axes[0, 1].plot(history.history['val_loss'], label='Validation Loss')
axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)
axes[0, 1].set_xlim([0, 5])

# Training accuracy
axes[1, 0].plot(history.history['accuracy'], label='Training Accuracy')
axes[1, 0].set_title('Training Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)
axes[1, 0].set_xlim([0, 5])

# Validation accuracy
axes[1, 1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1, 1].set_title('Validation Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)
axes[1, 1].set_xlim([0, 5])

plt.tight_layout()
plt.show()




import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# --- Loss (Training + Validation) ---
axes[0].plot(history.history['loss'], label='Training Loss')
axes[0].plot(Testing, label='Validation Loss')
axes[0].set_title('Training vs Validation Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(True)

# --- Accuracy (Training + Validation) ---
axes[1].plot(history.history['accuracy'], label='Training Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
axes[1].set_title('Training vs Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.show()





import matplotlib.pyplot as plt
import numpy as np # Import numpy

# 1. Training + validation accuracy curves
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')

# 2. Add test accuracy (constant line)
# Calculate test accuracy using the model and test_flow
test_loss, test_acc = model.evaluate(test_flow, verbose=0) # Calculate test loss and accuracy

plt.axhline(y=test_acc, color='r', linestyle='--', label=f'Test Accuracy: {test_acc:.2f}')

# Formatting
plt.title('Training, Validation, and Test Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# --- 1. Prepare y_true and y_pred_proba ---
y_true = []
y_pred_proba = []

for class_name in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.endswith(".jpg"):
                img_path = os.path.join(class_path, fname)

                # Load + preprocess image
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0) / 255.0

                # Predict probability distribution
                probas = model.predict(img_array, verbose=0)

                y_true.append(class_name)
                y_pred_proba.append(probas[0])

y_pred_proba = np.array(y_pred_proba)

# --- 2. Binarize labels (for multi-class ROC) ---
class_names = list(train_flow.class_indices.keys())
y_true_bin = label_binarize(y_true, classes=class_names)

# --- 3. Compute ROC + AUC for each class ---
plt.figure(figsize=(10, 8))
for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

# --- 4. Plot random baseline ---
plt.plot([0, 1], [0, 1], 'k--', label='Random Chance')

# --- 5. Formatting ---
plt.title('ROC Curve (Multi-Class)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()



import matplotlib.pyplot as plt

# Get values from history
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12,5))

# Accuracy plot
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

# Loss plot
plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()



import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Get predictions on validation/test data
# If using ImageDataGenerator flow
val_flow.reset()  # make sure generator starts from the beginning
y_pred_probs = model.predict(val_flow)
y_prediction = np.argmax(y_pred_probs, axis=1)
y_test = val_flow.classes  # true labels

# 2️⃣ Classification report
print(classification_report(y_test, y_prediction, target_names=val_flow.class_indices.keys()))

# 3️⃣ Accuracy
acc = accuracy_score(y_test, y_prediction)
print("Accuracy:", acc)



# 4️⃣ Confusion matrix
cm = confusion_matrix(y_test, y_prediction)

# Plot the confusion matrix as a heatmap
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=val_flow.class_indices.keys(),
            yticklabels=val_flow.class_indices.keys())
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()




import os
import matplotlib.pyplot as plt

# Path to dataset (update this if needed)
data_dir = "maize_dataset/train" # Using train split for visualization

# Class folders
classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

# Count images in each class
counts = {cls: len(os.listdir(os.path.join(data_dir, cls))) for cls in classes}

# Bar Chart
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)  # Left plot
plt.bar(counts.keys(), counts.values(), color="skyblue")
plt.title("A) Bar Chart")
plt.ylabel("Number of Images")
plt.xticks(rotation=20)

# Pie Chart
plt.subplot(1, 2, 2)  # Right plot
plt.pie(counts.values(),
        labels=counts.keys(),
        autopct="%.2f%%",
        startangle=90,
        colors=["cornflowerblue", "tomato", "lightgrey", "gold"])
plt.title("B) Pie Chart")

plt.tight_layout()
plt.show()




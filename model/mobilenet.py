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



import os
print("Classes:", os.listdir("data"))
for cls in os.listdir("data"):
    n_files = len(os.listdir(f"data/{cls}"))
    print(f"{cls}: {n_files} images")


import os
from sklearn.model_selection import train_test_split
import shutil

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
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


IMG_SIZE = (224,224)
BATCH_SIZE = 32
DST_DIR = "maize_dataset" # Assuming DST_DIR is defined in a previous cell

train_dir = os.path.join(DST_DIR, "train")
val_dir = os.path.join(DST_DIR, "val")
test_dir = os.path.join(DST_DIR, "test")


train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_flow = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_flow = val_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

test_flow= test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)



train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)



import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Number of classes
num_classes = len(train_flow.class_indices)

# Load MobileNetV2 as base
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(IMG_SIZE[0],IMG_SIZE[1],3))

# Add custom top layers
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)

# Create final model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model initially
for layer in base_model.layers:
    layer.trainable = False


# Compile the model with the initial frozen base
model.compile(optimizer=Adam(1e-4), loss="categorical_crossentropy", metrics=["accuracy"])

# Train the initially frozen model
history_frozen = model.fit(train_flow, validation_data=val_flow, epochs=20)

# Unfreeze some layers for fine-tuning
# For fine-tuning, it's common to unfreeze the later layers of the base model
# We'll unfreeze the last 30 layers for example
for layer in base_model.layers[-30:]:
    layer.trainable = True


# Recompile the model with a lower learning rate for fine-tuning
model.compile(optimizer=Adam(1e-5), loss="categorical_crossentropy", metrics=["accuracy"])

# Continue training with unfrozen layers (fine-tuning)
history_fine_tune = model.fit(train_flow, validation_data=val_flow, epochs=20)

# Concatenate the history objects for plotting later
history = tf.keras.callbacks.History()
history.history['loss'] = history_frozen.history['loss'] + history_fine_tune.history['loss']
history.history['val_loss'] = history_frozen.history['val_loss'] + history_fine_tune.history['val_loss']
history.history['accuracy'] = history_frozen.history['accuracy'] + history_fine_tune.history['accuracy']
history.history['val_accuracy'] = history_frozen.history['val_accuracy'] + history_fine_tune.history['val_accuracy']




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





import os
import numpy as np
from tensorflow.keras.preprocessing import image

test_folder = "/content/CV-Corn-Disease-Detection/maize_dataset/test"
correct = 0
total = 0

# Get class names from the test_flow
class_names = list(test_flow.class_indices.keys())

for class_name in os.listdir(test_folder):
    class_path = os.path.join(test_folder, class_name)
    if os.path.isdir(class_path):
        for fname in os.listdir(class_path):
            if fname.endswith(".jpg"):
                img_path = os.path.join(class_path, fname)

                # Load and preprocess the image using TensorFlow
                img = image.load_img(img_path, target_size=IMG_SIZE)
                img_array = image.img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array /= 255.0  # Rescale the image

                # Make prediction
                predictions = model.predict(img_array)
                predicted_class_index = np.argmax(predictions)
                predicted_class_name = class_names[predicted_class_index]

                # Get the true label from the directory name
                true_label = class_name

                if predicted_class_name == true_label:
                    correct += 1
                total += 1

print(f"Test Accuracy: {correct/total*100:.2f}%")





import numpy as np
import matplotlib.pyplot as plt

# Get a batch of validation images and labels
images, labels = next(iter(val_flow))   # images shape: (batch, h, w, 3)

# Predict on these images
preds = model.predict(images)
pred_classes = np.argmax(preds, axis=1)   # predicted class indices
true_classes = np.argmax(labels, axis=1)  # true class indices

# Map indices to class names
class_names = list(val_flow.class_indices.keys())

# Plot some results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
axes = axes.flatten()

for i, ax in enumerate(axes):
    ax.imshow(images[i])
    ax.axis("off")

    true_label = class_names[true_classes[i]]
    pred_label = class_names[pred_classes[i]]

    # Color titles: green if correct, red if wrong
    color = "green" if true_label == pred_label else "red"

    ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                 fontsize=10, pad=10, color=color)

plt.tight_layout()
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Get all test data predictions
test_flow.reset()  # ensure generator is at the beginning
y_true = []
y_pred = []

for i in range(len(test_flow)):
    images, labels = next(test_flow)
    predictions = model.predict(images)
    y_true.extend(np.argmax(labels, axis=1))
    y_pred.extend(np.argmax(predictions, axis=1))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(test_flow.class_indices.keys()))
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix on Test Data")
plt.show()





import matplotlib.pyplot as plt

# Train & Validation from history
train_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs_range = range(1, len(train_acc) + 1)

# Test accuracy (evaluated after training)
test_loss, test_acc = model.evaluate(test_flow, verbose=0)

plt.figure(figsize=(8,6))

# Plot train & validation accuracy
plt.plot(epochs_range, train_acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")

# Plot test accuracy (flat line for comparison)
plt.axhline(y=test_acc, color='red', linestyle='--', label=f"Test Accuracy ({test_acc:.2f})")

plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Train, Validation, and Test Accuracy")
plt.legend()
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






import os
import matplotlib.pyplot as plt

# Path to dataset (update this if needed)
data_dir = "maize_dataset/train" # Changed to point to the training data in maize_dataset


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





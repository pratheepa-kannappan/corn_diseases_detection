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





import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import cv2
import numpy as np




# 2. Preprocessing / segmentation function (optional)
def remove_background(img_bgr):
    """
    Example: convert to HSV, mask non-leaf pixels, return masked image.
    You can adapt from the CV-Corn repo’s segmentation code.
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    # Define green range (you may need to tune)
    lower = np.array([25, 40, 40])
    upper = np.array([85, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    # Morphological operations
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    # Apply mask
    res = cv2.bitwise_and(img_bgr, img_bgr, mask=mask)
    return res



data_dir = '/content/data'





# Define transforms
train_transform = transforms.Compose([
    transforms.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),  # get BGR for OpenCV
    transforms.Lambda(remove_background),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Lambda(lambda img: cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)),
    transforms.Lambda(remove_background),
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

train_dataset = datasets.ImageFolder(os.path.join("maize_dataset", 'train'), transform=train_transform)
val_dataset = datasets.ImageFolder(os.path.join("maize_dataset", 'val'), transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

num_classes = len(train_dataset.classes)
print("Classes:", train_dataset.classes)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



model = models.resnet18(pretrained=True)
# Replace final layer
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)




 import os
import matplotlib.pyplot as plt

# Path to dataset (update this if needed)
data_dir = "/content/CV-Corn-Disease-Detection/maize_dataset/train" # Using train split for visualization

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




def train_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

def validate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss/total, correct/total

num_epochs = 10
best_val_acc = 0.0
train_loss_history = []
train_acc_history = []
val_loss_history = []
val_acc_history = []


for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion)
    scheduler.step()

    train_loss_history.append(train_loss)
    train_acc_history.append(train_acc)
    val_loss_history.append(val_loss)
    val_acc_history.append(val_acc)

    print(f"Epoch {epoch}/{num_epochs-1}, Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    # Save best
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")




import os
import numpy as np
import torch
from PIL import Image # Use PIL for image loading to be consistent with the rest of the notebook
from torchvision import transforms # Use torchvision transforms to be consistent

test_folder = "/content/CV-Corn-Disease-Detection/maize_dataset/test"
correct = 0
total = 0

# Get class names from the train_dataset (consistent with data loading cell)
class_names = train_dataset.classes

model.eval() # Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Ensure device is defined if not already

with torch.no_grad(): # Disable gradient calculation for inference
    for class_name in os.listdir(test_folder):
        class_path = os.path.join(test_folder, class_name)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if fname.endswith(".jpg"):
                    img_path = os.path.join(class_path, fname)

                    # Load and preprocess the image using PyTorch transforms
                    img = Image.open(img_path).convert('RGB')
                    # Using val_transform as it includes resizing,ToTensor and Normalize
                    img_tensor = val_transform(img).unsqueeze(0).to(device)

                    # Make prediction
                    outputs = model(img_tensor) # Forward pass through the model
                    _, predicted_class_index = torch.max(outputs, 1) # Get predicted class index
                    predicted_class_name = class_names[predicted_class_index.item()] # Get class name

                    # Get the true label from the directory name
                    true_label = class_name

                    if predicted_class_name == true_label:
                        correct += 1
                    total += 1

print(f"Test Accuracy: {correct/total*100:.2f}%")





import numpy as np
import matplotlib.pyplot as plt
import torch

# Get a batch of validation images and labels
# images shape: (batch, 3, h, w) for PyTorch tensors
images, labels = next(iter(val_loader))

# Predict on these images
model.eval() # Set model to evaluation mode
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
images = images.to(device)
with torch.no_grad(): # Disable gradient calculation for inference
    outputs = model(images) # Forward pass through the model
    preds = torch.softmax(outputs, dim=1).cpu().numpy() # Get probabilities and move to CPU as numpy array

pred_classes = np.argmax(preds, axis=1)   # predicted class indices
true_classes = labels.cpu().numpy()  # true class indices, move to CPU as numpy array

# Map indices to class names
class_names = list(val_loader.dataset.classes) # Get class names from the dataset

# Plot some results
fig, axes = plt.subplots(2, 4, figsize=(16, 8))  # 2 rows, 4 columns
axes = axes.flatten()

for i, ax in enumerate(axes):
    # Transpose image tensor from (C, H, W) to (H, W, C) for plotting
    img = images[i].cpu().numpy().transpose((1, 2, 0))
    # Unnormalize image for display if normalization was applied
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    ax.imshow(img)
    ax.axis("off")

    true_label = class_names[true_classes[i]]
    pred_label = class_names[pred_classes[i]]

    # Color titles: green if correct, red if wrong
    color = "green" if true_label == pred_label else "red"

    ax.set_title(f"True: {true_label}\nPred: {pred_label}",
                 fontsize=10, pad=10, color=color)

plt.tight_layout()
plt.show()






from PIL import Image

def predict_image(model, image_path, transform, class_names):
    model.eval()
    img = Image.open(image_path).convert('RGB')
    x = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(x)
        _, pred = torch.max(out, 1)
    return class_names[pred.item()]

# Example:
class_names = train_dataset.classes
print("Prediction for sample:", predict_image(model, "/content/CV-Corn-Disease-Detection/maize_dataset/test/Common_Rust/Corn_Common_Rust (1001).JPG", val_transform, class_names))







import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Training loss
axes[0, 0].plot(range(len(train_loss_history)), train_loss_history, label='Training Loss', color='blue')
axes[0, 0].set_title('Training Loss')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Validation loss
axes[0, 1].plot(range(len(val_loss_history)), val_loss_history, label='Validation Loss', color='orange')
axes[0, 1].set_title('Validation Loss')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Training accuracy
axes[1, 0].plot(range(len(train_acc_history)), train_acc_history, label='Training Accuracy', color='green')
axes[1, 0].set_title('Training Accuracy')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Accuracy')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Validation accuracy
axes[1, 1].plot(range(len(val_acc_history)), val_acc_history, label='Validation Accuracy', color='red')
axes[1, 1].set_title('Validation Accuracy')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.show()




                      
import matplotlib.pyplot as plt

# Example history (replace with yours)
epochs = range(1, len(train_acc_history) + 1)

plt.figure(figsize=(8,6))

# Training accuracy
plt.plot(epochs, train_acc_history, label="Train Accuracy", color="green")

# Validation accuracy
plt.plot(epochs, val_acc_history, label="Validation Accuracy", color="red")

# Test accuracy (fixed line)
test_acc = 91.72
plt.axhline(y=test_acc/100, color="blue", linestyle="--", label=f"Test Accuracy ({test_acc}%)")

# Labels and title
plt.title("Training vs Validation vs Test Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()




                      
import matplotlib.pyplot as plt

# Use the history lists created during training
train_acc = train_acc_history
val_acc = val_acc_history
train_loss = train_loss_history
val_loss = val_loss_history
epochs_range = range(1, len(train_acc) + 1)

plt.figure(figsize=(12,5))

# Accuracy
plt.subplot(1,2,1)
plt.plot(epochs_range, train_acc, label="Train Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()

# Loss
plt.subplot(1,2,2)
plt.plot(epochs_range, train_loss, label="Train Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

plt.tight_layout()
plt.show()







import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import torch

# Define the get_all_preds function
def get_all_preds(model, loader, device):
    all_preds = torch.tensor([])
    all_labels = torch.tensor([])
    model.eval()
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            preds = model(imgs)
            all_preds = torch.cat((all_preds, preds.cpu()), dim=0)
            all_labels = torch.cat((all_labels, labels.cpu()), dim=0)
    return all_labels.numpy(), np.argmax(all_preds.numpy(), axis=1)

# Load the best model state dict
model.load_state_dict(torch.load("best_model.pth"))

# Get predictions
true_labels, pred_labels = get_all_preds(model, val_loader, device)

# Confusion Matrix
cm = confusion_matrix(true_labels, pred_labels)

# Class names (if using ImageFolder)
if hasattr(train_loader.dataset, "classes"):
    class_names = train_loader.dataset.classes
else:
    class_names = [str(i) for i in range(len(set(true_labels)))]

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap="Blues", values_format="d", xticks_rotation=45)
plt.title("Confusion Matrix - Validation Set")
plt.show()

# Classification Report
print("\nClassification Report:\n")
print(classification_report(true_labels, pred_labels, target_names=class_names))






from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import numpy as np

# --- ROC Curve & AUC ---
# Get model predictions on validation set
model.eval()
y_true = []
y_scores = []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        outputs = model(imgs)  # raw logits
        probs = torch.softmax(outputs, dim=1)  # convert to probabilities

        y_true.extend(labels.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_scores = np.array(y_scores)

# Binarize labels for multi-class ROC
n_classes = y_scores.shape[1]
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

# Compute ROC curve and AUC for each class
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Micro-average ROC (aggregate)
fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_scores.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# --- Plot ROC ---
plt.figure(figsize=(8, 6))
for i in range(n_classes):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"Class {i} (AUC = {roc_auc[i]:.2f})")

plt.plot(fpr["micro"], tpr["micro"],
         label=f"Micro-average (AUC = {roc_auc['micro']:.2f})",
         color="navy", linestyle="--", linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', lw=2)  # diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve and AUC")
plt.legend(loc="lower right")
plt.show()







                   
                   

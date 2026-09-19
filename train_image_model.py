import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Hyperparameters and Config
DATA_DIR = 'data'
BATCH_SIZE = 32
NUM_EPOCHS = 50
LEARNING_RATE = 0.0001
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def build_resnet50_model():
    """Create a ResNet50 model with the best torchvision API available."""
    resnet50_fn = models.resnet50

    if hasattr(models, "ResNet50_Weights"):
        try:
            return resnet50_fn(weights=models.ResNet50_Weights.DEFAULT)
        except Exception:
            pass

    try:
        return resnet50_fn(pretrained=True)
    except TypeError:
        return resnet50_fn()

def main():
    print(f"Using device: {DEVICE}")
    if DEVICE.type == 'cpu':
        print("\nWARNING: PyTorch is not detecting your RTX 4060 CUDA GPU!")
        print("To fix this, install the CUDA version of PyTorch with the following command:")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121\n")

    # 1. Data Augmentation and Normalization
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Loading Data
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
                      for x in ['train', 'valid', 'test']}
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, shuffle=(x == 'train'), num_workers=4)
                   for x in ['train', 'valid', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Classes: {class_names}")
    print(f"Dataset sizes: {dataset_sizes}")

    # 3. Model Setup (ResNet50 Transfer Learning)
    model = build_resnet50_model()
    
    # Fine-tuning: we DO NOT freeze layers, so the whole model trains
    # Replace the final fully connected layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, len(class_names))
    
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    # Optimize ALL parameters for fine-tuning
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Add a learning rate scheduler to decay LR by half every 10 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    # 4. Training Loop
    best_acc = 0.0
    
    for epoch in range(NUM_EPOCHS):
        print(f'Epoch {epoch+1}/{NUM_EPOCHS}')
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # deep copy the model if it's the best validation accuracy
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_chest_ct_model.pth')

        # Step the learning rate scheduler at the end of each epoch
        scheduler.step()

    print(f'Best valid Acc: {best_acc:4f}')
    print("Training complete. Model weights saved to 'best_chest_ct_model.pth'.")

    # 5. Evaluation on Test Set
    print("\n--- Evaluating on Test Set ---")
    model.load_state_dict(torch.load('best_chest_ct_model.pth'))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))
    
    # 6. Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix - Chest CT Classification')
    plt.tight_layout()
    plt.savefig('chest_ct_confusion_matrix.png')
    print("Confusion matrix saved to 'chest_ct_confusion_matrix.png'.")

if __name__ == '__main__':
    main()

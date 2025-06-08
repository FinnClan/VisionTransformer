import matplotlib.pyplot as plt
import torch.nn as nn
import torch
from sklearn.metrics import accuracy_score

# -----------------------
# Training Function
# -----------------------
def train_model(device ,model, train_loader, epochs=50):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Set the model to training mode
    model.train()

    loss_history = []
    acc_history = []

    for epoch in range(epochs):
        running_loss = 0.0
        all_preds, all_labels = [], []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            # Perform forward pass on the model
            outputs = model(images)

            # Calculate loss
            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            running_loss += loss.item()

        # Compute accuracy for this epoch
        acc = accuracy_score(all_labels, all_preds)
        avg_loss = running_loss / len(train_loader)

        loss_history.append(avg_loss)
        acc_history.append(acc)

        print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}, Accuracy = {acc:.4f}")

    # Plot loss and accuracy
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Loss')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, label='Accuracy', color='green')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_curves.png")  # Save to file
    plt.show()

    return model
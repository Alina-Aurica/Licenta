import torch

def validate_model(model, validationLoader, criterion):
    # the model is in evaluation mode
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    # no gradients needed for validation
    with torch.no_grad():
        for inputs, labels in validationLoader:
            # forward pass
            outputs = model(inputs)
            # calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            # Predictions
            _, predicted = torch.max(outputs, 1)
            total_correct += (predicted == labels).sum().item()
            total_samples += labels.size(0)

    avg_loss = total_loss / len(validationLoader)
    accuracy = total_correct / total_samples * 100

    print(f'Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')

    # the model is back to training mode
    model.train()

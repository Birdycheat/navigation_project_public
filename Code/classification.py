import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from utils import load_to_array, display_repartition, display_repartition_freq
import matplotlib.pyplot as plt
from random import shuffle as mix

def normalize(X, xi, xs):
    return (X-xi)/(xs-xi)

def load_data():
    Nwindows = 10
    ratio_train = 0.8
    windows = np.random.randint(100, 1000, size = Nwindows)
    X_train_list = []
    y_train_list = []
    X_test_list = []
    y_test_list = []

    freq_classes_train = np.array([0, 0, 0])
    freq_classes_test = np.array([0, 0, 0])
    
    number_of_file = 0
    # count the number of files and store it
    files = []
    for folder in os.listdir("Data/Clean"):
        folder_path = "Data/Clean/" + folder
        for file in os.listdir(folder_path):
            number_of_file += 1
            file_path = folder_path + "/" + file
            files.append(file_path)
    mix(files)
    limit_train = int(np.floor(number_of_file*ratio_train))
    file_number = 0
    for file_path in files:
        file_number += 1
        if "stair" in file_path:
            classe = np.array([1, 0, 0])
        if "walk" in file_path:
            classe = np.array([0, 1, 0])
        if "bike" in file_path:
            classe = np.array([0, 0, 1])
        np_gyr = load_to_array(file_path)
        
        X = np_gyr[1:, :]
        X = normalize(X, np.min(X), np.max(X_max))
        y = classe.reshape((-1, 1))*np.ones((np_gyr.shape[1]))
        if (file_number < limit_train):
            for window in windows:
                X_train_list += np.split(X.T, np.arange(1, np.ceil(X.shape[1]/window), dtype=int)*window)
                y_train_list += np.split(y.T, np.arange(1, np.ceil(X.shape[1]/window), dtype=int)*window)
                freq_classes_train += classe*X.shape[1]
        else:
            for window in windows:
                X_test_list += np.split(X.T, np.arange(1, np.ceil(X.shape[1]/window), dtype=int)*window)
                y_test_list += np.split(y.T, np.arange(1, np.ceil(X.shape[1]/window), dtype=int)*window)
                freq_classes_test += classe*X.shape[1]
    
    
        


    display_repartition_freq(freq_classes_train, title = "répartition des classes dans les données d'entraînement")
    display_repartition_freq(freq_classes_test, title = "répartition des classes dans les données de test")
    
    train_loader = (X_train_list, y_train_list)
    test_loader = (X_test_list, y_test_list)
    return train_loader, test_loader



class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(LSTMClassifier, self).__init__()
        self.hidden_size = hidden_size   
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.input_size = input_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, num_classes)
        self.fc2 = nn.Softmax(dim = 1)

    def forward(self, x, device):
        # Reshape input data to (batch_size, seq_length, input_size)
        x = x.view(-1, 1, self.input_size).to(device)
        
        
        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        # Forward propagate LSTM
        
        out, (_, _) = self.lstm(x, (h0, c0))

        out = self.fc1(out[:,-1,:])
        #
        out = self.fc2(out)
        
        return out



def train(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in zip(train_loader[0], train_loader[1]):
        inputs, labels = torch.from_numpy(inputs).to(device).to(torch.float32), torch.from_numpy(labels).to(device)

        optimizer.zero_grad()
        outputs = model(inputs, device).to(device)

        loss = criterion(outputs, torch.argmax(labels, dim=1)).to(device)
        loss.backward()
        optimizer.step()
        
        _, indexes = torch.max(outputs.data, 1)
        predicted = torch.zeros((indexes.size(0), model.num_classes), dtype = int).to(device)
        predicted[:, indexes] = 1
        correct += torch.sum(torch.sum(torch.square(predicted - labels), 1) == 0).item()
        
        total += labels.size(0)

        running_loss += loss.item()
    accuracy = 100 * correct / total
    return running_loss / len(train_loader[0]), accuracy

def evaluate(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in zip(test_loader[0], test_loader[1]):
            inputs, labels = torch.from_numpy(inputs).to(device).to(torch.float32), torch.from_numpy(labels).to(device)
            
            outputs = model(inputs, device).to(device)
            loss = criterion(outputs, torch.argmax(labels, dim=1)).to(device)

            running_loss += loss.item()
            
            _, indexes = torch.max(outputs.data, 1)
            predicted = torch.zeros((indexes.size(0), model.num_classes), dtype = int).to(device)
            predicted[:, indexes] = 1
            correct += torch.sum(torch.sum(torch.square(predicted - labels), 1) == 0).item()
            
            total += labels.size(0)

    accuracy = 100 * correct / total
    return running_loss / len(test_loader[0]), accuracy


def predict(model, inputs, device):
    model.eval()
    inputs = torch.from_numpy(inputs).float().to(device)
    with torch.no_grad():
        outputs = model(inputs, device)
        _, indexes = torch.max(outputs.data, 1)
        predicted = torch.zeros((indexes.size(0), model.num_classes)).to(device)
        predicted[:, indexes] = 1
    return predicted.cpu().numpy()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader, test_loader = load_data()

    input_size = 9
    hidden_size = 200
    num_layers = 1
    num_classes = 3
    num_epochs = 1000
    learning_rate = 0.001

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    epochs = []
    train_losses = []
    test_losses = []
    test_accuracies = []
    train_accuracies = []
    

    fig, (ax1, ax2) = plt.subplots(2, 1)
    for epoch in range(num_epochs):
        train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion, device)
        
        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        epochs.append(epoch)
        
        ax1.clear()
        ax1.plot(epochs, train_losses, label="Train Loss")
        ax1.plot(epochs, test_losses, label="Test Loss")
        ax1.legend()

        ax2.clear()
        ax2.plot(epochs, train_accuracies, label = "Train Accuracy")
        ax2.plot(epochs, test_accuracies, label = "Test Accuracy")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("%")
        ax2.legend()

        fig.canvas.draw()
        plt.pause(0.001)
        
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {train_loss:.4f}, '
              f'Test Loss: {test_loss:.4f}, '
              f'Test Accuracy: {test_accuracy:.2f}%')
    # Save the trained model
    torch.save(model.state_dict(), 'lstm_classifier.pth')


if __name__ == "__main__":
    main()



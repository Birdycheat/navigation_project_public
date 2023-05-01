import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from classification import LSTMClassifier, evaluate, normalize, predict
from utils import display_npgyr
from os import listdir


X_max = 22.55206892968221
X_min = -4.552975998940403
def load_new_data(file_path):
    loaded_data = np.load(file_path)
    X = np.block([[loaded_data['t']], [loaded_data['x']]])
    X[1:, :] = normalize(loaded_data['x'], X_max, X_min)
    return X

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Replace these by the model previously trained parameters
    input_size = 6
    hidden_size = 200
    num_layers = 1
    num_classes = 3

    model = LSTMClassifier(input_size, hidden_size, num_layers, num_classes).to(device)
    model.load_state_dict(torch.load('lstm_classifier.pth'))
    model.eval()

    names = ['AG', 'MD', 'GB']
    extension = ["npz"]
    
    for name in names:
        directory = "Data/Clean/" + name
        for file in listdir(directory):
            if file.split(".")[-1] in extension:
                new_data_file = directory + '/' + file
                print(new_data_file)
                X = load_new_data(new_data_file)
                predictions = predict(model, X[1:, :], device)
                display_npgyr(X, one_hot = True, labelisation=True, labels=predictions)

if __name__ == "__main__":
    main()

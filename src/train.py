import os
import glob
import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from torch.nn.utils.rnn import pad_sequence

from sklearn import metrics

import custom_dataset
import training
from model import DigitModel
from improved_model import DigitModel_improved
import json

from torch import nn
from torchsummary import summary
import re






def extract_numbers(input_string):
    
    numbers = re.findall(r'\d', input_string)

    
    numbers = list(map(int, numbers))

    return numbers

def calculate_accuracy(predictions, targets):
    correct_predictions = 0

    for pred, target in zip(predictions, targets):
        pred_list = extract_numbers(pred)
        target_list = target.tolist() if isinstance(target, torch.Tensor) else list(target)

        if len(pred_list) == len(target_list) and all(a == b for a, b in zip(pred_list, target_list)):
            correct_predictions += 1

    accuracy = correct_predictions / len(predictions)
    return accuracy

def remove_duplicates(x):
    if len(x) < 2:
        return x
    fin = ""
    for j in x:
        if fin == "":
            fin = j
        else:
            if j == fin[-1] and j != '*':
                continue
            else:
                fin += j
    return fin

def decode_predictions(preds):
    preds = preds.permute(1, 0, 2)
    preds = torch.softmax(preds, 2)
    preds = torch.argmax(preds, 2)
    preds = preds.detach().cpu().numpy()
    cap_preds = []

    for j in range(preds.shape[0]):
        temp = []
        for k in preds[j, :]:
            k = k.item()
            if k == 10:
                temp.append('*')  # Replace 10 with '*'
            else:
                temp.append(str(k))
        tp = "".join(temp)
        cap_preds.append(remove_duplicates(tp))  # Remove duplicates 

    return cap_preds





def collate_fn(batch):
    # Sort the batch by the length of the image sequence (descending order)
    batch.sort(key=lambda x: len(x["images"]), reverse=True)
    images = [item["images"] for item in batch]
    labels = [item["labels"].numpy().tolist() for item in batch]  # Convert to list

    # Pad the image sequences
    padded_images = pad_sequence(images, batch_first=True)

    # Pad labels with (blank) manually
    max_len = max(len(label) for label in labels)
    padded_labels = torch.full((len(labels), max_len), 10, dtype=torch.long)
    for i, label in enumerate(labels):
        padded_labels[i, :len(label)] = torch.tensor(label, dtype=torch.long)  # Convert to tensor

    return {
        "images": padded_images,
        "labels": padded_labels
    }

# Uncomment below and specify the desired data for train and val
#train_size = 2000
#val_size = 200
def run_training():
    # Define your transform, including resizing
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Image and Label directories:
    root_directory_train = "../train_data/combined_images_mnist"
    json_file_train = "../train_data/train_labels.json"
    root_directory_val = "../data/task1_val"
    json_file_val = "../gt/task1_val.json"

    #root_directory_test = "../data/task1_val"
    #json_file_val_test = "../gt/task1_val.json"

    # Create a custom dataset instance for train and validation
    train_dataset = custom_dataset.CustomDataset(
        root_directory_train,
        json_file_train,
        resize=(28, 140),
        transform=transform,
    )
    # Uncomment below for using a portion of specified data for training 
    #train_subset = Subset(train_dataset, list(range(train_size)))
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_dataset = custom_dataset.CustomDataset(
        root_directory_val,
        json_file_val,
        resize=(28, 140),
        transform=transform,
    
    )

    # Uncomment below for using a portion of specified data for val
    #val_subset = Subset(val_dataset, list(range(val_size)))

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        collate_fn=collate_fn
    )
    

    #for batch in train_loader:
    #    images, labels = batch['images'], batch['labels']
    #    print("Sample Images:", images.shape)
    #    print("Sample Labels:", labels)
    #    break

    val_targets_orig = [sample['labels'] for sample in val_loader.dataset]

    model = DigitModel(num_chars=11)  
    model.to("cpu")
    summary(model, input_size=(3, 28, 140))

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.8, patience=5, verbose=True
    )
    for epoch in range(200):
        train_loss = training.train_fn(model, train_loader, optimizer)
        print(f"Epoch {epoch + 1}/{10}, Train Loss: {train_loss}")
        valid_preds, test_loss = training.eval_fn(model, val_loader)

        #for i in range(min(5, len(valid_preds))):
        #    print("Ground truth labels:", val_targets_orig[i])
        #    decoded_preds = decode_predictions(valid_preds[i])
        #    print("Model predictions:", decoded_preds)

        valid_digit_preds = []
        for vp in valid_preds:
            current_preds = decode_predictions(vp)  
            valid_digit_preds.extend(current_preds)
        #combined = list(zip(val_targets_orig, valid_digit_preds))
        #print(combined[:20])

        accuracy = calculate_accuracy(valid_digit_preds, val_targets_orig)
        #print("Target Types:", type(val_targets_orig[0]))
        #print("Prediction Types:", type(valid_digit_preds[0]))

        #print("Target len:", len(val_targets_orig))
        #print("Prediction len:", len(valid_digit_preds))

        #print("Target Values:", val_targets_orig)
        #print("Prediction Values:", valid_digit_preds)

        #print("Target len:", len(val_targets_orig[0]))
        #print("Prediction len:", len(valid_digit_preds[0]))

        flat_val_targets_orig = [item for sublist in val_targets_orig for item in sublist.numpy()]
        flat_valid_digit_preds = [item for sublist in valid_digit_preds for item in sublist]

        # Convert the lists to numpy arrays
        flat_val_targets_orig = np.array(flat_val_targets_orig)
        flat_valid_digit_preds = np.array(flat_valid_digit_preds)
        
        # Now, calculate accuracy
        #accuracy = metrics.accuracy_score(flat_val_targets_orig, flat_valid_digit_preds)
        print(
            f"Epoch={epoch}, Train Loss={train_loss}, Test Loss={test_loss} , Accuracy={accuracy} "
        )
        scheduler.step(test_loss)
        #Accuracy={accuracy}

if __name__ == "__main__":
    run_training()

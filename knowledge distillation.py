import os
import sys
import json
from tqdm import tqdm
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
from model import vgg

def run_eval(model, dataloader, device):
    model.eval()
    loss_func = nn.CrossEntropyLoss()
    acc_list, loss_list = [], []
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(tqdm(dataloader)):
            inputs, labels = inputs.float().to(device), labels.to(device)
            preds= model(inputs)
            pred_idx = preds.max(1).indices
            acc = (pred_idx == labels).sum().item() / labels.size(0)
            acc_list.append(acc)
            loss = loss_func(preds, labels).item()
            loss_list.append(loss)

    final_loss = np.array(loss_list).mean()
    final_acc = np.array(acc_list).mean()

    return final_loss, final_acc
    

def run_finetune_distillation(student_model, teacher_model, train_dataloader, valid_dataloader, device, alpha=0.99, \
    temperature=8, n_epochs=2, learning_rate=1e-4, weight_decay=0.0):
    
    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate, weight_decay=weight_decay)       

    best_valid_acc = 0.0
    best_model = None
    for epoch in range(n_epochs):
        print('Start finetuning with distillation epoch {}'.format(epoch))
        loss_list = []

        # train
        student_model.train()
        for i, (inputs, labels) in enumerate(tqdm(train_dataloader)):
            optimizer.zero_grad()
            inputs, labels = inputs.float().to(device), labels.to(device)
            with torch.no_grad():
                teacher_preds = teacher_model(inputs)
            
            preds = student_model(inputs)
            soft_loss = nn.KLDivLoss()(F.log_softmax(preds / temperature, dim=1),
                                       F.softmax(teacher_preds / temperature, dim=1))
            hard_loss = F.cross_entropy(preds, labels)
            loss = soft_loss * (alpha * temperature * temperature) + hard_loss * (1. - alpha)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

        # validation
        valid_loss, valid_acc = run_eval(student_model, valid_dataloader, device)
        train_loss = np.array(loss_list).mean()
        print('Epoch {}: train loss {:.4f}, valid loss {:.4f}, valid acc {:.4f}'.format
              (epoch, train_loss, valid_loss, valid_acc))

        if valid_acc > best_valid_acc:
            best_valid_acc = valid_acc
            best_model = copy.deepcopy(student_model).to(device)

    print("Best validation accuracy: {}".format(best_valid_acc))

    student_model = best_model
    return student_model
        
        
def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
        "val": transforms.Compose([transforms.Resize((224, 224)),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])}

    # data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))  # get data root path
    data_root = os.path.abspath(os.path.join(os.getcwd()))
    
    print(data_root)
    image_path = os.path.join(data_root, "dataset")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"),
                                         transform=data_transform["train"])
    train_num = len(train_dataset)

    # {'cat':0, 'dog':1}
    animal_list = train_dataset.class_to_idx
    cla_dict = dict((val, key) for key, val in animal_list.items())
    # write dict into json file
    json_str = json.dumps(cla_dict, indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    batch_size = 32
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=nw)

    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform["val"])
    val_num = len(validate_dataset)
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size, shuffle=False,
                                                  num_workers=nw)
    print("using {} images for training, {} images for validation.".format(train_num,
                                                                           val_num))

    # teacher model
    teacher_model = vgg(model_name="vgg16", num_classes=2).to(device)
    # load model weights
    teacher_weights_path = "./vgg16Net.pth"
    assert os.path.exists(teacher_weights_path), "file: '{}' dose not exist.".format(teacher_weights_path)
    teacher_model.load_state_dict(torch.load(teacher_weights_path, map_location=device))
    teacher_model = teacher_model.to(device)
    
    # student model
    student_weights_path = "./pruned_vgg16_net.pth"
    assert os.path.exists(student_weights_path), "file: '{}' does not exist.".format(student_weights_path)
    # load model weights    
    student_model = torch.load(student_weights_path, map_location=device)
    student_model = student_model.to(device)
    
    # finetune_model
    finetuned_model = run_finetune_distillation(student_model, teacher_model, train_loader, train_loader, device, alpha=0.99, \
                        temperature=8, n_epochs=20, learning_rate=1e-4, weight_decay=0.0)
    
    # save model
    save_path = "./finetuned_vgg16_net.pth"
    torch.save(finetuned_model, save_path)
    
    print('Finished Training')


if __name__ == '__main__':
    main()
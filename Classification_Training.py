from self_transformers import Albumentations
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os
import matplotlib.pyplot as plt
import argparse
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
from earlystopping import EarlyStopping
from datetime import datetime

def load_data(data_dir):

    data_transforms = {
    'train': transforms.Compose([
        Albumentations(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
    ]),
    'val': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
        
    ]),

    'test': transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float)
        
    ]),

}


    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x]) for x in ['train', 'val', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=32, shuffle=True) for x in ['train', 'val', 'test']}
    
    print(image_datasets)

    return image_datasets, dataloaders

def create_model(model, device, weights = None):
    
    model = model(weights = weights) if weights != None else model()

    #Finetune Final few layers to adjust for tiny imagenet input
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(nn.Linear(num_ftrs, 1), nn.Sigmoid())
    model.to(device)
    
    return model


def test(dataloader, model, loss_fn):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test_losses = []
    results = []
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device, dtype=torch.float)
            pred = model(X)
            test_loss += loss_fn(pred, y.reshape(-1,1)).item()
            preds = [1 if i > 0.5 else 0 for i in pred]
            label = [1 if i > 0.5 else 0 for i in y.reshape(-1,1)]
            correct += sum([1 if preds[i] == label[i] else 0 for i in range(0, len(preds))])

    test_loss /= num_batches
    correct /= size
    test_losses.append(test_loss)
            
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    return test_loss, model



def train(dataloader, model, loss_fn, optimizer):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    train_losses = []

    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):

        X, y = X.to(device), y.to(device, dtype=torch.float)

        # Compute prediction error
        output = model(X)

        # Calculate loss
        loss = loss_fn(output, y.reshape(-1,1))

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            train_losses.append(loss)



def run_training(data_dir, model, model_name, loss_fn, optimizer, epochs = 50, patience = 10):
    
    early_stopping = EarlyStopping(patience=patience, verbose=True, path=model_name+'_best.pt')

    image_datasets, dataloaders = load_data(data_dir)

    for t in range(epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(dataloaders['train'], model, loss_fn, optimizer)
        test_loss, model_chkpnt = test(dataloaders['val'], model, loss_fn)
        
        #Earlystopping
        early_stopping(test_loss, model_chkpnt) 
        if early_stopping.early_stop:
            print("Early stopping")
            break
            
        # Release memory
        torch.cuda.empty_cache()
        
    print("Done!")
    print("Weight saved at runs/train/"+opt.name+"/weights/")
    print(device)


    torch.save(model, model_name+'_last.pt')
    print("Model saved at runs/train/"+opt.name+"/")

    eval(model, image_datasets['test'], 0.5)


def eval(last_model, test_data, thres):

    print("Evaluating.....")
    
    predicted = []
    actual = []

    for i in range(0, len(test_data)):
        x, y = test_data[i][0], test_data[i][1]
        x=x.to(device)
        last_model.eval()
        with torch.no_grad():
            pred = last_model(x.unsqueeze(0))
            predicted.append([1 if x > thres else 0 for x in pred][0])
            actual.append(y)

    #Generate the confusion matrix for test set
    cf_matrix = confusion_matrix(actual, predicted)

    ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='g')

    ax.set_title('Confusion Matrix with labels for Test Data\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Save the results
    plt.savefig("runs/train/"+opt.name+"/confusion_matrix.jpg")
    
    print("Completed!")
    


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=str(datetime.now()), help='model name')
    parser.add_argument('--model', type=str, default='resnet18', help='model type')
    parser.add_argument('--epoch', type=int, default=5, help='model type')
    parser.add_argument('--patience', type=int, default=10, help='num of patience')
    opt = parser.parse_args()

    #create parent folder
    model_path = "runs/train/"+opt.name+"/weights/"
    if not os.path.exists(model_path):
        os.makedirs(model_path)

  

    if opt.model == 'resnet50':
        from torchvision.models import resnet50 as resnet, ResNet50_Weights as resnet_weights

    elif opt.model == 'resnet34':
        from torchvision.models import resnet34 as resnet, ResNet34_Weights as resnet_weights

    elif opt.model == 'resnet18':
        from torchvision.models import resnet18 as resnet, ResNet18_Weights as resnet_weights

    #Create Resnet50 model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = create_model(resnet, device, resnet_weights.IMAGENET1K_V1)

    #Following is the loss function and optimization used for baseline model
    #Loss Function
    loss_fn = nn.BCELoss()

    # Observe that all parameters are being optimized
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    model_name = model_path+opt.name+'_'+ opt.model
    
    run_training('/data/yihwee/MinorTampering/3d_tampering/train_test_val_4', model, model_name, loss_fn, optimizer, opt.epoch, opt.patience)
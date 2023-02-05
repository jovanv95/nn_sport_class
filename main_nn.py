# %%
import torch
import torchvision
import torchvision.transforms as transforms

import os
import matplotlib.pyplot as plt
import numpy as np

# path to the data


sport_data_set = './images'




# %%
print(len(os.listdir('./images')))

# %%
# transform for mean and std

sport_data_set_ms = transforms.Compose([transforms.ToTensor(),transforms.Resize((225,225))])


# %%
# applying transformation and selecting the data 

sport_data_set = torchvision.datasets.ImageFolder(root = sport_data_set , transform= sport_data_set_ms )
sport_data_set

# %%

n = len(sport_data_set)  # total number of examples
n_test = int(0.15 * n)  # take ~10% for test
test_set = torch.utils.data.Subset(sport_data_set, range(n_test))  # take first 10%
train_set = torch.utils.data.Subset(sport_data_set, range(n_test, n))  # take the rest   

# %%



# %%
# loading the dataset, need to do so in batches or else we run out of RAM

train_loader_ms = torch.utils.data.DataLoader(dataset = train_set, batch_size = 32, shuffle=False)


# %%
# function for calculating std and mean

def get_mean_and_std(loader):
    mean = 0.
    std = 0.
    total_img_count = 0
    # looping thrue each batch
    for images, _ in loader:
        # number of images in batch
        images_count_in_batch = images.size(0)
        # resizeing the image tensor in the batch in order to reduce the dimensions of the tensor form 4 to 3
        images = images.view(images_count_in_batch, images.size(1), -1)
        mean += images.mean(2).sum(0)
        std += images.std(2).sum(0)
        total_img_count += images_count_in_batch

    mean /= total_img_count
    std /= total_img_count

# return a proxy mean and std , we cant get the real one because we cant load the whole data set, so we calculate the avrage for each batch and then the avrage for all the batches 
    return mean,std



# %%
# returns the mean and std

mean , std = get_mean_and_std(train_loader_ms)



# %%
train_trans = transforms.Compose([transforms.ToTensor(), transforms.RandomHorizontalFlip(),transforms.RandomRotation(10),transforms.Normalize(mean,std),transforms.Resize((224,224))])

test_trans = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean,std),transforms.Resize((224,224))])

# %%
test_set.dataset.transform = test_trans
train_set.dataset.transform = train_trans

# %%
train_dataset = train_set
test_dataset = test_set

# %%
def show_trans_img(dataset):
    loader = torch.utils.data.DataLoader(dataset,batch_size = 6,shuffle=True)
    batch = next(iter(loader))
    Images,lables = batch
    grid = torchvision.utils.make_grid(Images,nrow=3)
    plt.figure(figsize=(11,11))
    plt.imshow(np.transpose(grid,(1,2,0)))
    print('lables:', lables)



# %%

#Ovde povecavaj batchove dok ti graficka ne ode na maksimum

batches = 32

train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=batches,shuffle=True)
test_loader = torch.utils.data.DataLoader(train_set,batch_size=batches,shuffle=True)

# %%
def set_device():
    if torch.cuda.is_available():
        dev = 'cuda:0'
    else:
        dev = 'cpu'
    return torch.device(dev)


# %%
import torchvision.models as models
import torch.nn as nn
import torch.optim as optim
#chose a model, but set weights to None so you can train it yourself
resnet_18_model = models.resnet18(weights=None)

num_ftrs = resnet_18_model.fc.in_features
number_of_classes = len(os.listdir('./images'))
resnet_18_model.fc = nn.Linear(num_ftrs,number_of_classes)
device = set_device()
resnet_18_model = resnet_18_model.to(device)
loss_fn = nn.CrossEntropyLoss()
# lr 0.01 to 0.1 experminet whit it
# momenntum makes gradient desecnt faster
# weight_decay extra error to loss function , prevents overfiting
optimizer = optim.SGD(resnet_18_model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.003)

# %%
def evaluate_model_on_test_set(model,test_loader):
    model.eval()
    predicted_correctly_on_epoch = 0
    total = 0
    device =set_device()

    with torch.no_grad():
        for data in test_loader:
            images , lables = data
            images = images.to(device)
            lables = lables.to(device)
            total += lables.size(0)

            outputs = model(images)

            _ , predicted = torch.max(outputs.data,1)

            predicted_correctly_on_epoch += (predicted == lables).sum().item()
    
    epoch_acc = 100.0 * predicted_correctly_on_epoch / total
    print('     -Test dataset. Got %d out of %d images correctly(%.3f%%)' % (predicted_correctly_on_epoch,total, epoch_acc))

# %%
def save_checkpoint(state,filename = 'model_checkpoint.pth.tar'):
    print('=> Saveing checkpoint')
    torch.save(state,filename,_use_new_zipfile_serialization=False)




# %%
def load_checkpoint(checkpoint):
    print('=> Loading checkpoint')
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['state_dict'])
    epoch.load_state_dict(checkpoint['state_dict'])


# %%
def train_nn(model,train_loader,test_loader,criterion,optimizer,n_epoch):
    device = set_device()
    save_number = 0
    for epoch in range(n_epoch):
        print('Epoch number %d' % (epoch + 1))
        model.train()
        running_loss = 0.0
        running_correct = 0.0
        total = 0
        checkpoint = {'state_dict' : model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        save_checkpoint(checkpoint)
        save_checkpoint(checkpoint,filename=f'model_epoch_{save_number}')
        save_number += 1
        for data in train_loader:
            images , lables = data
            images = images.to(device)
            lables = lables.to(device)
            total += lables.size(0)

            optimizer.zero_grad()

            outputs = model(images)

            _ , predicted = torch.max(outputs.data,1)

            loss = criterion(outputs,lables)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_correct += (lables == predicted).sum().item()

        epoch_loss = running_loss/len(train_loader)
        epoch_acc = 100.0 * running_correct / total

        print("         -Training dataset. Got %d out of %d images correctly(%.3f%%). Epoch loss: %.3f" % (running_correct,total,epoch_acc,epoch_loss))
        evaluate_model_on_test_set(model,test_loader)
        
    print("Finished")

# %%
train_nn(resnet_18_model,train_loader,test_loader,loss_fn,optimizer,100)



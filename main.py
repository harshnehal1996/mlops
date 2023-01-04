import argparse
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import click
from global_vars import isTrain
from data import mnist
from model import MyAwesomeModel


@click.group()
def cli():
    pass


@click.command()
@click.option("--lr", default=1e-3, help='learning rate to use for training')
def train(lr):
    print("Training day and night")
    print(lr)

    # TODO: Implement training loop here
    model = MyAwesomeModel()
    model.cuda()
    dataset, _ = mnist()
    train_set, val_set = torch.utils.data.random_split(dataset, [20000, 5000])
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.5, 0.999))
    trainloader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True, pin_memory=True)  
    valloader = torch.utils.data.DataLoader(val_set, batch_size=64)

    epoch = 40
    loss_fn = nn.NLLLoss()

    for i in range(epoch):
        global isTrain 

        running_train_loss = 0
        running_val_loss = 0
        model.train()
        isTrain = True
        
        for images, labels in trainloader:        
            optimizer.zero_grad()
            output = model(images.cuda())
            loss = loss_fn(output, labels.cuda())
            loss.backward()
            running_train_loss += loss.item()
            optimizer.step()
        
        model.eval()
        isTrain = False
        with torch.no_grad():
            for images, labels in valloader:
                output = model(images.cuda())
                loss = loss_fn(output, labels.cuda())
                running_val_loss += loss.item()
            
            if i % 5 == 0:
                print('saving model...')
                torch.save({
                    'epoch': i,
                    'state_dict': model.state_dict(),
                    'validation loss' : running_val_loss,
                    'train loss' : running_train_loss}, os.path.join(os.getcwd(), 'saved_models/model_' + str(i // 5)  + '.pth'))
            
        print('train_loss : ', running_train_loss / len(trainloader))
        print('val_loss : ', running_val_loss / len(valloader))


@click.command()
@click.argument("model_checkpoint")
def evaluate(model_checkpoint):
    print("Evaluating until hitting the ceiling")
    print(model_checkpoint)

    # TODO: Implement evaluation logic here
    model = MyAwesomeModel()
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'])
    
    _, test_dataset = mnist()
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=64)
    model.eval()
    global isTrain
    isTrain = False
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            output = model(images)
            prediction = torch.argmax(output, axis=-1)
            correct += torch.sum(prediction == labels).item()
            total += len(labels)
        
        print('accuracy : ', correct / total)

cli.add_command(train)
cli.add_command(evaluate)


if __name__ == "__main__":
    cli()


    
    
    
    
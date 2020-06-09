from __future__ import print_function
import os
import json
import argparse
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter

from polyaxon_client.tracking import Experiment, get_outputs_path, get_log_level, get_data_paths

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Linear(784, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.input(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(epoch, writer, experiment, args, model, device, train_loader, optimizer):
    model.train()
    mean_loss = 0
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        mean_loss += loss.sum().item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    dlen = len(train_loader.dataset)
    mean_loss /= dlen
    acc = 100. * correct / dlen
    template = 'Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
    experiment.log_metrics(train_loss=mean_loss, train_acc=acc)
    writer.add_scalar('loss/train', mean_loss, epoch)
    writer.add_scalar('acc/train', acc, epoch)
    logger = logging.getLogger('main')
    logger.info('%s', template.format(mean_loss, correct, dlen, acc))


def test(epoch, writer, experiment, args, model, device, test_loader):
    model.eval()
    mean_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            mean_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    dlen = len(test_loader.dataset)
    mean_loss /= dlen
    acc = 100. * correct / dlen
    template = 'Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'
    experiment.log_metrics(test_loss=mean_loss, test_acc=acc)
    writer.add_scalar('loss/test', mean_loss, epoch)
    writer.add_scalar('acc/test', acc, epoch)
    logger = logging.getLogger('main')
    logger.info('%s', template.format(mean_loss, correct, dlen, acc))

def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 1000)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=15, metavar='N',
                        help='number of epochs to train (default: 9)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    args = parser.parse_args()

    experiment = Experiment()
    logger = logging.getLogger('main')
    logger.setLevel(get_log_level())

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    logger.info('%s', device)

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(get_data_paths()['mnist'], train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(get_data_paths()['mnist'], train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)

    model = Net().to(device)
    model_path = os.path.join(get_outputs_path(), 'model.p')
    state_path = os.path.join(get_outputs_path(), 'state.json')

    start = 1

    if os.path.isfile(model_path):
        model.load_state_dict(torch.load(model_path))
        logger.info('%s', 'Model Loaded')
    if os.path.isfile(state_path):
        with open(state_path, 'r') as f:
            data = json.load(f)
            start = data['epoch']
        logger.info('%s', 'State Loaded')

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    with SummaryWriter(log_dir=get_outputs_path()) as writer:
        for epoch in range(start, args.epochs + 1):
            train(epoch, writer, experiment, args, model, device, train_loader, optimizer)
            test(epoch, writer, experiment, args, model, device, test_loader)
            torch.save(model.state_dict(), model_path)
            with open(state_path, 'w') as f:
                data = {
                    'epoch' : epoch
                }
                json.dump(data, f)

if __name__ == '__main__':
    main()
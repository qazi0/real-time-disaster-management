import os
import torch
import argparse
import torch.optim as optim

from torch.utils.data import DataLoader
from model.label_smoothing import LabelSmoothingCrossEntropy
from model.ernet import ErNET
from model.squeeze_ernet import Squeeze_ErNET
from model.squeeze_ernet_redconv import Squeeze_RedConv
from dataloaders.aider import AIDER, aider_transforms, squeeze_transforms
from pytorch_lightning.metrics import F1

saves_dir = 'saves'
debug = False
prev_acc = 0

def train(args, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):

        if debug:
            print('Data shape= ', data.shape, ' Target shape= ', target.shape)

        # target = torch.as_tensor(target, dtype=torch.long)
        if args.cuda:
            data, target = data.cuda(), target.cuda()

        optimizer.zero_grad()
        output = model(data)
        # output = output.cuda()

        if debug:
            print('dtype of model output = ', output.dtype)

        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data))


def test(args):
    model.eval()
    test_loss = 0
    correct = 0
    global prev_acc
    print('Test loader length: ', len(test_loader.dataset))

    with torch.no_grad():

        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()

            output = model(data)
            test_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    new_acc = 100. * correct.float() / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(test_loss, correct,
                                                                                 len(test_loader.dataset), new_acc))
    fixed_preds = pred.view(1, -1)
    f_score = F1(num_classes=5)
    score = f_score(fixed_preds[0].cpu(), target.cpu()) * 100
    print('\n F1_SCORE : {:.5f}'.format(score))


def validation_test(args, save_model=False):
    model.eval()
    valid_loss = 0
    correct = 0
    global prev_acc
    with torch.no_grad():
        for data, target in valid_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            valid_loss += criterion(output, target).data
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    valid_loss /= len(valid_loader.dataset)
    new_acc = 100. * correct.float() / len(valid_loader.dataset)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(valid_loss, correct,
                                                                                       len(
                                                                                           valid_loader.dataset),
                                                                                       new_acc))
    if new_acc > prev_acc:
        # save model
        if save_model:
            torch.save(model, save_location)
            print('Model saved at: ', save_location)
        prev_acc = new_acc


if __name__ == '__main__':

    torch.backends.cudnn.benchmark = True

    parser = argparse.ArgumentParser(description='PyTorch TRAINING on AIDER VOC')
    parser.add_argument('--model', type=str, default='ernet')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0001,
                        metavar='LR', help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5,
                        metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--root-dir', type=str, default='AIDER',
                        help='path to the root dir of AIDER')
    parser.add_argument('--weights', type=str, default=None,
                        help='path to the trained PyTorch weights (.pt) file')
    parser.add_argument('--eval', default=False, action='store_true',
                        help='perform evaluation of trained model')
    parser.add_argument('--resume', default=False, action='store_true',
                        help='resume training from last saved weights')
    parser.add_argument('--collab', default=False,
                        action='store_true', help='use dataset from google collab')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()

    print(args, end='\n\n')
    save_location = args.weights

    if not os.path.exists(saves_dir):
        os.mkdir(saves_dir)


    if args.model == 'ernet':
        model = ErNET()
        model_transforms = aider_transforms

        if not args.weights:
            save_location = os.path.join(saves_dir, 'ernet.pt')

    elif args.model == 'squeeze-ernet':
        model = Squeeze_ErNET()
        model_transforms = squeeze_transforms

        if not args.weights:
            save_location = os.path.join(saves_dir, 'squeeze-ernet.pt')

    elif args.model == 'squeeze-redconv':
        model = Squeeze_RedConv()
        model_transforms = squeeze_transforms

        if not args.weights:
            save_location = os.path.join(saves_dir, 'squeeze-redconv.pt')

    if args.cuda:
        torch.cuda.set_device(0)
        model.cuda()

    if args.collab:
        transformed_dataset = AIDER("dataloaders/aider_labels.csv", '/content/drive/My Drive/FYP/Dataset/AIDER/',
                                    transform=aider_transforms)
    else:
        transformed_dataset = AIDER(
            "dataloaders/aider_labels.csv", args.root_dir, transform=model_transforms)

    total_count = 6432
    train_count = int(0.5 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count
    train_set, valid_set, test_set = torch.utils.data.random_split(transformed_dataset,
                                                                   (train_count, valid_count, test_count))

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                              drop_last=True)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True,
                             drop_last=True)

    if debug:
        dataiter = iter(train_loader)
        images, labels = dataiter.next()
        print('Images.shape= ', images.shape)
        print('Labels.shape= ', labels.shape)

    criterion = LabelSmoothingCrossEntropy(reduction='sum')

    if args.eval:
        model = torch.load(os.path.join(save_location))
        test(args)
   
    else:
        if args.resume:
            model = torch.load(save_location)
            validation_test(args)
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        for epoch in range(1, args.epochs + 1):
            train(args, epoch)
            validation_test(args, save_model=True)
            if epoch % 40 == 0:
                optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * 0.0001

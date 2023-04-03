from __future__ import print_function

import os

import numpy as np
import pandas as pd
import scipy
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

import cmd_args
import model_mlp
import model_smallalexnet
import model_smallinception

from cifar10_data import CIFAR10RandomLabels
from cifar10_data import CIFAR10DatasetNoise
from mnist_like_data import MNISTRandomLabels
from mnist_like_data import MNISTDatasetNoise


print(f"Using PyTorch {torch.__version__}")


def showimg(img):
    npimg = img.numpy()
    plt.figure(figsize=(1, 1))
    plt.title(f"Img shape: {img.shape}")
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')
    plt.show()


class PerImageWhitening:
    """
    As described in the paper
    """
    def __init__(self):
        pass

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        debug = False

        mean = torch.mean(img, axis=(1, 2))
        std = torch.std(img, axis=(1, 2))
        std_in_case_std_is_zero = torch.Tensor([1. / np.sqrt(28 ** 2)] * 3)
        adjusted_stddev = torch.max(std, std_in_case_std_is_zero)

        new_img = F.normalize(img, mean=mean, std=adjusted_stddev)

        if debug:
            print(f"Mean: {mean}, shape: {mean.shape}")
            print(f"Adjusted std: {adjusted_stddev}, std: {std}, std if std zero: {std_in_case_std_is_zero}")
            showimg(img)

            mean = torch.mean(new_img, axis=(1, 2))
            std = torch.std(new_img, axis=(1, 2))
            std_in_case_std_is_zero = torch.Tensor([1. / np.sqrt(28 ** 2)] * 3)
            adjusted_stddev = torch.max(std, std_in_case_std_is_zero)

            print(f"New mean: {mean}, shape: {mean.shape}")
            print(f"New adjusted std: {adjusted_stddev}, std: {std}, std if std zero: {std_in_case_std_is_zero}")
            showimg(new_img)

        return new_img


def get_cifar10_data_loader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        PerImageWhitening()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        PerImageWhitening()
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.noise_type == 'dataset':    
        train_loader = torch.utils.data.DataLoader(
        CIFAR10DatasetNoise(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            n_components=args.n_components
                            ),
        batch_size=args.batch_size, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
        CIFAR10DatasetNoise(root='./data', train=False, download=True,
                            transform=transform_test, num_classes=args.num_classes,
                            n_components=args.n_components
                            ),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob,
                            gaussian=args.noise_type == 'gaussian',
                            shfld_pxls=args.noise_type == 'shuffle',
                            rnd_pxls=args.noise_type == 'random'),

        batch_size=args.batch_size, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
        CIFAR10RandomLabels(root='./data', train=False, download=True,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob,
                            gaussian=args.noise_type == 'gaussian',
                            shfld_pxls=args.noise_type == 'shuffle',
                            rnd_pxls=args.noise_type == 'random'),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, valid_loader


def get_cifar100_data_loader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        PerImageWhitening()
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.CenterCrop(28),
        PerImageWhitening()
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}
    train_loader = torch.utils.data.DataLoader(
        CIFAR100RandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob,
                            gaussian=args.noise_type == 'gaussian',
                            shfld_pxls=args.noise_type == 'shuffle',
                            rnd_pxls=args.noise_type == 'random'),

        batch_size=args.batch_size, shuffle=True, **kwargs)

    valid_loader = torch.utils.data.DataLoader(
        CIFAR100RandomLabels(root='./data', train=False, download=True,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob,
                            gaussian=args.noise_type == 'gaussian',
                            shfld_pxls=args.noise_type == 'shuffle',
                            rnd_pxls=args.noise_type == 'random'),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, valid_loader



def get_mnist_data_loader(args):
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    kwargs = {'num_workers': 1, 'pin_memory': True}

    if args.noise_type == 'dataset':    
        train_loader = torch.utils.data.DataLoader(
        MNISTDatasetNoise(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            n_components=args.n_components
                            ),
        batch_size=args.batch_size, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
        MNISTDatasetNoise(root='./data', train=False, download=True,
                            transform=transform_test, num_classes=args.num_classes,
                            n_components=args.n_components
                            ),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    else:
        train_loader = torch.utils.data.DataLoader(
        MNISTRandomLabels(root='./data', train=True, download=True,
                            transform=transform_train, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob,
                            gaussian=args.noise_type == 'gaussian',
                            shfld_pxls=args.noise_type == 'shuffle',
                            rnd_pxls=args.noise_type == 'random'),

        batch_size=args.batch_size, shuffle=True, **kwargs)

        valid_loader = torch.utils.data.DataLoader(
        MNISTRandomLabels(root='./data', train=False, download=True,
                            transform=transform_test, num_classes=args.num_classes,
                            corrupt_prob=args.label_corrupt_prob,
                            gaussian=args.noise_type == 'gaussian',
                            shfld_pxls=args.noise_type == 'shuffle',
                            rnd_pxls=args.noise_type == 'random'),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    return train_loader, valid_loader


def get_data_loaders(args):
    print(f'Loading {args.data} dataset.')
    if args.data == 'cifar10':
        train_loader, valid_loader = get_cifar10_data_loader(args)
    if args.data == 'cifar100':
        train_loader, valid_loader = get_cifar10_data_loader(args)
    elif args.data == 'mnist':
        train_loader, valid_loader = get_mnist_data_loader(args)

    return train_loader, valid_loader


def get_model(args):
    if args.data == 'mnist':
        input_dim = 28 * 28
    elif args.data == 'cifar10':
        input_dim = 28 * 28 * 3
    elif args.data == 'cifar100':
        input_dim = 28 * 28 * 3

    print(f'Initializing model {args.arch}')
    if args.arch == 'mlp':
        # create model
        n_units = [int(x) for x in args.mlp_spec.split('x')]  # hidden dims
        n_units.append(args.num_classes)  # output dim
        n_units.insert(0, input_dim)  # input dim
        
        model = model_mlp.MLP(n_units)
    elif args.arch == 'small-alexnet':
        model = model_smallalexnet.SmallAlexNet(args=args)
    elif args.arch == 'small-inception':
        model = model_smallinception.SmallInception(args=args)
    print(f'Done initializing model {args.arch}')

    if torch.cuda.is_available():
        print(f"Sending model {args.arch} to CUDA")
        model = model.cuda()
        print(f"Done sending model {args.arch} to CUDA")
    return model


def train_model(args, model, train_loader, val_loader):
    cudnn.benchmark = True

    # define loss function (criterion) and pptimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=args.momentum,
                                weight_decay=args.weight_decay)

    steps = 0
    step_history = {'steps': [], 'trn_loss': [], 'trn_prec1': [], 'val_loss': [], 'val_prec1': []}
    epoch_history = {'epoch': [], 'trn_loss': [], 'trn_prec1': [], 'val_loss': [], 'val_prec1': []}

    for epoch in range(args.first_epoch, args.epochs):
        if args.first_epoch !=0:
            model, _, _, _ = load_checkpoint(args, epoch=args.first_epoch)
            
        adjust_learning_rate(optimizer, epoch+args.first_epoch, args)

        # train for one epoch
        val_loss, val_prec1, tr_loss, tr_prec1, steps = train_epoch(train_loader, val_loader, model, criterion,
                                                                    optimizer,
                                                                    epoch, args,
                                                                    steps, step_history, epoch_history)

        save_checkpoint(args, model, epoch, optimizer)
        print(
            f'Epoch: {epoch:03d}: Acc-tr: {tr_prec1:6.2f}, Acc-val: {val_prec1:6.2f}, L-tr: {tr_loss:6.4f}, L-val: {val_loss:6.4f}')
        if tr_prec1>99. and epoch>=50:
          break
    return step_history, epoch_history


def train_epoch(train_loader, val_loader, model, criterion, optimizer, epoch, args, steps, step_history, epoch_history):
    """Train for one epoch on the training set"""
    # switch to train mode
    model.train()

    for i, (input, target) in enumerate(train_loader):
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        steps += 1

        if steps % 50 == 0 and False: #TODO: this should be removed
            val_loss, val_prec1 = eval_epoch(val_loader, model, criterion, epoch, args)
            tr_loss, tr_prec1 = eval_epoch(train_loader, model, criterion, epoch, args)
            print(f"steps: {steps:04d}, trn loss:{tr_loss:3.2f}, trn prec1:{tr_prec1:3.2f}")
            print(f"steps: {steps:04d}, val loss:{val_loss:3.2f}, val prec1:{val_prec1:3.2f}")

            # save step history
            step_history['steps'].append(steps)
            step_history['trn_loss'].append(tr_loss)
            step_history['trn_prec1'].append(tr_prec1)
            step_history['val_loss'].append(val_loss)
            step_history['val_prec1'].append(val_prec1)

    val_loss, val_prec1 = eval_epoch(val_loader, model, criterion, epoch, args)
    tr_loss, tr_prec1 = eval_epoch(train_loader, model, criterion, epoch, args)

    # save epoch history

    epoch_history['epoch'].append(epoch)
    epoch_history['trn_loss'].append(tr_loss)
    epoch_history['trn_prec1'].append(tr_prec1)
    epoch_history['val_loss'].append(val_loss)
    epoch_history['val_prec1'].append(val_prec1)

    # returns the average training loss, accuracy, for this epoch
    return val_loss, val_prec1, tr_loss, tr_prec1, steps


def eval_epoch(data_loader, model, criterion, epoch, args):
    """Evaluate model on data"""
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(data_loader):

        # send to CUDA if possible
        if torch.cuda.is_available():
            target = target.cuda(non_blocking=True)
            input = input.cuda()
        with torch.no_grad():
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target, topk=(1,))[0]
        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))

    return losses.avg, top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def print(self):
        print(self.avg)


def adjust_learning_rate(optimizer, epoch, args):
    """Decay factor of 0.95 per epoch, as in the paper"""
    lr = args.learning_rate * (0.95 ** epoch)  # a decay factor of 0.95
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_checkpoint(args, model, epoch, optimizer):
    exp_dir = os.path.join('models', args.exp_name + '/checkpoints')
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    print('Saving model into %s...' % exp_dir)
    save_path = exp_dir + f'/{args.name}_checkpoint_{epoch}.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(args, model=None, epoch=0, optimizer=None):
    if model:
        pass
    else:
        model = get_model(args)
    if optimizer:
        pass
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay)
    exp_dir = os.path.join('models', args.exp_name + '/checkpoints')
    load_path = exp_dir + f'/{args.name}_checkpoint_{epoch}.pth'
    print(f'Loading model from {load_path}')
    checkpoint = torch.load(load_path)
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print(f'Done loading model with all info')
        return model, optimizer, epoch, load_path
    except:
        model.load_state_dict(checkpoint['model_state_dict'])
        print('Done loading model')
        return model, 0, epoch, load_path


def save_history(args, step_history, epoch_history):
    exp_dir = os.path.join('models', args.exp_name + '/history')
    if not os.path.isdir(exp_dir):
        os.makedirs(exp_dir)
    df = pd.DataFrame(step_history)
    df.to_csv(exp_dir + f'/{args.name}_step_history.csv', index=False)
    df = pd.DataFrame(epoch_history)
    df.to_csv(exp_dir + f'/{args.name}_epoch_history.csv', index=False)


def get_jacobians_svs(model, args, data='train'):
    args.batch_size = 1
    svs = []
    
    train_loader, val_loader = get_data_loaders(args)

    if data == 'train':
        print("Computing Jacobian spectrum on the training set")
        loader = train_loader
    elif data == 'test':
        print("Computing Jacobian spectrum on the test set")
        loader = val_loader
    for i, (images, labels) in enumerate(loader):

        if torch.cuda.is_available():
            #print("Sent images, labels to cuda")
            labels = labels.cuda(non_blocking=True)
            images = images.cuda()

        jacobian = torch.autograd.functional.jacobian(model, images)
        spectrum = torch.linalg.svdvals(jacobian.squeeze().reshape(args.num_classes, -1)).cpu().numpy()

        svs.append(spectrum)
        if i % 10000 == 0:
            print(f"Computed Jacobian spectrum at {int(10000 + i)} points...")

    svs = np.array(svs).flatten()
    print(f"Mean spectrum {np.mean(svs)}")
    return svs


def calculate_spectra(args, data='both', delete_checkpoints=False):
    exp_dir = os.path.join('models', args.exp_name + '/spectra')
    if not os.path.isdir(exp_dir):
        try:
            os.makedirs(exp_dir)
        except:
            pass

    if data == 'both':
        for epoch in range(args.first_epoch, args.epochs):
            try:  # this is here to catch there being no model for some reason
                print("Loading checkpoint")
                checkpoint, _, _, checkpoint_path = load_checkpoint(args, epoch=epoch)

                print("Computing train spectrum")
                spectrum = get_jacobians_svs(checkpoint, args, 'train')
                print("Saving")
                np.savez_compressed(exp_dir + f'/{args.name}_train_spectrum_checkpoint_{epoch}.npz',
                                    spectrum)  # this one

                print("Computing test spectrum")
                spectrum = get_jacobians_svs(checkpoint, args, 'test')
                print("Saving")
                np.savez_compressed(exp_dir + f'/{args.name}_test_spectrum_checkpoint_{epoch}.npz',
                                    spectrum)  # this one

                print("Deleting checkpoint")
                if delete_checkpoints and epoch != args.epochs - 1:
                    os.remove(checkpoint_path)
            except:
                print("Couldn't find a model checkpoint from which to extract spectra.")

    else:  # if data is either train or test
        for epoch in range(args.first_epoch, args.epochs):
            try:
                checkpoint, _, _, checkpoint_path = load_checkpoint(args, epoch=epoch)
                spectrum = get_jacobians_svs(checkpoint, args, data)
                np.savez_compressed(exp_dir + f'/{args.name}_{data}_spectrum_checkpoint_{epoch}.npz',
                                    spectrum)  # this one
                if delete_checkpoints and epoch != args.epochs - 1:
                    os.remove(checkpoint_path)
            except:
                print("Couldn't find a model checkpoint from which to extract spectra.")



# calculate_spectra(args, data='both', delete_checkpoints=True)


def main():
    args = cmd_args.parse_args()
    if args.command == 'train':
        print("Loading data")
        train_loader, val_loader = get_data_loaders(args)
        print("Getting model")
        model = get_model(args)
        print("Training model")
        step_history, epoch_history = train_model(args, model, train_loader, val_loader)
        print("Saving history")
        save_history(args, step_history, epoch_history)

    if args.command == 'spectra':
        print("Calculating spectra and deleting checkpoints")
        calculate_spectra(args, data='both', delete_checkpoints=True)

    if args.command == 'plot':
        print("PLotting")
        plot_step_history(args, 'accuracy', start=args.thousand_step_start, stop=args.thousand_step_end)
        plot_step_history(args, 'loss', start=args.thousand_step_start, stop=args.thousand_step_end)
        plot_epoch_history(args, 'accuracy', start=args.epoch_start, stop=args.epoch_end)
        plot_epoch_history(args, 'loss', start=args.epoch_start, stop=args.epoch_end)

    if args.command == 'both':
        print("Training model, calculating spectra, deleting checkpoints")
        print("Loading data")
        train_loader, val_loader = get_data_loaders(args)
        print("Getting model")
        model = get_model(args)
        print("Training model")
        step_history, epoch_history = train_model(args, model, train_loader, val_loader)
        print("Saving history")
        save_history(args, step_history, epoch_history)
        print("Calculating spectra and deleting checkpoints")
        calculate_spectra(args, data='both', delete_checkpoints=True)



if __name__ == '__main__':
    main()

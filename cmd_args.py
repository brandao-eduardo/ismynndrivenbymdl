import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--n-components', type=int, default=0)
parser.add_argument('--command', default='train', choices=['train', 'spectra','plot', 'both'])
parser.add_argument('--data', default='cifar10', choices=['cifar10', 'mnist'])
parser.add_argument('--num-classes', type=int, default=10)
parser.add_argument('--data-augmentation', type=bool, default=False)
parser.add_argument('--thousand-step-start', type=int, default=0)
parser.add_argument('--thousand-step-end', type=int, default=-1)
parser.add_argument('--epoch-start', type=int, default=0)
parser.add_argument('--epoch-end', type=int, default=-1)
parser.add_argument('--noise-type', default='label', choices=['label', 'shuffle', 'random', 'gaussian', 'dataset'])
parser.add_argument('--label-corrupt-prob', type=int, default=0)

parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--first-epoch', type=int, default=0)
parser.add_argument('--learning-rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=0.)


parser.add_argument('--arch', default='mlp', choices=['mlp', 'small-alexnet', 'small-inception'])
parser.add_argument('--mlp-spec', default='512', help='mlp spec: e.g. 512x128x512 indicates 3 hidden layers')
parser.add_argument('--name', default='run3', help='Experiment name')


def format_experiment_name(args):
  name = ''#args.name
  if name != '':
    name += '_'
    
  name += args.data + '_'    

  if args.noise_type =='label':
    if args.label_corrupt_prob > 0:
      name += f'corrupt{args.label_corrupt_prob/100.}'

  else:
    name += args.noise_type
    name += '_'
    name += f'ncomp{args.n_components}_'

  name += args.arch
  if args.arch == 'mlp':
    name += args.mlp_spec
    
  name += '_lr{0}_mmt{1}'.format(args.learning_rate, args.momentum)
  if args.weight_decay > 0:
    name += '_Wd{0}'.format(args.weight_decay)
  else:
    name += '_NoWd'
  if not args.data_augmentation:
    name += '_NoAug'

  return name


def parse_args():
  args = parser.parse_args()
  args.exp_name = format_experiment_name(args)
  return args

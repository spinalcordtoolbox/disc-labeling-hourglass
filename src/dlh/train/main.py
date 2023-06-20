#=====================================================================================================
## code from https://github.com/bearpaw/pytorch-pose 
# Revised by Reza Azad (rezazad68@gmail.com)
# Revised by Nathan Molinier (nathan.molinier@gmail.com)
# 🐝 Wandb edit based on https://github.com/ivadomed/model_seg_mouse-sc_wm-gm_t1/blob/main/train.py
#=====================================================================================================

from __future__ import print_function, absolute_import
import os
import argparse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from progress.bar import Bar
from torch.utils.data import DataLoader 
import copy
import wandb
import json

from dlh.models.hourglass import hg
from dlh.models.atthourglass import atthg
from dlh.models import JointsMSELoss
from dlh.models.utils import AverageMeter, adjust_learning_rate, accuracy, dice_loss
from dlh.utils.train_utils import image_Dataset, SaveOutput, save_epoch_res_as_image2, save_attention, loss_per_subject
from dlh.utils.test_utils import CONTRAST, load_niftii_split
from dlh.utils.skeleton import create_skeleton
from dlh.utils.config2parser import parser2config, config2parser

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  


def main(args):
    '''
    Train hourglass network
    '''
    best_acc = 0
    weight_folder = args.weight_folder
    vis_folder = args.visual_folder
    contrasts = CONTRAST[args.contrasts]
    datapath = args.datapath
    wandb_mode = args.wandb
    label_suffix = args.suffix_label
    img_suffix = args.suffix_img
    
    # Create weights folder to store training weights
    if not os.path.exists(weight_folder):
        os.mkdir(weight_folder)
        
    # Create visualize folder to images created during training
    if not os.path.exists(vis_folder):
        os.mkdir(vis_folder)
    
    # Loading images for training and validation
    print('loading images...')
    imgs_train, masks_train, discs_labels_train, subjects_train, _ = load_niftii_split(datapath=datapath, 
                                                                                    contrasts=contrasts, 
                                                                                    split='train', 
                                                                                    split_ratio=args.split_ratio,
                                                                                    label_suffix=label_suffix,
                                                                                    img_suffix=img_suffix)
    
    imgs_val, masks_val, discs_labels_val, subjects_val, _ = load_niftii_split(datapath=datapath, 
                                                                            contrasts=contrasts, 
                                                                            split='val', 
                                                                            split_ratio=args.split_ratio,
                                                                            label_suffix=label_suffix,
                                                                            img_suffix=img_suffix)
    
    ## Create a dataset loader
    full_dataset_train = image_Dataset(images=imgs_train, 
                                       targets=masks_train,
                                       discs_labels_list=discs_labels_train,
                                       subjects_names=subjects_train,
                                       num_channel=args.ndiscs,
                                       use_flip = True,
                                       load_mode='train'
                                       )

    full_dataset_val = image_Dataset(images=imgs_val, 
                                    targets=masks_val,
                                    discs_labels_list=discs_labels_val,
                                    subjects_names=subjects_val,
                                    num_channel=args.ndiscs,
                                    use_flip = False,
                                    load_mode='val'
                                    )

    MRI_train_loader = DataLoader(full_dataset_train, 
                                batch_size=args.train_batch,
                                shuffle=True,
                                num_workers=0
                                )
    MRI_val_loader = DataLoader(full_dataset_val, 
                                batch_size=args.val_batch,
                                shuffle=False,
                                num_workers=0
                                )
    
    # idx is the index of joints used to compute accuracy (we detect N discs starting from C1 to args.ndiscs) 
    idx = [(i+1) for i in range(args.ndiscs)]

    # create model
    print("==> creating model stacked hourglass, stacks={}, blocks={}".format(args.stacks, args.blocks))
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.ndiscs)
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.ndiscs)
    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss().to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(
                                        model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay
                                        )
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr=args.lr,
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False
    
    # optionally resume from a checkpoint
    if args.resume:
       print("=> loading checkpoint to continue learing process")
       if args.att:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrasts}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])
       else:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrasts}_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        print('loading the pretrained weight')
        if args.att:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrasts}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])
        else:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrasts}_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])

        if args.attshow:
            loss, acc = show_attention(MRI_val_loader, model)
        else:
            loss, acc = validate(MRI_val_loader, model, criterion, epoch, idx, vis_folder)
        return
    
    if wandb_mode:
        # 🐝 initialize wandb run
        wandb.init(project='hourglass-network',config=vars(args))
    
        # 🐝 log gradients of the models to wandb
        wandb.watch(model, log_freq=100)
        
        # 🐝 add training script as an artifact
        artifact_script = wandb.Artifact(name='script', type='file')
        artifact_script.add_file(local_path=os.path.abspath(__file__), name=os.path.basename(__file__))
        wandb.log_artifact(artifact_script)
    
    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            MRI_train_loader.dataset.sigma *=  args.sigma_decay
            MRI_val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        epoch_loss, epoch_acc = train(MRI_train_loader, model, criterion, optimizer, epoch, idx, wandb_mode)

        if wandb_mode:
            wandb.log({"training_loss/epoch": epoch_loss})
            
            # 🐝 log train_loss over the epoch to wandb
            wandb.log({"training_loss/epoch": epoch_loss})
            
            # 🐝 log training learning rate over the epoch to wandb
            wandb.log({"training_lr/epoch": lr})
        
        # evaluate on validation set
        valid_loss, valid_acc, valid_dice = validate(MRI_val_loader, model, criterion, epoch, idx, vis_folder, wandb_mode)

        if wandb_mode:
            # 🐝 log valid_dice over the epoch to wandb
            wandb.log({"validation_dice/epoch": valid_dice})
        
        # remember best acc and save checkpoint
        if valid_acc > best_acc:
           state = copy.deepcopy({'model_weights': model.state_dict()})
           if args.att:
                torch.save(state, f'{weight_folder}/model_{args.contrasts}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}')
           else:
                torch.save(state, f'{weight_folder}/model_{args.contrasts}_stacks_{args.stacks}_ndiscs_{args.ndiscs}')
           best_acc = valid_acc
           best_acc_epoch = epoch + 1
    
    if wandb_mode:
        # 🐝 log best score and epoch number to wandb
        wandb.log({"best_accuracy": best_acc, "best_accuracy_epoch": best_acc_epoch})
    
        # 🐝 version your model
        best_model_path = f'{weight_folder}/model_{args.contrasts}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}'
        model_artifact = wandb.Artifact("hourglass", 
                                        type="model",
                                        description="Hourglass network for intervertebral discs labeling",
                                        metadata=vars(args)
                                        )
        model_artifact.add_file(best_model_path)
        wandb.log_artifact(model_artifact)
        
        # 🐝 close wandb run
        wandb.finish()

    
                

def train(train_loader, model, criterion, optimizer, ep, idx, wandb_mode):
    '''
    Train hourglass for one epoch
    
    :param train_loader: loaded training dataset
    :param model: loaded model
    :param criterion: loaded loss function
    :param optimizer: loaded solver
    :param idx: list of detected class
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    loss_dices = AverageMeter()
    
    # switch to train mode
    model.train()

    end = time.time()

    
    bar = Bar('Train', max=len(train_loader))
    
    # init subjects_loss to store individual loss for each subject in the training
    subjects_loss_dict = {} # subjects_loss_dict = {subject : subject_loss}
    for i, (inputs, targets, vis, subjects) in enumerate(train_loader):
        subjects = list(subjects)
        # measure data loading time
        data_time.update(time.time() - end)
        inputs, targets = inputs.to(device), targets.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)
        
        # compute output and calculate loss
        output = model(inputs) 
        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o, targets, vis)
            output = output[-1]
        else:  # single output
            loss = criterion(output, targets, vis)
        
        # Extract individual loss for each subject    
        sub_loss = loss_per_subject(pred=output, target=targets, vis=vis, criterion=criterion)
        
        if type(subjects) == list:
            for i, subject in enumerate(subjects):
                subjects_loss_dict[subject] = sub_loss[i] # add subjects name and individual loss to dict
        else:
            subjects_loss_dict[subjects] = sub_loss # add subjects name and individual loss to dict
        
        if wandb_mode:
            # 🐝 log train_loss for each step to wandb
            wandb.log({"training_loss/step": loss.item()})

        # measure accuracy and record loss
        acc = accuracy(output, targets, idx)
        losses.update(loss.item(), inputs.size(0))
        acces.update(acc[0], inputs.size(0))
        
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=(i+1),
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg*100,
                    acc=acces.avg
                    )
        bar.next()
    bar.finish()
    
    if wandb_mode:
        # 🐝 log bar plot with individual loss in wandb
        wandb.log(subjects_loss_dict)
    
    return losses.avg, acces.avg


def validate(val_loader, model, criterion, ep, idx, out_folder, wandb_mode):
    '''
    Compute validation dataset with hourglass for one epoch
    
    :param val_loader: loaded validation dataset
    :param model: loaded model
    :param criterion: loaded loss function
    :param ep: current epoch number
    :param idx: list of detected class
    :param out_folder: path out for generated visuals
    '''
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    loss_dices = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, target, vis) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            # compute output
            output = model(input)
            output = output[-1]
        
            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, vis)
                output = output[-1]
            else:  # single output
                loss = criterion(output, target, vis)
            acc = accuracy(output.cpu(), target.cpu(), idx)
            loss_dice = dice_loss(output, target)
            
            if wandb_mode:
                # 🐝 log validation_loss for each step to wandb
                wandb.log({"validation_dice/step": loss_dice})

            if i == 0:
                txt, res, targets, preds = save_epoch_res_as_image2(input, output, target, out_folder, epoch_num=ep, target_th=0.5, wandb_mode=wandb_mode)
                
                if wandb_mode:
                    # 🐝 log visuals for the first validation batch only in wandb
                    wandb.log({"validation_img/batch_1": wandb.Image(res, caption=txt)})
                    wandb.log({"validation_img/groud_truth": wandb.Image(targets, caption=f'ground_truth_{ep}')})
                    wandb.log({"validation_img/prediction": wandb.Image(preds, caption=f'prediction_{ep}')})
                
            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))
            loss_dices.update(loss_dice.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}| dice: {dice:.4f}'.format(
                        batch=(i+1),
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg*100,
                        acc=acces.avg,
                        dice=loss_dices.avg*100
                        )
            bar.next()

        bar.finish()
    return losses.avg, acces.avg, loss_dices.avg



def show_attention(val_loader, model):
    ## define the attention layer output
    save_output = SaveOutput()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            if layer.weight.size()[0]==1:     
                layer.register_forward_hook(save_output)
                break
    # switch to evaluate mode
    N = 1
    Sel= 0
    model.eval()
    with torch.no_grad():
        for i, (input, target, vis) in enumerate(val_loader):
            if i==Sel:
               input  = input [N:N+1]
               target = target[N:N+1]
               input  = input.to(device, non_blocking=True)
               target = target.to(device, non_blocking=True)
               output = model(input)
               att = save_output.outputs[0][0,0]
               output = output[-1]
               save_attention(input, output, target, att, target_th=0.6)
            
    return 0, 0

def create_json(args):
    '''
    This script creates a json file where training parameters will be stored
    '''
    # Data to be written
    data = {
        "datapath": args.datapath,
        "contrasts": args.contrasts,
        "suffix_img": args.suffix_img,
        "suffix_label": args.suffix_label,
        "wandb": args.wandb,
        "split_ratio": args.split_ratio,
        "resume": args.resume,
        "attshow": args.attshow,
        "epochs": args.epochs,
        "train_batch": args.train_batch,
        "val_batch": args.val_batch,
        "solver": args.solver,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "sigma_decay": args.sigma_decay,
        "schedule": args.schedule,
        "gamma": args.gamma,
        "evaluate": args.evaluate,
        "start_epoch": args.start_epoch,
        "weight_folder": os.path.abspath(args.weight_folder),
        "skeleton_folder": os.path.abspath(args.skeleton_folder),
        "visual-folder": os.path.abspath(args.visual_folder),
        "ndiscs": args.ndiscs, # model parameters
        "att": args.att,
        "stacks": args.stacks,
        "features": args.features,
        "blocks": args.blocks 
    }
    
    # Serializing json
    json_object = json.dumps(data, indent=4)
    
    # Writing to sample.json
    json_path = os.path.join(os.path.abspath(args.weight_folder),f'config_{os.path.basename(args.datapath)}_{args.contrasts}.json')
    with open(json_path, "w") as outfile:
        print(f"The config file {json_path} with all the training parameters was created")
        outfile.write(json_object)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training hourglass network')
    
    ## Config file mode
    parser.add_argument('--config', type=str, default='',
                        help='Path to the training config file Example: ~/<your_path>/config.json')
    
    ## Parameters (no config file mode)
    # General parameters
    parser.add_argument('--datapath', type=str,
                        help='Path to data folder generated using data_management/gather_data.py Example: ~/<your_dataset>/vertebral_data (Required if no config file)')
    parser.add_argument('-c', '--contrasts', type=str, metavar='N',
                        help='MRI contrasts: choices=["t1", "t2", "t1_t2"] (Required if no config file)')               
    parser.add_argument('--suffix-label', type=str, default='_labels-disc-manual',
                        help='Specify label suffix (default= "_labels-disc-manual")') 
    parser.add_argument('--suffix-img', type=str, default='',
                        help='Specify img suffix (default= "")')
    
    # Model parameters
    parser.add_argument('--ndiscs', type=int, default=15,
                        help='Number of discs to detect (default=15)')
    parser.add_argument('--wandb', default=True,
                        help='Train with wandb (default=True)')
    parser.add_argument('--split-ratio', default=(0.8, 0.1, 0.1),
                        help='Split ratio used for (train, val, test) (default=(0.8, 0.1, 0.1))')
    parser.add_argument('--resume', default=False, type=bool,
                        help='Resume the training from the last checkpoint (default=False)')  
    parser.add_argument('--attshow', default=False, type=bool,
                        help=' Show the attention map (default=False)') 
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run (default=120)')
    parser.add_argument('--train-batch', default=3, type=int, metavar='N', 
                        help='train batchsize (default=3)')
    parser.add_argument('--val-batch', default=4, type=int, metavar='N',
                        help='validation batchsize (default=4)')
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers: choices=["rms", "adam"] (default="rms")')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate (default=2.5e-4)')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum (default=0)')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default=0)')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch. (default=0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs. (default=[60, 90])')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule. (default=0.1)')
    parser.add_argument('-e', '--evaluate', default=False, type=bool,
                        help='Evaluate model on validation set (default=False)')
    parser.add_argument('--att', default=True, type=bool, 
                        help='Use attention or not (default=True)')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack (default=2)')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass (default=256)')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass (default=1)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts) (default=0)')
    parser.add_argument('--weight-folder', type=str, default='src/dlh/weights',
                        help='Folder where hourglass weights are stored and loaded. Will be created if does not exist. (default="src/dlh/weights")')
    parser.add_argument('--visual-folder', type=str, default='visualize',
                        help='Folder where visuals are stored. Will be created if does not exist. (default="visualize")')
    parser.add_argument('--skeleton-folder', type=str, default='src/dlh/skeletons',
                        help='Folder where skeletons are stored. Will be created if does not exist. (default="src/dlh/skeletons")')
    
    if parser.parse_args().config == '':
        # No config file mode
        args = parser.parse_args()
        json_name = f'config_{os.path.basename(args.datapath)}_{args.contrasts}.json'
        parser2config(args, path_out=os.path.join(parser.parse_args().weight_folder, json_name))  # Create json file with training parameters
        
    else:
        # Config file mode
        args = config2parser(parser.parse_args().config)
    
    main(args)  # Train the hourglass network
    create_skeleton(args)  # Create skeleton file to improve hourglass accuracy during testing
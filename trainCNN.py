#for split in {0..20}; do python trainCNN.py --filename=dataset/dataset_cat=custom_ASlag=45_MS=6_c=5_sigIt=8_days=2_minNStat=3.npy  --alpha_min=0.3 --ysmoothing=0 --twoDim=1 --dropout=0.0 --epochs=100 --batch_size=10 --augmentation=1 --split=$split; done

#for /l %s in (0,1,100) do python trainCNN.py --outfolder=local/ --model_name=2 --filename=dataset/dataset_cat=custom_ASlag=45_MS=6_c=5_sigIt=8_days=2_minNStat=3_thinElasticSheet=True.npy --alpha_min=0.1 --twoDim=1 --dropout=0.0 --epochs=300 --batch_size=5 --augmentation=1 --split_size=10 --split=%s --patience=20 --lr=0.001 --bootstrap_splits=1 --weight_decay=0.0001 
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_curve, auc
import argparse
from train_utils import GpsPolarDataset, format_trainingoutput_filename, enlarge
import matplotlib.pyplot as plt
import scipy.ndimage
import os
import train_utils

import time


def run_model(model, batch, loss_fn, class_weight, dvc, use_mask):
    x_batch, y_batch      = batch[0].to(dvc), batch[1].to(dvc)
    id_batch, alpha_batch = batch[2].to(dvc), batch[3].to(dvc)
    # propagate + loss
    model_out = model(x_batch, alpha_batch, use_mask)
    loss = loss_fn(model_out, y_batch)
    if(not(class_weight is None)):
        loss[y_batch==0] *= class_weight[0]
        loss[y_batch==1] *= class_weight[1]

    if(use_mask):
        return loss[alpha_batch>0].mean(), model_out
    else:
        return (alpha_batch*loss).mean(), model_out

parser = argparse.ArgumentParser()
## input/output management
parser.add_argument('--filename', type=str, help= "Should be something like a dataset (trainset)")
parser.add_argument('--outfolder', type=str, default='train_results/')
parser.add_argument('--year_test', type=str, default='2015' , help= "Train/Test split is based on years. This is the year to start the test set. Before is train+val, after is test")
parser.add_argument('--decorator', type=str, default="", help="String to insert at the beginning of ALL output filenames.")
parser.add_argument('--split', type=int, default=0, help="Number to enumerate the split / seed for the bootstrap-splitting")
parser.add_argument('--split_size', type=int, default=10, help="Size of the split: how many events in each split (size of validation set)")
parser.add_argument('--bootstrap_splits', type=int, default=1, help="Boolean to activate boostrap-style validation (draw train set at random, BUT withOUT replacement)")

## TODO: implement this ? Not so easy because components are mixed in polars or carts
# parser.add_argument('--ndays', type=int, default=0, help="number of days to keep (past)")

## pre-processing/data-structure related:
parser.add_argument('--alpha_min', type=float, default=0.1,\
                    help="Confidence level (in [0,1]) demanded to consider using a point for train/test. Higer means less data (more confidence).")
parser.add_argument('--alpha_max_dist',type=float, default=None,\
                    help="Confidence level demanded, expressed as a distance from closest stations, to be provided in units of sigma (sigma_interpolation). If provided, overloads the alpha_min value.")
parser.add_argument('--ysmoothing', type=float, default=0.0, \
                    help="If >0.01, apply Gaussian smoothing to the labels, with a radius of smoothing sigma=ysmoothing")
parser.add_argument('--enlarge', type=int, default=0, \
                    help="If 'enlarge' > 0 it enlarges the '1's in the ground truth by 'enlarge' pixels")
parser.add_argument('--regression', type=int, default=0, \
                    help="Kind of task: regression (predicting total energy release at a pixel) or classification (predicting whether an AS occured at that pixel).")
parser.add_argument('--twoDim', type=int, default=1, \
                    help="Discard the upward (True) component or not (False)") ##
parser.add_argument('--cartesian', type=int, default=0, \
                    help="False: use Polar (better), True: use cartesian (probably worse)")
# number of days: we could also choose to cut or somehow pre-process the temporal aspect of data here.

## limit tests
parser.add_argument('--stdRadial', type=int, default=0, \
                    help="True: standardize the radial channels (each event gets divided by its local std dev between its pixels)")
parser.add_argument('--killRadial', type=int, default=0, \
                    help="True: divides the radial channels by 100000")
parser.add_argument('--killAngular', type=int, default=0, \
                    help="True: divides the angular channels by 100000")
parser.add_argument('--distToCenterBaseline', type=int, default=0, help="Goes sigmoid mode and uses the dist to center to predict, and only that")
parser.add_argument('--radialNormBaseline', type=int, default=0, help="Goes sigmoid mode and uses the radial norm only to predict, and only that")
parser.add_argument('--strain', type=int, default=0, help="strain input")

parser.add_argument('--exclusionList', type=int, default=0, help="values: 0,1,2: excludes a hard-coded list of examples from the whole process (2), or keeps theses events in the test set (1).")


## model/architecture choices
## TODO: add layer sizes, etc. At least in the filename output,
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--model_name', type=int, default=2)

## training dynamics related:
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=0, help="LR scheduler step size (number of epochs before we divide LR by 2)")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--masked', type=int, default=0, help="whether to use hard masking. It's advised to set it to 0 (use soft masking)")

## always helps:
parser.add_argument('--sumDays', type=int, default=1)
parser.add_argument('--augmentation', type=int, default=0, help="Whether to use or not dataset augmentation.")
use_class_weight=True


args = parser.parse_args()

args.regression = bool(args.regression) # no regression atm
regression = False # no regression atm
args.twoDim = bool(args.twoDim)
args.cartesian = bool(args.cartesian)
args.augmentation = bool(args.augmentation)
args.augmentation = 0
print("data augmentation seems broken, don't use it.  args.augmentation = 0")
args.bootstrap_splits = bool(args.bootstrap_splits)
# args.model_type = str(args.model_type)
args.masked = bool(args.masked)

# parameters/settings for training 
# whether to use 3 spatial comps (N,E,U) or just (N,E)
# this is MANDATORY when using 'elastic sheet 2d' type of fit
usePolar = (args.cartesian == 0)
split = args.split
split_size = args.split_size

stdRadial  = args.stdRadial
killRadial = args.killRadial
killAngular= args.killAngular
if args.radialNormBaseline:
    args.sumDays=True
sumDays    = args.sumDays

alpha_min = args.alpha_min
alpha_max_dist = args.alpha_max_dist
if alpha_max_dist != None:
    # alpha_min = 0.3    ## setting by distance is preferred
    # alpha_max_dist = 1.5 ## in units of sigma
    alpha_min = np.exp(-0.5*alpha_max_dist**2) ## the minimal alpha for a pixel to be considered valid
alpha_max_dist = np.sqrt(-2*np.log(alpha_min))
print("maximal distance to a station allowed for a pixel (in units of sigma):", alpha_max_dist)    # " or in km:", alpha_max_dist*sigma_interpolation*cell_size_km)

###########################################################################################################

if(not os.path.exists(args.outfolder)):
    os.mkdir(args.outfolder)

print("args: ", args)
rootname = format_trainingoutput_filename(args)

# data loading
data_dict = np.load(args.filename, allow_pickle=True).item()
carts = data_dict['cartesian']
polars = data_dict['polar']
labels = data_dict['label']
rawlabels = labels.copy()
ids = data_dict['id']
days = data_dict['day']
comp_dict = data_dict['components']
rawalphas = data_dict['alpha'] ## continuous values


# torch datasets:
# seed manually torch and numpy to make training deterministic
seed=split
# seed=int(time.time())
    ## TODO: use random seed -- it's nice to check wheter we learned:
    ## if we forget the initial condition, i.e reach the same final perf (loss, val auc, etc) from different initializations, it means learning went ok
    ## and so we don't need to do many more epochs (at least as a first guess, then fine-tuning with more epochs can help, or help to overfit)
torch.manual_seed(seed)
np.random.seed(seed)


###################################################################################
## masking strategy: when too far from station, decrease imprtance or kill pixel ##

if(args.masked): ## hard masking -> creates artifacts at the boundarie sof the mask
    maskalphas = np.array(rawalphas>alpha_min, dtype=bool) # we make the [0,1] mask a {0,1} mask
    alphas = maskalphas  ## boolean
else:  ## soft masking: the loss is multiplied with alpha to decrease relative importance of far-away points
    ## we increase the radius of alphas' only when we use soft alpha ! For hard masking, we keep the curent radius
    magnification_factor = 1.5
    rawalphas = np.exp(np.log(rawalphas)/magnification_factor**2)
    alphas = rawalphas   ## real-valued

#################################################################
## pre-processing of the labels (AS/no-AS state of each pixel) ##

if args.enlarge > 0:
    print("WARNING ! - args.enlarge>0 is deprecated. Use ysmoothing instead (fast, no need bor numba, and mostly, it is isotropic).")
    labels = enlarge(rawlabels, args.enlarge)

if args.ysmoothing > 0.01:
    smoothedLabels = rawlabels.copy()
    ## smoothing with Gaussian filter. Note it will make the 1's surrounded with 0's into a smaller value (we then boost it back)
    sigma_smoothing = args.ysmoothing
    truncate = 2*sigma_smoothing
    for nevent in range(labels.shape[0]):
        smoothedLabels[nevent] = scipy.ndimage.gaussian_filter(labels[nevent], sigma=sigma_smoothing, truncate=truncate)
    ## we find the smoothed value at pixels with a single AS (no neighboring AS) so as to boost values
    ## in terms of learning, it amount to a scaling of the learning rate
    ## in terms of plots, it's nice to have y in the range [0,1+] (occasionally larger than 1)
    boostfactor = np.sort(smoothedLabels.max(axis=1).max(axis=1))
    boostfactor = boostfactor[boostfactor>0][0]
    smoothedLabels/=boostfactor
    smoothedLabels = np.minimum(smoothedLabels,1.0)
    labels = smoothedLabels.copy()
    labels = np.array(labels>0.1, dtype=int) ## TODO: make sure this simplification is useless ? But probably it helps clearning out things when computing performance
    # del smoothedLabels


#######################################################################
## sum input days (0 and -1 typically, but works also for more days) ##
## this is a coarse-graining of time (for inputs), very useful       ##
if sumDays:
    print("we sum the input days to get an effective single-day signal")
    if args.twoDim:
        ## first we sum the cartesian-coords data over all input days:
        carts_resum = np.zeros((carts.shape[0], 2, carts.shape[2], carts.shape[3]))
        for i, direction in enumerate(["north", "east"]):
            for comp in comp_dict[direction]:
                print(comp)
                carts_resum[:,i] += carts[:,comp]
        ## we have now an input of "effective one day" shape (we resummed all days into 1)

        polars = np.zeros((carts.shape[0], 3, carts.shape[2], carts.shape[3]))
        ## radial component:
        polars[:,0] = (carts_resum[:,0]**2+carts_resum[:,1]**2)**0.5
        comp_dict["radial"] = np.array([0])

        ## angular components:
        polars[:,1] = carts_resum[:,0]/polars[:,0]
        polars[:,2] = carts_resum[:,1]/polars[:,0]
        comp_dict["angular"] = np.array([1,2])


        if args.strain :
            strain_y = np.gradient(carts_resum[:,0], axis=1)[:,None,:,:]
            strain_x = np.gradient(carts_resum[:,1], axis=2)[:,None,:,:]
            strain_xy = (np.gradient(carts_resum[:,0], axis=2)+np.gradient(carts_resum[:,1], axis=1))[:,None,:,:]
            strain = np.concatenate([strain_y, strain_x, strain_xy], axis=1)
            polars = strain
    else:
        print("sumDays on 2 components not coded yet. It's not urgent and this way the code remains clear, explicit is better than implicit")



###################################################################################
## compute dist to center
n_rows, n_cols = labels.shape[1], labels.shape[2]
ii, jj = np.meshgrid(np.arange(0, n_rows), np.arange(0, n_cols), indexing='ij')
ii -= n_rows//2
jj -= n_cols//2
positions = np.vstack([ii.flatten(), jj.flatten()]).T
dist_to_center = np.sum(positions**2,1).reshape(n_cols,n_rows)
dist_to_center = np.array(dist_to_center**0.5, dtype= float)/ 10.0

dist_to_center = np.repeat(dist_to_center[None, :], repeats=labels.shape[0], axis=0)[:,None,:,:]
# input_dataset = np.concatenate((input_dataset, dist_to_center), axis=1)

###################################################################################
## hacks to make some typical scenario consistent, in terms of options choices.  ##

if args.model_name <=2:
    model_type="2col"

if args.killRadial or args.killAngular or args.strain : ## overrides previous choices
    model_type = "1col"
    args.model_name=3  ##single-column architecture

if args.distToCenterBaseline:           ## overrides previous choices
    ## stupid 0-param model
    args.model_name=4 # sigmoid mode
    model_type = "sigmoid"
    input_dataset =  - dist_to_center
    # input_dataset = np.concatenate((input_dataset, dist_to_center), axis=1)
if args.radialNormBaseline:
    ## stupid 0-param model
    args.model_name=4 # sigmoid mode
    model_type = "sigmoid"
    input_dataset =  polars[:,comp_dict["radial"]]

# run trainCNN.py --filename dataset/dataset_cat\=custom_ASlag\=45_MS\=6_c\=5_sigIt\=8_days\=2_mS\=3_tE\=1.npy   --twoDim 1 --bootstrap_splits 1 --split_size 10  --weight_decay 1e-5  --alpha_min 0.1   --b
#     ...: atch_size 4  --masked 0  --ysmoothing 0 --stdRadial 0 --killRadial 0 --sumDays 1 --distToCenter 1 --distToCenterBaseline 1 --epochs 50 --split 12 --lr 1e-1



###################################################################################

if model_type == "1col":
    ## we mess with the input to see how much each channel is crucial (/leads to silly overfitting)
    if args.killRadial:
        column_feats_name = 'angular'
        comp_dict['radial'] = np.array([])
    if args.killAngular:
        column_feats_name = 'radial'
        comp_dict['angular'] = np.array([])
    if args.strain :
        column_feats_name = 'STRAIN'
        column_feats = np.array([0,1,2])
    else:
        column_feats = comp_dict[column_feats_name]

    print("Using", column_feats, "for first (and ONLY) column.")
    c1_in = len(column_feats)

    input_dataset= polars[:, column_feats]


elif model_type == "2col":
    if(usePolar):
        input_dataset = polars
    else:
        input_dataset = carts

    # select proper components for the neural net
    if(args.twoDim):  ## 2D-constrained fit, using a thin elastic sheet model
        if(usePolar):
            column1_feats_name = 'radial'
            column2_feats_name = 'angular'
        else:
            column1_feats_name = 'north'
            column2_feats_name = 'east'
    else: ## simple gaussian (order 0) interpolation
        if(usePolar):
            column1_feats_name = 'radial3d'
            column2_feats_name = 'angular3d'
        else:
            column1_feats_name = 'vector'
            column2_feats_name = 'scalar'
    column1_feats = comp_dict[column1_feats_name]
    column2_feats = comp_dict[column2_feats_name]
    print("Using", column1_feats, "for first column.")
    print("Using", column2_feats, "for second column.")
    c1_in = len(column1_feats)
    c2_in = len(column2_feats)

elif model_type == "sigmoid":
    assert(input_dataset.shape[1]==1)
    ## actually, the input 1x1 conv layer could handle more than 1 input channel, but it's not the spirit of this test.

###################################################################################

if stdRadial: ## standardize the radial channel(s),event by event, to avoid over-exposed "pictures"
    assert(args.killRadial==False)
    print("standardizing radial features, which are, I believe, components number:", comp_dict['radial'])
    for c in comp_dict['radial']:
        input_dataset[:,c] /= input_dataset[:,c].std(axis=(1,2))[:,None,None]
    # ## we mess with the input to see how much each channel is crucial (/leads to silly overfitting)
    # for c in comp_dict[column1_feats_name]:
    #     if stdRadial:
    #         input_dataset[:,c] /= input_dataset[:,c].std(axis=(1,2))[:,None,None]
    #     if killRadial:
    #         input_dataset[:,c] /= 100000.0
    # for c in comp_dict[column2_feats_name]:
    #     if killAngular:
    #         input_dataset[:,c] /= 100000.0

###################################################################################
## test split, train - val split(s), dat augmentation (broken), data loaders

# train (+val)/test split -> done by date
if len(args.year_test)==4:
    args.year_test =int(args.year_test)
    trainval_idx = days < np.datetime64('%4i-01-01' % args.year_test) ## train+validation : before 2015.
else:
    pass
    print("test event are only events after the year:" )
    trainval_idx = days < np.datetime64(args.year_test) ## train+validation : before 2015.

exclude_also_from_test=False
if abs(args.exclusionList)>1:
    exclude_also_from_test=True
if args.exclusionList>0:
    pass
    ## here instead we specify which events to INCLUDE in the dataset (typically a bit more numerous than the exclusion list above)
    # relative_ids =[22,23,27,29,30, 28, 31, 32, 50,54, 62]
    # ids[relative_ids] == array([341, 344, 409, 426, 430, 422, 437, 438, 520, 534, 556])
    absolute_ids = np.array([341, 344, 409, 426, 430, 422, 437, 438, 520, 534, 556])
    excluded_indices = []
    for i, abs_id in enumerate(ids):
        if abs_id in absolute_ids:
            excluded_indices.append(i)
    excluded_indices = np.array(excluded_indices)
    trainval_idx[excluded_indices] = False

if args.exclusionList<0: ## here we keep only the big-norm ones (max larger than 19)
    # relative_ids = np.array([ 0,  1,  2,  3,  4,  5,  6,  9, 10, 12, 13, 14, 15, 16, 17, 18, 19,       20, 21, 29, 30, 31, 32, 33, 34, 35, 36, 37, 40, 41, 42, 43, 44, 45,       49, 51, 52, 56, 57, 58, 59, 60, 61, 63, 64, 65, 66, 67, 69, 70])
    # ids[relative_ids]
    absolute_ids = np.array([159, 161, 202, 204, 206, 214, 220, 262, 273, 277, 282, 291, 294,       297, 300, 308, 311, 317, 319, 426, 430, 437, 438, 440, 446, 452,       454, 456, 471, 472, 473, 480, 481, 484, 514, 521, 525, 541, 546,       548, 551, 552, 553, 562, 563, 567, 569, 570, 576, 577])
    excluded_indices = []
    ## nice
    for i, abs_id in enumerate(ids):
        if abs_id in absolute_ids:
            excluded_indices.append(i)
    excluded_indices = np.array(excluded_indices)
    trainval_idx[excluded_indices] = False

# if args.exclusionList<0:
#     ## here instead we specify which events to INCLUDE in the dataset (typically a bit more numerous than the exclusion list above)
#     # relative_ids =[22,23,27,29,30, 28, 31, 32, 50,54, 62]
#     # ids[relative_ids] == array([341, 344, 409, 426, 430, 422, 437, 438, 520, 534, 556])
#     # absolute_ids = np.array([341, 344, 409, 426, 430, 422, 437, 438, 520, 534, 556])

#     included_indices = []
#     for i, abs_id in enumerate(ids):
#         if abs_id in absolute_ids:
#             included_indices.append(i)
#     included_indices = np.array(included_indices)
#     trainval_idx = included_indices.copy()
#     # trainval_idx[excluded_indices] = False

test_idx = 1-trainval_idx
if exclude_also_from_test :
    test_idx[excluded_indices] = False
# if args.exclusionList<1:
#     TOOD: construire le test comme il faut.

test_idx    = np.argwhere( test_idx    ).flatten()
trainval_idx = np.argwhere(trainval_idx).flatten()  # train+val

assert(len(test_idx)>0)
if(len(test_idx)==0):
    print("Test size is 0. Closing.")
    exit()


####
if args.bootstrap_splits == False:    ## Leave-One-out approach: (or, leave a few, split_size=1 or split_size="a few")
    if split_size*(split+1) >len(trainval_idx):
        print("split number larger than possible splits. Leaving.")
        exit()
    split_train_idx = np.concatenate([trainval_idx[:split_size*split], trainval_idx[split_size*(split+1):]])
    split_val_idx   = trainval_idx[split_size*split:split_size*(split+1)]
    print("Number of samples in train set (without augmentation):", len(split_train_idx))
    print("Number of samples in test set:", len(test_idx))

else:   ## take split_size events for validation, and the rest for train (draw at random and repeat experiment many times)
    Ntrain = len(trainval_idx) - split_size
    assert(Ntrain > 0)
    rng = np.random.default_rng(seed=split)  ## we want to have reproducible splits
    split_train_idx = rng.choice(trainval_idx, Ntrain, replace=False )
    split_val_idx = []
    for e in trainval_idx:
        if e not in split_train_idx:
            split_val_idx.append(e)
    split_val_idx = np.array(split_val_idx)


if(args.augmentation and usePolar):
    ## none(=rot(0)) + 3 rots (90,180,270),
    ## horiz flip    + 3 (rots + horiz flip)
    ## = 8 total -> ok
    transformations =  [(0,), (1,), (2,), (3,),
                        (4,), (1, 4), (2, 4), (3,4)]
    train_dataset = GpsPolarDataset(input_dataset[split_train_idx],
                                    labels[split_train_idx],
                                    ids[split_train_idx],
                                    alphas[split_train_idx],
                                    comp_dict,
                                    transformations)
else:
    train_dataset = torch.utils.data.TensorDataset(torch.tensor(input_dataset[split_train_idx], dtype=torch.float32),
                                torch.tensor(labels[split_train_idx], dtype=torch.float32),
                                torch.tensor(ids[split_train_idx], dtype=torch.long),
                                torch.tensor(alphas[split_train_idx], dtype=torch.float32)
                                )
validation_dataset = torch.utils.data.TensorDataset(torch.tensor(input_dataset[split_val_idx], dtype=torch.float32),
                            torch.tensor(labels[split_val_idx], dtype=torch.float32),
                            torch.tensor(ids[split_val_idx], dtype=torch.long),
                            torch.tensor(alphas[split_val_idx], dtype=torch.float32)
                            )
test_dataset = torch.utils.data.TensorDataset(torch.tensor(input_dataset[test_idx], dtype=torch.float32),
                            torch.tensor(labels[test_idx], dtype=torch.float32),
                            torch.tensor(ids[test_idx], dtype=torch.long),
                            torch.tensor(alphas[test_idx], dtype=torch.float32)
                            )

# we only shuffle in train_loader
train_loader      = torch.utils.data.DataLoader(train_dataset     , batch_size=args.batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=len(split_val_idx), shuffle=False)
test_loader       = torch.utils.data.DataLoader(test_dataset      , batch_size=len(test_idx), shuffle=False)

# cuda/cpu training
if(torch.cuda.is_available()):
    dvc = torch.device('cuda')
    print('Using CUDA')
else:
    dvc = torch.device('cpu')
    print('Using CPU')

###################################################################################

model_name = ["standard", "super_simple_model", "3x3kernels_model", "singleColumn",  "sigmoid"][args.model_name]

## standard (Vicio's) model:
if model_name=="standard":
    c1_mod_channels = [c1_in,1]
    c1_mod_kernels  = [1]
    c2_mod_channels = [c2_in, 10,10,10,10,10,5,5,5,1]
    c2_mod_kernels  = [3,3,3,3,3,3,3,3,3]
    c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],20, 6,1]
    c_mod_kernels   = [1]
    # c1_mod_channels = [c1_in,10,10,10]
    # c1_mod_kernels  = [3,3,3]
    # c2_mod_channels = [c2_in,64,64,64]
    # c2_mod_kernels  = [3,3,3]
    # c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],20, 6,1]
    # c_mod_kernels   = [1]
    # c1_mod_channels = [c1_in,3,3,3]
    # c1_mod_kernels  = [5,5,5]
    # c2_mod_channels = [c2_in,3,3,3]
    # c2_mod_kernels  = [5,5,5]
    # c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],3,3,1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],20, 6,1]
    # c_mod_kernels   = [5,5,1]
    #c1_mod_channels = [c1_in,10,10,10]
    #c1_mod_kernels  = [5,5,5]
    #c2_mod_channels = [c2_in,10,10,10]
    #c2_mod_kernels  = [5,5,5]
    #c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],10,5,1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],20, 6,1]
    #c_mod_kernels   = [5,5,1]
    ## receptive field: 21x21
elif model_name=="super_simple_model":
    c1_mod_channels = [c1_in,1]
    c1_mod_kernels  = [1]
    c2_mod_channels = [c2_in,1]
    c2_mod_kernels  = [1]
    c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],20, 10,1]
    c_mod_kernels   = [1]
elif model_name=="3x3kernels_model":
    #c1_mod_channels = [c1_in,5,5,5,5,5]
    #c1_mod_kernels  = [3,3,3,3,3]
    #c2_mod_channels = [c2_in,5,5,5,5,5]
    #c2_mod_kernels  = [3,3,3,3,3]
    #c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],5,5,1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1] #[c1_mod_channels[-1]+c2_mod_channels[-1],20, 10,1]
    #c_mod_kernels   = [3,3,1]
    c1_mod_channels = [c1_in, 10,10,10,10,10]
    c1_mod_kernels  = [3,3,3,3,3]
    c2_mod_channels = [c2_in, 10,10,10,10,10]
    c2_mod_kernels  = [3,3,3,3,3]
    c_mod_channels  = [c1_mod_channels[-1]+c2_mod_channels[-1],10,10,5,5,1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1]
    c_mod_kernels   = [3,3,3,3,1]
elif model_name== "singleColumn":
    c1_mod_channels = [0]
    c1_mod_kernels  = [0]
    c2_mod_channels = [0]
    c2_mod_kernels  = [0]
    c_mod_channels = [input_dataset.shape[1], 10,10,10,10,10,5,5,5,5,1]
    c_mod_kernels  = [3,3,3,3,3,3,3,3,3,1]
    # c_mod_channels  = [input_dataset.shape[1],5,5,5,5,1] #[c1_mod_channemodel_namels[-1]+c2_mod_channels[-1],1]
    # c_mod_kernels   = [3,3,3,3,1]
    # c_mod_channels  = [input_dataset.shape[1],5,10,15,20,1] #[c1_mod_channels[-1]+c2_mod_channels[-1],1]
    # c_mod_kernels   = [3,3,3,3,1]
elif model_name== "sigmoid":
    c1_mod_channels = [0]
    c1_mod_kernels  = [0]
    c2_mod_channels = [0]
    c2_mod_kernels  = [0]
    assert(input_dataset.shape[1]==1)
    c_mod_channels  = [1]
    c_mod_kernels   = [1]
    # input_dataset
    ## receptive field: 13x13
# elif model_name="baseline_distance":
#     input_dataset

channels = (c1_mod_channels, c2_mod_channels, c_mod_channels)
kernels  = (c1_mod_kernels , c2_mod_kernels,  c_mod_kernels )


totchannels = np.sum(np.array(c1_mod_channels+c2_mod_channels+c_mod_channels))
totkernels = np.sum(np.array(c1_mod_kernels+c2_mod_kernels+c_mod_kernels))
summary=""
summary+="c1_mod_channels: "+str(c1_mod_channels) +"\n"
summary+="c1_mod_kernels: "+str(c1_mod_kernels) +"\n"
summary+="c2_mod_channels: "+str(c2_mod_channels) +"\n"
summary+="c2_mod_kernels: "+str(c2_mod_kernels) +"\n"
summary+="c_mod_channels: "+str(c_mod_channels) +"\n"
summary+="c_mod_kernels: "+str(c_mod_kernels) +"\n"
summary_name = "_arch-summary_nc="+str(totchannels)+"_nk="+str(totkernels)+".txt"
summary_name = rootname + summary_name
with open(summary_name, "a") as flow:
    flow.write(summary)

###################################################################################

# cnn model
if model_type=='1col':
    # elif args.model_type == "1col":
    model = train_utils.SingleColumConv(channels, kernels, dropout=args.dropout)
elif model_type == "sigmoid":
    model = train_utils.sigmoid()
else:
    # if args.model_type == "2col":
    model = train_utils.TwoColumnConv(column1_feats, column2_feats, channels, kernels, dropout=args.dropout, regression=regression)

## we could hack the intialization of some parameters, but it's not super useful
# if model_type=='2col' and args.model_name==0 :
#     model.state_dict()['c1_mod.0.0.weight']*=0
#     model.state_dict()['c1_mod.0.0.weight']-=0.1
###################################################################################

model = model.to(dvc)
# optimizer, we use ADAM
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
# the parameters threshold and factor are freeley modifiable
if(args.patience > 0):
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.patience, \
        gamma=0.5, last_epoch= -1, verbose=True)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau\
    #     (optimizer, 'min', verbose=True, \
    #     patience=args.patience, \
    #     threshold=1e-3, factor=0.5)

# no reduction as we filter our outputs
if(regression):
    loss_fn = nn.MSELoss(reduction='none')
else:
    # loss_fn = nn.BCELoss(reduction='sum')
    loss_fn = nn.BCELoss(reduction='none')

###################################################################################

if(use_class_weight):
    ## class weight are inversely proportional to class fequency
    ## it's estimated from the masked data, restricted to the train+val data
    ## we include validation data in the estimate, so that all splits have the same class_weight
    class_weight = torch.zeros(2,device=dvc)
    ## this code works for continuous alpha AND for binary alpha:
    tot_num_pixels = labels[trainval_idx][alphas[trainval_idx]>alpha_min].size
    tot_num_active_pixels  = (labels[trainval_idx][alphas[trainval_idx]>alpha_min]).sum() ## note: this choice is debatable. It makes the class_weight equal to the raw (not-smoothed) case
    tot_num_passive_pixels = tot_num_pixels - tot_num_active_pixels ## same result but more reliable depening on other choices.
    ## older version
    # tot_num_pixels = labels[trainval_idx][alphas[trainval_idx]==1].size
    # tot_num_active_pixels  = (labels[trainval_idx][alphas[trainval_idx]==1]).sum() ## note: this choice is debatable. It makes the class_weight equal to the raw (not-smoothed) case
    # tot_num_passive_pixels = tot_num_pixels - tot_num_active_pixels
    class_weight[1] = tot_num_passive_pixels/tot_num_pixels
    class_weight[0] = 1 - class_weight[1]
    # if args.masked == 0 :
else:
    class_weight = None
print("We re-weight pixels by their class using:  class_weight=", class_weight)

###################################################################################

print(model.parameters)
# print(model.state_dict())

####
train_losses = np.zeros(args.epochs)
train_auc    = np.zeros(args.epochs)
val_losses   = np.zeros(args.epochs)
val_auc      = np.zeros(args.epochs)

best_val_auc = 0
best_epoch = -1
best_val_loss = np.inf

for epoch in range(args.epochs):
    ## train part
    model.train()

    train_epoch_losses = []
    cat_trues = np.array([])
    cat_preds = np.array([])
    for batch in train_loader:

        loss, model_out = run_model(model, batch, loss_fn, class_weight, dvc, args.masked)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_epoch_losses.append(loss.item())

        ## compute the  AUC (a bit costly, but ok)
        if(args.masked):
            alpha_mask = (batch[3]>0).cpu()
        else:
            alpha_mask = (batch[3]>alpha_min).cpu()
        y_pred = model_out[alpha_mask].cpu()
        ## store stuff to predict the 1-epoch-average train loss or AUC
        cat_trues = np.concatenate((cat_trues, batch[1][alpha_mask]) , axis=None)
        cat_preds = np.concatenate((cat_preds, y_pred.cpu().detach()), axis=None)
        # fpr, tpr, _ = roc_curve(batch[1][alpha_mask], y_pred.cpu())

    ## handles the y-smooth case by rounding it, which is perfectly fgine, GT only slightly deviates from 0 or 1 when smoothed (it is enlarged in space)
    fpr, tpr, _ = roc_curve(np.array(np.round(cat_trues), dtype=int), cat_preds)
    train_auc_score = auc(fpr, tpr)
    train_auc[epoch] = train_auc_score
    train_losses[epoch] = np.mean(train_epoch_losses)


    ## validation part
    model.eval()
    with torch.no_grad():
        for batch in validation_loader: ## actually there is a single batch, it's required
            loss, model_out = run_model(model, batch, loss_fn, class_weight, dvc, args.masked)
            if(args.patience > 0):
                scheduler.step()    ## StepLR
                # scheduler.step(loss.item()) ReduceLROnPlateau

        val_losses[epoch] = loss.item()
        if(args.masked):
            alpha_mask = (batch[3]>0).cpu()
        else:
            alpha_mask = (batch[3]>alpha_min).cpu()
        y_pred = model_out[alpha_mask].cpu()
        ## handles the y-smooth case by rounding it, which is perfectly fgine, GT only smally deviates from 0 or 1.
        fpr, tpr, _ = roc_curve(np.array(np.round(batch[1][alpha_mask]  ), dtype=int),   y_pred.cpu()    )
        val_auc_score = auc(fpr, tpr)
        #if(epoch % 10 == 0):
        #    print("Epoch",epoch,val_auc_score, loss.item(), train_losses[-1])
        val_auc[epoch] = val_auc_score

    if (epoch > 20 and  val_auc_score > best_val_auc and train_auc_score > 0.51 ) :
        best_val_auc = val_auc_score
        best_epoch = epoch
        best_model_state_dict = model.state_dict().copy() ## dict: automatically a full copy ?
        best_val_loss = val_losses[epoch]

    # print(model.state_dict())
    print("train auc %.2f , val auc %.2f , train L %.5f , val L %.5f  , auc-gap %.2f  , L-gap %.5f , lag-since-best %i" \
        %(train_auc_score,  val_auc_score, train_losses[epoch], val_losses[epoch], best_val_auc - val_auc_score, val_losses[epoch]-best_val_loss, epoch- best_epoch ) \
        )

    if epoch - best_epoch > 150:
        print("It's been long since we didn't beat a record (validation AUC). We're probably into overfitting, and now we early-stop.")
        break

if best_epoch == -1:
    best_model_state_dict = model.state_dict()
    best_epoch = args.epochs-1

## with this setup we cannot see the overfitting, but plots are less crazy then
train_losses[best_epoch:] = train_losses[best_epoch]
train_auc   [best_epoch:] = train_auc   [best_epoch]
val_losses  [best_epoch:] = val_losses  [best_epoch]
val_auc     [best_epoch:] = val_auc     [best_epoch]


# save the model
torch.save(model.state_dict()   , rootname+"lastModel.t")
torch.save(best_model_state_dict, rootname+"model.t") ## better save the best model

model.load_state_dict(best_model_state_dict)

# save the losses (using early stopping, i.e. discard the later part)
np.savez(rootname+ "losses.npz", \
            input_file=args.filename,\
            train_losses=train_losses, \
            train_auc=train_auc, \
            val_losses=val_losses, \
            val_auc=val_auc,
            trainval_idx=trainval_idx,
            train_idx=split_train_idx,
            val_idx=split_val_idx,
            test_idx=test_idx,
            id_trainval = ids[trainval_idx],
            id_train=ids[split_train_idx],
            id_val=ids[split_val_idx],
            id_test = ids[test_idx])


model.eval()
with torch.no_grad():
    model_out = model(torch.tensor(input_dataset, dtype=torch.float32).to(dvc),
                      torch.tensor(alphas, dtype=torch.float32).to(dvc), args.masked)

    np.savez(rootname+ "ypred_dataset.npz", y_pred=model_out.cpu().numpy(),\
                                            y_true=labels)


ypreds=np.array(model_out.cpu().numpy())
fpr, tpr, _ = roc_curve( labels[test_idx][alphas[test_idx]>alpha_min], ypreds[test_idx][alphas[test_idx]>alpha_min])
auc_score = auc(fpr, tpr)
with open(rootname[:40]+"_test_aucs.txt", "a") as f:
    f.write( "%i  %f  \n" %(split, auc_score))



# ypred_mean=np.array(model_out.detach())
# day=0

# # nevent=11
# # train_utils.plot_nice_map(carts, rawlabels, rawalphas, ids, ypred_mean, nevent, day, alpha_min, args)
# # plt.savefig("typical heatmap event=%i_day=%i.png" % (nevent, day))

# # day=0
# # train_utils.plot_nice_map(carts, rawlabels, rawalphas, ids, ypred_mean, nevent, day, alpha_min, args)
# # plt.savefig("typical heatmap event=%i_day=%i.png" % (nevent, day))

# train_utils.plot_nice_map(carts_resum, rawlabels, rawalphas, ids, ypred_mean, nevent, day, alpha_min, args)
# plt.savefig("heatmap_event=%i_day=sum.png" % (nevent))
# plt.close()

# for nevent in split_val_idx:
#     train_utils.plot_nice_map(carts_resum, rawlabels, rawalphas, ids, ypred_mean, nevent, day, alpha_min, args)
#     plt.savefig("heatmap_event=%i_validation_split=%i_day=sum.png" % (nevent, split))
#     plt.close()

# for nevent in split_train_idx:
#     train_utils.plot_nice_map(carts_resum, rawlabels, rawalphas, ids, ypred_mean, nevent, day, alpha_min, args)
#     plt.savefig("heatmap_event=%i_train_split=%i_day=sum.png" % (nevent, split))
#     plt.close()


def unit_test_transformation(input_dataset, labels, alphas, comp_dict, carts, transformation_idx=0,nevent=11,day=0):


    # x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
    #             torch.tensor(input_dataset[nevent])[None, ...] , \
    #             torch.tensor(labels[nevent])[None, ...]  , \
    #             torch.tensor(alphas[nevent])[None, ...], \
    #             transformation_idx, \
    #             comp_dict['angular'][0::2], \
    #             comp_dict['angular'][1::2] )
    print("we don't try to do each transform, but try a single 90 rotation, apply it 4 times: we do not fall back... :(  ")
    x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
                torch.tensor(input_dataset[nevent])[None, ...] , \
                torch.tensor(labels[nevent])[None, ...]  , \
                torch.tensor(alphas[nevent])[None, ...], \
                1, \
                comp_dict['angular'][0::2], \
                comp_dict['angular'][1::2] )

    x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
                x_t[0][None, ...] , \
                y_t[0][None, ...]  , \
                a_t[0][None, ...], \
                1, \
                comp_dict['angular'][0::2], \
                comp_dict['angular'][1::2] )
    x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
                x_t[0][None, ...] , \
                y_t[0][None, ...]  , \
                a_t[0][None, ...], \
                1, \
                comp_dict['angular'][0::2], \
                comp_dict['angular'][1::2] )
    x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
                x_t[0][None, ...] , \
                y_t[0][None, ...]  , \
                a_t[0][None, ...], \
                1, \
                comp_dict['angular'][0::2], \
                comp_dict['angular'][1::2] )
    fig, (ax1, ax2) = plt.subplots(ncols=2) # , figsize=(10, 10))
    nlen = carts.shape[2]
    nwid = carts.shape[3]

    North = carts[nevent, day*3+0]
    East  = carts[nevent, day*3+1]

    North_rotated  = x_t[0, day*3+0]
    East_rotated   = x_t[0, day*3+1]

    arrow_sparsity=3
    arrow_scale=0.5
    ## build the arrow map
    ipxs = np.zeros(nlen*nwid)
    jpxs = np.zeros(nlen*nwid)
    dxs = np.zeros(nlen*nwid)
    dys = np.zeros(nlen*nwid)
    npix = 0
    for ipix in range(0,nlen,arrow_sparsity):
        for jpix in range(0,nwid,arrow_sparsity):
            ipxs[npix] = ipix
            jpxs[npix] = jpix
            dxs[npix] = East [ipix,jpix] * arrow_scale
            dys[npix] = North[ipix,jpix] * arrow_scale
            # if alpha_filter[ipix,jpix] == True:  ## only show arrow where the signal can be trusted
            ax1.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='black', length_includes_head=True,head_width=0.006)
            dxs[npix] = East_rotated [ipix,jpix] * arrow_scale
            dys[npix] = North_rotated[ipix,jpix] * arrow_scale
            ax2.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='black', length_includes_head=True,head_width=0.006)
            npix+=1
    plt.show()

## unit test: (for data augmentation)
# unit_test_transformation(input_dataset, labels, alphas, comp_dict, carts, transformation_idx=1,nevent=11,day=0)












####################################################################################################################
## New: this way we are sure this data has ot been trained on.. (except early-stopping on it)
# model.eval()
# with torch.no_grad():
#     model_out = model(torch.tensor(input_dataset[split_val_idx], dtype=torch.float32).to(dvc),
#                       torch.tensor(alphas[split_val_idx], dtype=torch.float32).to(dvc),args.masked)

#     np.savez(rootname+ "ypred_val.npz", y_pred=model_out.cpu().numpy())

# y =model_out.cpu().numpy()
# for ev in range(10):
#     plt.imshow( y[ev] )
#     plt.show()


# plt.figure()
# plt.plot(train_losses, label ='train loss')
# plt.plot(val_losses, label ='valid loss')
# plt.legend()
# plt.figure()
# plt.plot(train_auc, label ='train AUC')
# plt.plot(val_auc, label ='valid AUC')
# plt.legend()
# plt.close()







#torch.save(model.state_dict(), rootname+"model.t")
#
#train_losses = np.array(train_losses)
#val_losses   = np.array(val_losses)
#test_losses  = np.array(test_losses)


# special_trainPlusval_loader
## special train+validation prediction part
#special_auc =[]
#model.eval()
#with torch.no_grad():
#    for batch in special_trainPlusval_loader:
#        loss, model_out = run_model(model, batch, loss_fn, class_weight, dvc)
#        alpha_mask = (batch[3]>0).cpu()
#        y_pred = model_out[alpha_mask].cpu()
#        if args.ysmoothing < 0.01 :
#            fpr, tpr, _ = roc_curve(batch[1][alpha_mask], y_pred.cpu())
#            auc_score = auc(fpr, tpr)
#        else:
#            auc_score = 0.5
#        special_auc.append(auc_score)
## ypredtrain = model_out[] ##TODO: give already the splits? or just their index to avoid needless duplication
#np.savez(rootname+ "ypred_train+val.npz", yspe=model_out, split_train_idx=split_train_idx, split_val_idx=split_val_idx)



### y-pred distributions (for train+val set!) (assume low overfitting)
#plt.figure(figsize=(10,6))
#ys_pred = model_out[alpha_mask]
#ys_true = batch[1][alpha_mask]
#bins=np.linspace(0,1,100)
#Pofy_given_0, _ = np.histogram(ys_pred[ys_true<0.5].flatten(),bins, density=True)
#Pofy_given_1, _ = np.histogram(ys_pred[ys_true>=0.5].flatten(),bins, density=True)
#plt.title("y-pred distributions (for train+val set!) (assume low overfitting)")
#plt.plot(bins[1:],Pofy_given_0, label=r"$P(y^{pred}|y^{true}<0.5)$")
#plt.plot(bins[1:],Pofy_given_1, label=r"$P(y^{pred}|y^{true}>0.5)$", color= 'red')
#plt.legend()
#plt.xlim([0,1])
#plt.ylim([0,5])
#plt.savefig(rootname+ "P(y|y=0,1).png")
#plt.close()
#
#
### losses and val AUC as function of epochs
#best_epoch = np.argmin(val_losses)
#fig, ax = plt.subplots(figsize=(10,6))
#ax.semilogy(val_losses, label='Validation Loss')
#ax.plot(test_losses, label='Test Loss')
#ax.plot(train_losses, label='Train Loss')
#ax.axvline(best_epoch, ls="--", color='k')
#ax.set_xlabel("Epoch", fontsize=20)
#ax.set_ylabel("Loss", fontsize=20)
#ax.set_xlim([0,200])
#ax.set_ylim([0.005,0.1])
#ax2 = ax.twinx()
#ax2.plot(val_auc, ls="--", label='Validation AUC')
## ax2.plot(test_auc, label='Test AUC')
#ax2.set_ylabel("AUC", fontsize=20)
#ax2.set_xlim([0,200])
#ax2.set_ylim([0.5,1])
## Add legend of the second y-axis to the first y-axis
#handles1, labels1 = ax.get_legend_handles_labels()
#handles2, labels2 = ax2.get_legend_handles_labels()
#ax.legend(handles1 + handles2, labels1 + labels2, fontsize=20)
#plt.title("split="+str(split)+" best epoch:"+str(best_epoch)\
#    +" L_val*="   +str( np.round(val_losses[best_epoch]  ,4)) \
#    +" L_train*=" +str( np.round(train_losses[best_epoch],4)))
#plt.savefig(rootname+ "losses+auc.png")
#plt.close()
#
#
#for nev in range(2):
#    nevent = split_val_idx[0]+nev
#    day=0
#    ypred = model_out[nevent]
#    outputname = rootname+"heatmaps_id="+str(ids[nevent])+"_day="+str(day)+".png"
#    plot_nice_map(nevent, day, alpha_min, ypred, labels, carts, rawlabels, rawalphas, ids,outputname=outputname)
#    # plt.show()
#    plt.close()
#
#if 0:
#    num_active_pix = (alphas>0.0).sum(1).sum(1)
#    plt.scatter(np.arange(num_active_pix.shape[0]), num_active_pix)
#    plt.xlabel("event number")
#    plt.ylabel("#(alpha>0) pixels")
#    plt.savefig("number of active pixels per sample")
#    plt.show()
#

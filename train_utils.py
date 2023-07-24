import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

# from numba import njit

# @njit(nogil=True)
def enlarge(y, amount):
    y_enlarged = np.copy(y)
    indices = np.argwhere(y > 0)
    for index in indices:
        for row in range(max(index[1]-amount,0), min(index[1]+amount+1, y.shape[1])):
            for col in range(max(index[2]-amount,0), min(index[2]+amount+1, y.shape[2])):
                y_enlarged[index[0], row, col] = y[index[0], index[1], index[2]]
    return y_enlarged

def apply_augmentation_2d_polar(x, y,alpha, k, sin_channels, cos_channels):
    if(k==0):
        return torch.clone(x), torch.clone(y), torch.clone(alpha)
    elif(k==4):
        f_x = torch.flip(torch.clone(x), dims=[3])
        f_y = torch.flip(torch.clone(y), dims=[2])
        f_alpha = torch.flip(torch.clone(alpha), dims=[2])
        f_x[:, cos_channels, ...] = - f_x[:, cos_channels, ...]
        return f_x, f_y, f_alpha
    else:
        # x shape is (samples, channels, pixel, pixel)
        # torch.rot90(..., k={1,2,3}, dims=[2,3])
        r_x = torch.rot90(x, k, dims=[2,3])
        # y/alpha shape is (samples, pixel, pixel)
        # torch.rot90(..., k={1,2,3}, dims=[1,2])
        r_y = torch.rot90(y, k, dims=[1,2])
        r_alpha = torch.rot90(alpha, k, dims=[1,2])
        sins = torch.clone(r_x[:,sin_channels,...])
        coss = torch.clone(r_x[:,cos_channels,...])
        if(k==1):
            r_x[:,sin_channels,... ] = coss
            r_x[:, cos_channels,...] = -sins
        elif(k==2):
            r_x[:,sin_channels,... ] =  -sins
            r_x[:, cos_channels,...] = -coss
        elif(k==3):
            r_x[:,sin_channels,... ] = coss
            r_x[:, cos_channels,...] = -sins
        return r_x, r_y, r_alpha

class GpsPolarDataset(Dataset):

    def __init__(self, inputs,labels, ids, alphas,comp_dict, transforms = [(0)]):
        self._x = torch.tensor(inputs, dtype=torch.float32)
        self._y = torch.tensor(labels, dtype=torch.float32)
        self._alpha = torch.tensor(alphas, dtype=torch.float32)
        self._id = torch.tensor(ids, dtype=torch.long)
        self._transfs = transforms
        self._cdict = comp_dict
    def __len__(self):
        return len(self._id)
    def __getitem__(self, idx):
        with torch.no_grad():
            transf_idx = int(np.random.choice(len(self._transfs)))
            transf = self._transfs[transf_idx]
            x_t, y_t, a_t = apply_augmentation_2d_polar(self._x[idx][None,...],
                                                        self._y[idx][None,...],
                                                        self._alpha[idx][None,...],
                                                        transf[0],
                                                        self._cdict['angular'][0::2],
                                                        self._cdict['angular'][1::2])
            if(len(transf)>1):
                x_t, y_t, a_t = apply_augmentation_2d_polar(x_t,
                                                        y_t,
                                                        a_t,
                                                        transf[1],
                                                        self._cdict['angular'][0::2],
                                                        self._cdict['angular'][1::2])
            out_tuple = (x_t[0], y_t[0], self._id[idx], a_t[0])
            return out_tuple




def plot_nice_map(carts, rawlabels, rawalphas, ids, ypred_mean, nevent, day, alpha_min, args, arrow_sparsity = 3, arrow_scale=0.2, outputname="_.png", vmax=20, w1=0.013436042151292123):
    # alphas = rawalphas[nevent]
    if args.twoDim :
        dim =2
    else:
        dim = 3

    North = carts[nevent, day*dim+0]
    East  = carts[nevent, day*dim+1]
    ys    = rawlabels[nevent]
    ypred = ypred_mean[nevent]  # single event

    nlen = carts.shape[2]
    nwid = carts.shape[3]

    if args.twoDim :
        norm = (North**2 + East**2)**0.5
    else:
        Up    = carts[nevent, day*dim+2]
        norm = (North**2 + East**2 + Up**2)**0.5
    ys_scatter = np.where(ys==1)
    alpha_filter = (rawalphas[nevent]>alpha_min)
    stations_locs = np.where(rawalphas[nevent]>0.997)

    fpr, tpr, _ = roc_curve(rawlabels[nevent][alpha_filter],ypred[alpha_filter])
    auc_score = auc(fpr, tpr)

    # w1 = labels[alpha_mask].mean()
    train_val_loss = ((1-w1)*(rawlabels[nevent]  )* np.log(  ypred+1e-10) \
                        + w1*(1-rawlabels[nevent])* np.log(1-ypred+1e-10)   )[alpha_filter].mean()


    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(10, 10))
    alphacov = int(np.mean(alpha_filter)*100)

    fig.suptitle("id="+str(ids[nevent])+r", $\overline{\alpha}$="+str(alphacov)+"%"  + "  AUC="+str(auc_score)[:4] + "  Loss="+str(train_val_loss)[:7])

    heatmap0 = norm * alpha_filter
    im0 = ax1.imshow(heatmap0.T, vmin=0, vmax=vmax)
    ax1.set_title("Signal Norm")
    fig.colorbar(im0, ax=ax1, shrink=0.3)

    heatmap2 = ypred*alpha_filter
    im2 = ax2.imshow(heatmap2.T, vmin=0, vmax=1)
    ax2.set_title('predicted y')
    fig.colorbar(im2, ax=ax2, shrink=0.3)

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
            if alpha_filter[ipix,jpix] == True:  ## only show arrow where the signal can be trusted
                ax1.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='black', length_includes_head=True)
                ax2.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='black', length_includes_head=True)
            npix+=1

    ax1.scatter(stations_locs[0],stations_locs[1], color='k', marker= 's')
    ax2.scatter(stations_locs[0],stations_locs[1], color='k', marker= 's')
    ax1.scatter(ys_scatter[0], ys_scatter[1], color='red', marker= 's')
    ax2.scatter(ys_scatter[0], ys_scatter[1], color='red', marker= 's')

    ax1.xlim([0,50])
    ax2.xlim([0,50])
    ax1.ylim([0,50])
    ax2.ylim([0,50])
    plt.savefig(outputname)
    fig.tight_layout()
    # plt.show()
    return (ipxs, jpxs, dxs, dys)




# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.imshow(labels[nevent])
# transformation_idx=2
# x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
#             torch.tensor(input_dataset[nevent])[None, ...] , \
#             torch.tensor(labels[nevent])[None, ...]  , \
#             torch.tensor(alphas[nevent])[None, ...], \
#             transformation_idx, \
#             comp_dict['angular'][0::2], \
#             comp_dict['angular'][1::2] )

# ax2.imshow(y_t[0])
# plt.show()

# fig, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.imshow(alphas[nevent])
# transformation_idx=1
# x_t, y_t, a_t =train_utils.apply_augmentation_2d_polar(\
#             torch.tensor(input_dataset[nevent])[None, ...] , \
#             torch.tensor(labels[nevent])[None, ...]  , \
#             torch.tensor(alphas[nevent])[None, ...], \
#             transformation_idx, \
#             comp_dict['angular'][0::2], \
#             comp_dict['angular'][1::2] )
# ax2.imshow(a_t[0])
# plt.show()




# ## build the arrow map
# ipxs = np.zeros(nlen*nwid)
# jpxs = np.zeros(nlen*nwid)
# dxs = np.zeros(nlen*nwid)
# dys = np.zeros(nlen*nwid)
# npix = 0
# for ipix in range(0,nlen,arrow_sparsity):
#     for jpix in range(0,nwid,arrow_sparsity):
#         ipxs[npix] = ipix
#         jpxs[npix] = jpix
#         dxs[npix] = East [ipix,jpix] * arrow_scale
#         dys[npix] = North[ipix,jpix] * arrow_scale
#         if alpha_filter[ipix,jpix] == True:  ## only show arrow where the signal can be trusted
#             ax1.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='black')
#             # ax2.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='white')
#             # ax3.arrow(ipxs[npix], jpxs[npix], dxs[npix], dys[npix] , color='black')
#         npix+=1

# ax1.scatter(stations_locs[0],stations_locs[1], color='k', marker= 's')
# # ax2.scatter(stations_locs[0],stations_locs[1], color='white', marker= 's')
# # ax3.scatter(stations_locs[0],stations_locs[1], color='k', marker= 's')
# ax1.scatter(ys_scatter[0], ys_scatter[1], color='red', marker= 's')
# # ax2.scatter(ys_scatter[0], ys_scatter[1], color='red', marker= 's')
# # ax3.scatter(ys_scatter[0], ys_scatter[1], color='red', marker= 's')
# plt.show()


def format_trainingoutput_filename(args):
        head_len = len(args.filename.split("_")[0])+1
        rootname = args.outfolder + args.decorator + "res_"+args.filename[head_len:-4]
        rootname += "_yt"+str(args.year_test)
        rootname += args.decorator
        rootname += "_alpha="+str(args.alpha_min)
        # rootname += "_alpha_max_dist="+str(args.alpha_max_dist)
        rootname += "_ysmoo="+str(args.ysmoothing)
        rootname += "_reg="+str(int(args.regression))
        rootname += "_2D="+str(int(args.twoDim))
        rootname += "_strain="+str(int(args.strain))
        rootname += "_pol="+str(1-int(args.cartesian))
        rootname += "_drp="+str(args.dropout)
        rootname += "_beta="+str(args.weight_decay)
        rootname += "_lr="+str(args.lr)
        rootname += "_LRSp="+str(args.patience) # Learning Rate Scheduler Patience
        rootname += "_mb="+str(args.batch_size)
        rootname += "_ag="+str(int(args.augmentation))
        rootname += "_enl="+str(int(args.enlarge))
        rootname += "_ep="+str(args.epochs)
        rootname += "_bs="+str(int(args.bootstrap_splits))
        rootname += "_msk="+str(int(args.masked))
        rootname += "_sR="+str(int(args.stdRadial))
        rootname += "_kR="+str(int(args.killRadial))
        rootname += "_kA="+str(int(args.killAngular))
        rootname += "_dcB="+str(int(args.distToCenterBaseline))
        rootname += "_rNB="+str(int(args.radialNormBaseline))
        rootname += "_xclu="+str(int(args.exclusionList))
        rootname += "_split="+str(args.split)
        rootname += "_"
        return rootname
       
# def load_gps_dataset(filename, alpha_min, twoDimensional=False):
#     #load dataset
#     dict_dataset = np.load(filename, allow_pickle=True).item()
#     carts = []  # cartesian signal
#     ys = []
#     alphas = []     # alphas (confidence level)
#     days = []   ## absolute day of the MS
#     ids = []
#     regions = []
#     for id, (x, y, day, reg) in dict_dataset.items():
#         alpha = x[:,:,:,-1]
#         alpha[np.isnan(alpha)] = 0
#         #alpha_score = np.heaviside(alpha-alpha_min, 0).sum(axis=(1,2))/(alpha.shape[1]*alpha.shape[2])
#         #if((alpha_score > 0).all()):
#         ids.append(id)
#         ys.append(y)
#         x = x[...,:-1]
#         ## convert alpha\in [0,1] to alpha\in {0,1}
#         alpha = np.heaviside(alpha-alpha_min,0)[...,None]
#         alpha = np.repeat(alpha, repeats=x.shape[-1], axis=-1)
#         alpha[np.isnan(x)] = 0
#         x[np.isnan(x)] = 0
#         carts.append(x)
#         alphas.append(alpha)
#         days.append(day)
#         regions.append(reg)
#     carts = np.array(carts)
#     ys = np.array(ys)
#     alphas = np.array(alphas)[:,-1,:,:,0] ## taking last day and 1st component
#     days = np.array(days)
#     ids = np.array(ids)
#     regions = np.array(regions)


#     # polar repr.
#     if(twoDimensional):
#         ## shape is pixel,pixel,Ndays+3 (or +5)
#         polars = np.zeros((carts.shape[:-1] + (3,)))
#         polars[...,0] = np.sqrt(carts[...,0]**2+carts[...,1]**2)
#         polars[...,1] = carts[...,0]/polars[..., 0]
#         polars[...,2] = carts[...,1]/polars[...,0]
#     else:
#         polars = np.zeros((carts.shape[:-1] + (5,)))
#         polars[...,0] = np.sqrt(carts[...,0]**2+carts[...,1]**2)
#         polars[...,1] = np.abs(carts[...,2])
#         polars[...,2] = carts[...,0]/polars[..., 0]
#         polars[...,3] = carts[...,1]/polars[...,0]
#         polars[...,4] = carts[...,2]/np.linalg.norm(carts, axis=-1)

#     # feature sizes
#     n_days = carts.shape[1] # here the days  are given by 2nd components
#     n_space_components = carts.shape[-1] ## ==3
#     n_polar_components = polars.shape[-1] ## == 3 or 5

#     # reshaping + transposing
#     carts  = carts.transpose((0,2,3,1,4)) # send the days axis after the pixels
#     carts  = carts.reshape((carts.shape[:3] + (-1,))).transpose((0, 3, 1, 2)) # carts.shape[:3]==(Nsamples,pixel,pixel)
#     polars = polars.transpose((0,2,3,1,4))
#     polars = polars.reshape((polars.shape[:3] + (-1,))).transpose((0, 3, 1, 2)) # transpose((0, 3, 1, 2)): put the channels before (pixel,pixel) (for torch)


#     # feature indices
#     north_components = np.arange(0, n_days*n_space_components, n_space_components)
#     east_components  = np.arange(1, n_days*n_space_components, n_space_components)
#     vector_components = np.vstack([north_components,
#                                    east_components]).T.flatten()

#     ## scalar_components: relate to the cartesian representation (telling us how the object transforms under rotation)
#     if(twoDimensional):
#         scalar_components = None
#     else:
#         scalar_components = np.arange(2, n_days*n_space_components, n_space_components)
#     if(twoDimensional):
#         radial_components = np.arange(0, n_days*n_polar_components, n_polar_components)
#         angular_components = np.vstack([np.arange(1, n_days*n_polar_components, n_polar_components),
#                                         np.arange(2, n_days*n_polar_components, n_polar_components)]).T.flatten()
#     else:
#         radial_components = np.vstack( [np.arange(0, n_days*n_polar_components, n_polar_components),
#                                         np.arange(1, n_days*n_polar_components, n_polar_components)]).T.flatten()
#         angular_components = np.vstack([np.arange(2, n_days*n_polar_components, n_polar_components),
#                                         np.arange(3, n_days*n_polar_components, n_polar_components),
#                                         np.arange(4, n_days*n_polar_components, n_polar_components)]).T.flatten()
#     return carts, polars, ys, alphas, days, ids, regions, n_days,north_components,east_components,vector_components,scalar_components,radial_components,angular_components

class TwoColumnConv(nn.Module):
    def __init__(self, f1, f2,  channels, kernels, dropout=0.2, regression=False, padding_mode='replicate') -> None:
        super(TwoColumnConv,self).__init__()
        self.feats_1 = f1
        self.feats_2 = f2

        (c1_mod_channels, c2_mod_channels, c_mod_channels) = channels
        (c1_mod_kernels , c2_mod_kernels,  c_mod_kernels ) = kernels

        self.c1_mod = nn.ModuleList()
        for i, (c_in, c_out, k) in enumerate(zip(c1_mod_channels[:-1], c1_mod_channels[1:], c1_mod_kernels)):
            self.c1_mod.append(nn.Sequential(nn.Conv2d(c_in,
                c_out,
                kernel_size=k, 
                stride=1,
                padding='same',
                padding_mode=padding_mode),
                #self.c1_mod.append(nn.BatchNorm2d(c_out))
                nn.Dropout(dropout),
                nn.ReLU()))

        self.c2_mod = nn.ModuleList()
        for i, (c_in, c_out, k) in enumerate(zip(c2_mod_channels[:-1], c2_mod_channels[1:], c2_mod_kernels)):
            self.c2_mod.append(
                nn.Sequential(nn.Conv2d(c_in,
                    c_out,
                    kernel_size=k, 
                    stride=1,
                    padding='same',
                    padding_mode=padding_mode),
                    #self.c2_mod.append(nn.BatchNorm2d(c_out))
                    nn.Dropout(dropout),
                    nn.ReLU()))
        
        self.c_mod = nn.ModuleList()
        for i, (c_in, c_out, k) in enumerate(zip(c_mod_channels[:-1], c_mod_channels[1:], c_mod_kernels)):
            block = nn.Sequential()
            block.append(nn.Conv2d(c_in,
                c_out,
                kernel_size=k, 
                stride=1,
                padding='same',
                padding_mode=padding_mode))
            
            if(i < len(c_mod_kernels)-1):
                #self.c_mod.append(nn.BatchNorm2d(c_out))
                block.append(nn.Dropout(dropout))
                block.append(nn.ReLU())
            self.c_mod.append(block)
        if(regression):
            self.c_mod.append(nn.Conv2d(c_mod_channels[-1], c_mod_channels[-1], 1, 1))
            self.act_f = nn.ReLU()
        else:
            self.act_f = nn.Sigmoid()

    def forward(self, x, mask, use_mask=True):

        ## there are two parallel nets at first:
        x_1 = x[:,self.feats_1,...]
        for m1 in self.c1_mod:
            x_1 = m1(x_1)
            if(use_mask):
                x_1 = x_1*mask[:,None,...]

        x_2 = x[:,self.feats_2,...]
        for m2 in self.c2_mod:
            x_2 = m2(x_2)
            if(use_mask):
                x_2 = x_2*mask[:,None,...]

        ## then merged into 1:
        x_comb = torch.cat([x_1,x_2], dim=1)
        for m in self.c_mod:
            x_comb = m(x_comb)
            if(use_mask):
                x_comb = x_comb*mask[:,None,...]
        if(use_mask):
            return (self.act_f(x_comb)*mask[:,None,...])[:,0,...]
        else:
            return self.act_f(x_comb)[:,0,...]



class SingleColumConv(nn.Module):
    def __init__(self, channels, kernels, dropout=0.2, padding_mode='replicate') -> None:
        super(SingleColumConv,self).__init__()

        ## dirt, to go fast:
        (c1_mod_channels, c2_mod_channels, c_mod_channels) = channels
        (c1_mod_kernels , c2_mod_kernels,  c_mod_kernels ) = kernels
        channels = c_mod_channels
        kernels = c_mod_kernels

        self.c_mod = nn.ModuleList()
        for i, (c_in, c_out, k) in enumerate(zip(channels[:-1], channels[1:], kernels)):
            block = nn.Sequential()
            block.append(nn.Conv2d(c_in,
                c_out,
                kernel_size=k,
                stride=1,
                padding='same',
                padding_mode=padding_mode))

            if(i < len(kernels)-1):
                #self.c_mod.append(nn.BatchNorm2d(c_out))
                block.append(nn.Dropout(dropout))
                block.append(nn.ReLU())
            self.c_mod.append(block)
        self.act_f = nn.Sigmoid()


    def forward(self, x, mask, use_mask=True):
        for m in self.c_mod:
            x = m(x)
            if(use_mask):
                x = x*mask[:,None,...]
        if(use_mask):
            return (self.act_f(x)*mask[:,None,...])[:,0,...]
        else:
            return self.act_f(x)[:,0,...]





class sigmoid(nn.Module):
    def __init__(self) -> None:
        super(sigmoid,self).__init__()

        self.c_mod = nn.ModuleList()
        block = nn.Sequential()
        block.append(nn.Conv2d(1,
            1,
            kernel_size=1,
            stride=1,
            padding='same',
            padding_mode='replicate'))
        self.c_mod.append(block)
        self.act_f = nn.Sigmoid()

    def forward(self, x, mask, use_mask=True):
        for m in self.c_mod:
            x = m(x)
        if(use_mask):
            return (self.act_f(x)*mask[:,None,...])[:,0,...]
        else:
            return self.act_f(x)[:,0,...]

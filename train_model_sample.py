from time import perf_counter as t
import random
import torch
import torch.nn.functional as F
import logging
import numpy as np
from models import GCL_clu
from torch.optim.lr_scheduler import ExponentialLR
from termcolor import cprint
import models.GCL_loader
from imbalanced import ImbalancedDatasetSampler
from metrics import compute_metrics
from tqdm import tqdm


def supposloss(h_1, h_2, mask, running_device):
    """
    multi-positive contrastive learning loss function
    Args:
        h_1,h_2: data augmented embedding representations
        mask: matrix of positive and negative relationships between samples
    Returns: loss
    """
    temp = 0.07
    base_temp = 0.07
    sample_sim = torch.div(torch.matmul(h_1, h_2.T), temp)
    # for numerical stability
    logits_max, _ = torch.max(sample_sim, dim=1, keepdim=True)
    logits = sample_sim - logits_max.detach()
    # tile mask, mask used for filling the diagonal with 0
    logits_mask = torch.scatter(
        torch.ones_like(mask), 1,
        torch.arange(sample_sim.shape[0]).view(-1, 1).to(running_device), 0)
    mask = mask * logits_mask
    # compute log_prob
    exp_logits = torch.exp(logits) * logits_mask  # exclude diagonal
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
    # add 1e-20 to avoid inexistence of positive anchors
    mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-20)
    # loss
    loss_sample = - (temp / base_temp) * mean_log_prob_pos
    loss_sample = loss_sample.mean()

    return loss_sample

def correlation_reduction_loss(S):
    """
    the correlation reduction loss L: MSE for S and I (identical matrix)
    Args:
        S: the cross-view correlation matrix S
    Returns: L
    """
    return torch.diagonal(S).add(-1).pow(2).mean() + off_diagonal(S).pow(2).mean()
def off_diagonal(x):
    """
    off-diagonal elements of x
    Args:
        x: the input matrix
    Returns: the off-diagonal elements of x
    """
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def train_scMPCL(args, adata):
    """
        Training nsDCC model with adata
        Args:
            args: parameters
            adata: the input adata
    """
    # ===================================================#
    # Setting the random seed
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    # ===================================================#
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    running_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ===================================================#
    # Setting data augmentation
    args_transformation = [{
        # mask
        'mask_percentage': 0.8, 'apply_mask_prob': 0.8,
        # (Add) gaussian noise
        'noise_percentage': 0.8, 'sigma': 0.5, 'apply_noise_prob': 0
    },{
        # mask
        'mask_percentage': 0.8, 'apply_mask_prob': 0.0,
        # (Add) gaussian noise
        'noise_percentage': 0.8, 'sigma': 0.5, 'apply_noise_prob': 0.8
    }]

    train_dataset = models.GCL_loader.scRNAMatrixInstance(
        adata=adata,
        obs_label_colname=args.dataset_class,
        transform=True,
        args_transformation_list=args_transformation
    )
    num_cluster = len(train_dataset.unique_label)
    print("Train dataset num cells: ", train_dataset.num_cells)
    args.clu_cfg.append(num_cluster)

    # adjust batch_size according to sample size
    if train_dataset.num_cells < 512:
        args.batch_size = train_dataset.num_cells
        args.pcl_r = train_dataset.num_cells

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True,
        sampler=ImbalancedDatasetSampler(train_dataset, weights=train_dataset.weights), drop_last=True)
    eval_sampler = None
    eval_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, sampler=eval_sampler, drop_last=False)

    # =======================Create Model============================#
    num_feature = train_dataset.num_genes # 输入数据维度
    model = GCL_clu(num_feature, out_channel=args.hidden_dim, fea_dropout=args.fea_dropout,
                    clu_cfg=args.clu_cfg, clu_dropout=args.clu_dropout, clu_batch_norm=True)
    model.to(running_device)
    # define the Adam optimizer and set the decay of learning rate
    train_optimizer = torch.optim.Adam(model.parameters(), lr=args.train_lr, weight_decay=args.weight_decay)
    train_scheduler = ExponentialLR(train_optimizer, gamma=0.99)

    # =======================Train============================#
    print("========= Train! =========")
    start = t() # statistic runtime
    earlystop_count = 0 # for determining whether to stop early
    min_loss = 1e4
    for epoch in tqdm(range(args.train_epochs + 1)):
        model.train()
        # store loss values for model tuning
        loss_all = 0
        iters = len(train_loader)

        for i, (images, index, label) in enumerate(train_loader):
            # images[1] corresponds to the samples that have been augmented by random mask
            # images[1] corresponds to the samples that have been augmented by gaussian noise
            images[1] = images[1].to(running_device)
            images[2] = images[2].to(running_device)
            h_1, pred_1 = model(images[1], epoch)
            h_2, pred_2 = model(images[2], epoch)

            # ------------Instance-level contrast (multiple positive sample contrastive loss function)------------
            index = index.reshape((1, -1))
            index_combined = torch.cat((index, index), dim=1)
            h_combined = torch.cat((h_1, h_2), dim=0)
            sample_mask = torch.eq(index_combined, index_combined.T).float().to(running_device)
            loss_sample = supposloss(h_combined, h_combined, sample_mask, running_device)


            # ------------cluster-level contrast (unit matrix constraint)------------
            sim = torch.mm(F.normalize(pred_1.t(), p=2, dim=1), F.normalize(pred_2, p=2, dim=0)).to('cuda')
            loss_clu_sim = correlation_reduction_loss(sim)
            loss_clu = loss_clu_sim

            # total loss
            loss = loss_sample + loss_clu
            loss_all += loss

            # backward
            loss.backward()
            train_optimizer.step()
            train_scheduler.step()

        loss_all = loss_all / iters

        # If the loss does not decrease in 100 epochs, then the training process stops early
        if loss_all < min_loss:
            earlystop_count = 0
            min_loss = loss_all
            # save the current model
            torch.save(model.state_dict(), args.train_path)
        else:
            earlystop_count += 1
        if earlystop_count == 100:
            break
    noe = t()
    print('Train total time: ', noe - start)

    # 加载新模型测试预训练结果
    new_model = GCL_clu(num_feature, out_channel=args.hidden_dim, fea_dropout=args.fea_dropout,
                    clu_cfg=args.clu_cfg, clu_dropout=args.clu_dropout, clu_batch_norm=True)
    new_model.to(running_device)
    new_model.load_state_dict(torch.load(args.train_path))
    new_model.eval()
    embs_a = torch.Tensor()
    pred_a_lables = torch.Tensor()
    gt_labels = torch.Tensor()
    pred_a_orig = torch.Tensor()
    for i, (images, index, label) in enumerate(eval_loader):
        # get embedding
        h_a, pred_a = new_model(images[0].to(running_device))
        if i == 0:
            embs_a = h_a
            gt_labels = label
            pred_a_orig = pred_a
            pred_a = pred_a.argmax(dim=1)
            pred_a_lables = pred_a
        else:
            embs_a = torch.cat((embs_a, h_a), dim=0)
            gt_labels = torch.cat((gt_labels, label), dim=0)
            pred_a_orig = torch.cat((pred_a_orig, pred_a), dim=0)
            pred_a = pred_a.argmax(dim=1)
            pred_a_lables = torch.cat((pred_a_lables, pred_a), dim=0)
    embs_a = embs_a / embs_a.norm(dim=1)[:, None]
    embs_a = np.array(embs_a.cpu().detach().numpy())

    logging.info("========== nsDCC testing process ==========")
    logging.info("Cluster num is set to {}".format(num_cluster))
    print("========== nsDCC testing process ==========")
    print("Cluster num is set to {}".format(num_cluster))
    cluHead_eval_supervised_metrics = compute_metrics(embs_a, gt_labels.numpy(), pred_a_lables.cpu().numpy())
    # the results of the evaluation
    logging.info("Clustering performance of nsDCC: {}".format(cluHead_eval_supervised_metrics))
    print("Clustering performance of nsDCC: {}".format(cluHead_eval_supervised_metrics))




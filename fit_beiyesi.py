import cv2
import os
import random
import torch
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from .Miou import Pa,calculate_miou
import torch.nn.functional as F
import torch.nn as nn

def write_options(model_savedir,args,best_acc_val):
    aaa = []
    aaa.append(['lr',str(args.lr)])
    aaa.append(['batch',args.batch_size])
    aaa.append(['save_name',args.save_name])
    aaa.append(['seed',args.batch_size])
    aaa.append(['best_val_acc',str(best_acc_val)])
    aaa.append(['warm_epoch',args.warm_epoch])
    aaa.append(['end_epoch',args.end_epoch])
    f = open(model_savedir+'option'+'.txt', "a")
    for option_things in aaa:
        f.write(str(option_things)+'\n')
    f.close()


def sample_normal_jit(mu, log_var):
    sigma = torch.exp(log_var / 2)
    eps = mu.mul(0).normal_()
    z = eps.mul_(sigma).add_(mu)
    return z, eps

def set_seed(seed=1): # seed的数值可以随意设置，本人不清楚有没有推荐数值
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    #根据文档，torch.manual_seed(seed)应该已经为所有设备设置seed
    #但是torch.cuda.manual_seed(seed)在没有gpu时也可调用，这样写没什么坏处
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def fit(epoch,epochs, model, trainloader, valloader,device,criterion,optimizer,CosineLR,args):
    # with tqdm(total=len(trainloader), ncols=120, ascii=True) as t:
    scaler = GradScaler()
    if torch.cuda.is_available():
        model.to('cuda')
    running_loss = 0
    model.train()
    train_pa_whole = 0
    train_iou_whole = 0
    epoch_iou = 0
    phi_rho = args.phi_rho
    gamma_rho = args.gamma_rho
    phi_upsilon = args.phi_upsilon
    gamma_upsilon = args.gamma_upsilon
    phi_omega = args.phi_omega
    gamma_omega = args.gamma_omega
    alpha_pi = args.alpha_pi
    beta_pi = args.beta_pi
    sigma_0 = args.sigma_0
    Dx = torch.zeros([1, 2, 3, 3], dtype=torch.float)
    Dx[:, :, 1, 1] = 1
    Dx[:, :, 1, 0] = Dx[:, :, 1, 2] = Dx[:, :, 0, 1] = Dx[:, :, 2, 1] = -1 / 4
    Dx = nn.Parameter(data=Dx, requires_grad=False)

    for batch_idx, (imgs, masks) in enumerate(trainloader):
        # t.set_description("Train(Epoch{}/{})".format(epoch, epochs))
        imgs, masks_cuda = imgs.to(device), masks.to(device)
        imgs = imgs.float()
        with autocast():
            masks_pred, gx, mu_x, log_var_x, gm, mu_m, log_var_m, gz, mu_z, log_var_z = model(imgs)
            masks_cuda = masks_cuda.squeeze(1)
            K = 2
            _, _, W, H = imgs.shape
            a = torch.unsqueeze(gx[:, 0, :, :], dim=1)
            b = torch.unsqueeze(gm[:, 0, :, :], dim=1)
            gx = torch.cat([gx, a], dim=1)
            gm = torch.cat([gm, b], dim=1)
            residual = imgs - (gx + gm)
            mu_rho_hat = (2 * gamma_rho + 1) / (
                    residual * residual + 2 * phi_rho
            )
            # mu_rho_hat = torch.clamp(mu_rho_hat, 1e4, 1e8)

            N = torch.sum(mu_rho_hat).detach()
            n, _ = sample_normal_jit(gm, torch.log(1 / mu_rho_hat))

            # Image line upsilon
            alpha_upsilon_hat = 2 * gamma_upsilon + K
            Dx = Dx.to('cuda').to(torch.float16)
            difference_x = F.conv2d(mu_x, Dx, padding=1)
            beta_upsilon_hat = (
                    torch.sum(
                        mu_z * (difference_x * difference_x + 2 * torch.exp(log_var_x)),
                        dim=1,
                        keepdim=True,
                    )
                    + 2 * 1e-8
            )  # B x 1 x W x H
            mu_upsilon_hat = alpha_upsilon_hat / beta_upsilon_hat
            # mu_upsilon_hat = torch.clamp(mu_upsilon_hat, 1e6, 1e10)

            mu_z_expanded = mu_z.expand(-1, 2, -1, -1)
            # Seg boundary omega
            difference_z = F.conv2d(
                mu_z_expanded, Dx.expand(K, 2, 3, 3), padding=1, groups=1
            )  # B x K x W x H
            alpha_omega_hat = 2 * gamma_omega + 1
            pseudo_pi = torch.mean(mu_z, dim=(2, 3), keepdim=True)
            beta_omega_hat = (
                    pseudo_pi * (difference_z * difference_z + 2 * torch.exp(log_var_z))
                    + 2 * phi_omega
            )
            mu_omega_hat = alpha_omega_hat / beta_omega_hat
            # mu_omega_hat = torch.clamp(mu_omega_hat, 1e2, 1e6)

            # Seg category probability pi
            _, _, W, H = imgs.shape
            alpha_pi_hat = alpha_pi + W * H / 2
            beta_pi_hat = (
                    torch.sum(
                        mu_omega_hat * (difference_z * difference_z + 2 * torch.exp(log_var_z)),
                        dim=(2, 3),
                        keepdim=True,
                    )
                    / 2
                    + beta_pi
            )
            digamma_pi = torch.special.digamma(
                alpha_pi_hat + beta_pi_hat
            ) - torch.special.digamma(beta_pi_hat)

            # compute loss-related
            kl_y = residual * mu_rho_hat.detach() * residual

            kl_mu_z = torch.sum(
                digamma_pi.detach() * difference_z * mu_omega_hat.detach() * difference_z,
                dim=1,
            )
            kl_sigma_z = torch.sum(
                digamma_pi.detach()
                * (2 * torch.exp(log_var_z) * mu_omega_hat.detach() - log_var_z),
                dim=1,
            )

            kl_mu_x = torch.sum(
                difference_x * difference_x * mu_upsilon_hat.detach() * mu_z.detach(), dim=1
            )
            kl_sigma_x = (
                    torch.sum(
                        2 * torch.exp(log_var_x) * mu_upsilon_hat.detach() * mu_z.detach(),
                        dim=1,
                    )
                    - log_var_x
            )

            kl_mu_m = sigma_0 * mu_m * mu_m
            kl_sigma_m = sigma_0 * torch.exp(log_var_m) - log_var_m

            loss_y = torch.sum(kl_y) / N
            loss_mu_m = torch.sum(kl_mu_m) / N
            loss_sigma_m = torch.sum(kl_sigma_m) / N
            loss_mu_x = torch.sum(kl_mu_x) / N
            loss_sigma_x = torch.sum(kl_sigma_x) / N
            loss_mu_z = torch.sum(kl_mu_z) / N
            loss_sigma_z = torch.sum(kl_sigma_z) / N
            loss_Bayes = loss_y + loss_mu_m + loss_sigma_m + loss_mu_x + loss_sigma_x + loss_mu_z + loss_sigma_z

            #masks_pred = masks_pred[0]

            loss_dice = criterion(masks_pred, masks_cuda)
            # loss2 = criterion1(masks_pred, masks_cuda)
            # loss3 = criterion2(masks_pred, masks_cuda)
            # a = 0.6
            # b = 0.2
            # c = 0.2
            # loss = a * loss1 + b * loss2 + c *loss3
            loss = loss_dice + loss_Bayes * args.Bayes_weight

        masks_cuda_max = torch.max(masks_cuda)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # with torch.no_grad():
        predicted = masks_pred.argmax(1)
        # train_pa = Pa(predicted,masks_cuda)
        train_iou = calculate_miou(predicted,masks_cuda,2)
        # train_pa_whole += train_pa.item()
        train_iou_whole += train_iou.item()
        running_loss += loss.item()
        # epoch_acc = train_pa_whole/(batch_idx+1)
        epoch_iou = train_iou_whole/(batch_idx+1)
        # t.set_postfix(loss='{:.3f}'.format(running_loss / (batch_idx + 1)),
        #               train_pa='{:.2f}%'.format(epoch_acc*100),train_iou='{:.2f}%'.format(epoch_iou*100))
        # t.update(1)
    # epoch_acc = correct / total
    epoch_loss = running_loss / len(trainloader.dataset)
    # with tqdm(total=len(valloader), ncols=120, ascii=True) as t:
    val_running_loss = 0
    val_pa_whole = 0
    val_iou_whole = 0
    model.eval()
    with torch.no_grad():
        for batch_idx,(imgs, masks) in enumerate(valloader):
            # t.set_description("val(Epoch{}/{})".format(epoch, epochs))
            imgs, masks_cuda = imgs.to('cuda'), masks.to('cuda')
            imgs = imgs.float()
            masks_pred, gx, mu_x, log_var_x, gm, mu_m, log_var_m, gz, mu_z, log_var_z, = model(imgs)



            #masks_pred = masks_pred[0]



            predicted = masks_pred.argmax(1)
            # val_pa = Pa(predicted,masks_cuda)
            val_iou = calculate_miou(predicted,masks_cuda,2)
            # val_pa_whole += val_pa.item()
            val_iou_whole += val_iou.item()
            masks_cuda = masks_cuda.squeeze(1)

            loss1 = criterion(masks_pred, masks_cuda)
            # loss2 = criterion1(masks_pred, masks_cuda)
            # loss3 = criterion2(masks_pred, masks_cuda)
            # a = 0.6
            # b = 0.2
            # c = 0.2
            # loss = a * loss1 + b * loss2 + c *loss3
            loss = loss1

            # loss = criterion(masks_pred, masks_cuda)
            val_running_loss += loss.item()
            epoch_val_acc = val_pa_whole/(batch_idx+1)
            epoch_val_iou = val_iou_whole/(batch_idx+1)
            # t.set_postfix(loss='{:.3f}'.format(val_running_loss / (batch_idx + 1)),
            #               val_pa='{:.2f}%'.format(epoch_val_acc*100),val_iou='{:.2f}%'.format(epoch_val_iou*100))
            # t.update(1)
        # epoch_test_acc = test_correct / test_total
    epoch_val_loss = val_running_loss / len(valloader.dataset)
    #if epoch > 2:
    CosineLR.step()
    #if epoch > 2:
    # if epoch < 2:
    #     epoch_val_iou = epoch_val_iou - 0.2

    return epoch_loss, epoch_iou, epoch_val_loss, epoch_val_iou

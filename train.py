import os
import numpy as np
import random
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from dataset import create_folds, Dataset
from model import UNet
from loss import dice_and_ce_loss
from utils import resample_array, output2file
from metric import eval
from config import cfg

def initial_net(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv3d):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
            nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

def initialization():

    train_record = []
    for i in range(cfg['commu_times']):
        order = [(j + i) % len(cfg['node_list']) for j in range(len(cfg['node_list']))]
        random.shuffle(order)
        train_record.append(order)

    nodes = []
    val_fold = None
    test_fold = None
    weight_sum = 0
    for node_id, [node_name, d_name, d_path, fraction] in enumerate(cfg['node_list']):

        folds, _ = create_folds(d_name, d_path, node_name, fraction, exclude_case=cfg['exclude_case'])

        # create training fold
        train_fold = folds[0]
        d_train = Dataset(train_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=True)
        dl_train = data.DataLoader(dataset=d_train, batch_size=cfg['batch_size'], shuffle=True, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

        # create validaion fold
        if val_fold is None:
            val_fold = folds[1]
        else:
            val_fold.extend(folds[1])

        # create testing fold
        if test_fold is None:
            test_fold = folds[2]
        else:
            test_fold.extend(folds[2])

        print('{0:s}: train = {1:d}'.format(node_name, len(d_train)))
        weight_sum += len(d_train)

        local_model = nn.DataParallel(module=UNet(in_ch=1, base_ch=32, cls_num=cfg['cls_num']))
        local_model.cuda()
        initial_net(local_model)

        optimizer = optim.SGD(local_model.parameters(), lr=cfg['lr'], momentum=0.99, nesterov=True)
        
        lambda_func = lambda epoch: (1 - epoch / (cfg['commu_times'] * cfg['epoch_per_commu']))**0.9
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_func)

        nodes.append([local_model, optimizer, scheduler, node_name, len(d_train), dl_train])

    d_val = Dataset(val_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False)
    dl_val = data.DataLoader(dataset=d_val, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

    d_test = Dataset(test_fold, rs_size=cfg['rs_size'], rs_spacing=cfg['rs_spacing'], rs_intensity=cfg['rs_intensity'], label_map=cfg['label_map'], cls_num=cfg['cls_num'], aug_data=False)
    dl_test = data.DataLoader(dataset=d_test, batch_size=cfg['test_batch_size'], shuffle=False, pin_memory=True, drop_last=False, num_workers=cfg['cpu_thread'])

    print('{0:s}: val/test = {1:d}/{2:d}'.format(node_name, len(d_val), len(d_test)))

    for i in range(len(nodes)):
        nodes[i][4] = nodes[i][4] / weight_sum
        print('Weight of {0:s}: {1:f}'.format(nodes[i][3], nodes[i][4]))

    return nodes, train_record, dl_val, dl_test

def exchange_local_models(nodes, train_record, commu_t, node_id):
    new_node_id = train_record[commu_t][node_id]
    return new_node_id, nodes[new_node_id][3], nodes[new_node_id][4], nodes[new_node_id][5]

def train_local_model(local_model, optimizer, scheduler, data_loader, epoch_num):
    train_loss = 0
    train_loss_num = 0
    for epoch_id in range(epoch_num):

        t0 = time.perf_counter()

        epoch_loss = 0
        epoch_loss_num = 0
        batch_id = 0
        for batch in data_loader:
            image = batch['data'].cuda()
            label = batch['label'].cuda()

            N = len(image)

            pred, pred_logit = local_model(image)
           
            print_line = 'Epoch {0:d}/{1:d} (train) --- Progress {2:5.2f}% (+{3:02d})'.format(
                epoch_id+1, epoch_num, 100.0 * batch_id * cfg['batch_size'] / len(data_loader.dataset), N)

            l_ce, l_dice = dice_and_ce_loss(pred, pred_logit, label)
            loss_sup = l_dice + l_ce
            epoch_loss += l_dice.item() + l_ce.item()
            epoch_loss_num += 1

            print_line += ' --- Loss: {0:.6f}({1:.6f}/{2:.6f})'.format(loss_sup.item(), l_dice.item(), l_ce.item())
            print(print_line)
            
            optimizer.zero_grad()
            loss_sup.backward()
            optimizer.step()

            del image, label, pred, pred_logit, loss_sup
            batch_id += 1

        train_loss += epoch_loss
        train_loss_num += epoch_loss_num
        epoch_loss = epoch_loss / epoch_loss_num
        lr = scheduler.get_last_lr()[0]

        print_line = 'Epoch {0:d}/{1:d} (train) --- Loss: {2:.6f} --- Lr: {3:.6f}'.format(epoch_id+1, epoch_num, epoch_loss, lr)
        print(print_line)

        scheduler.step()

        t1 = time.perf_counter()
        epoch_t = t1 - t0
        print("Epoch time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
            h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

    train_loss = train_loss / train_loss_num

    return train_loss

# mode: 'val' or 'test'
# commu_iters: communication iteration index, only available when mode == 'val'
def eval_local_model(nodes, data_loader, result_path, mode, commu_iters):
    t0 = time.perf_counter()

    if mode == 'val':
        metric_fname = 'metric_validation-{0:04d}'.format(commu_iters+1)            
        print('Validation ({0:d}/{1:d}) ...'.format(commu_iters+1, cfg['commu_times']))
    elif mode == 'test':
        metric_fname = 'metric_testing'
        print('Testing ...')
    
    gt_entries = []
    output_buffer = None
    std_buffer = None
    for batch_id, batch in enumerate(data_loader):
        image = batch['data'].cuda()
        N = len(image)

        Ey, Ey2 = None, None
        for [local_model, _, _, _, _, _] in nodes:

            local_model.eval()
            prob = local_model(image)
            y = prob[:,1,:].detach().cpu().numpy().copy()

            if Ey is None:
                Ey = np.zeros_like(y)
                Ey2 = np.zeros_like(y)
            Ey = Ey + y
            Ey2 = Ey2 + y**2

            del prob

        Ey = Ey/len(nodes)
        Ey2 = Ey2/len(nodes)
        mask = (Ey.copy()>0.5).astype(dtype=np.uint8)

        stddev = Ey2-Ey**2
        stddev[stddev<0] = 0
        stddev = np.sqrt(stddev) + 1.0
        
        print_line = '{0:s} --- Progress {1:5.2f}% (+{2:d})'.format(
            mode, 100.0 * batch_id * cfg['test_batch_size'] / len(data_loader.dataset), N)
        print(print_line)

        for i in range(N):
            sample_mask = resample_array(
                    mask[i,:], batch['size'][i].numpy(), batch['spacing'][i].numpy(), batch['origin'][i].numpy(), 
                    batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy())
            if output_buffer is None:
                output_buffer = np.zeros_like(sample_mask, dtype=np.uint8)
            output_buffer[sample_mask > 0] = 0
            output_buffer = output_buffer + sample_mask

            if batch['eof'][i] == True:
                output2file(output_buffer, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                    '{0:s}/{1:s}@{2:s}.nii.gz'.format(result_path, batch['dataset'][i], batch['case'][i]))
                output_buffer = None
                gt_entries.append([batch['dataset'][i], batch['case'][i], batch['label_fname'][i]])

            if mode == 'test':
                sample_stddev = resample_array(
                        stddev[i,:], batch['size'][i].numpy(), batch['spacing'][i].numpy(), batch['origin'][i].numpy(), 
                        batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), linear=True)
                if std_buffer is None:
                    std_buffer = np.zeros_like(sample_stddev)
                std_buffer[sample_stddev >= 1.0] = 0
                sample_stddev[sample_stddev < 1.0] = 0
                sample_stddev = sample_stddev - 1.0
                sample_stddev[sample_stddev < 0.0] = 0
                std_buffer = std_buffer + sample_stddev

                if batch['eof'][i] == True:
                    output2file(std_buffer, batch['org_size'][i].numpy(), batch['org_spacing'][i].numpy(), batch['org_origin'][i].numpy(), 
                        '{0:s}/{1:s}@{2:s}-std.nii.gz'.format(result_path, batch['dataset'][i], batch['case'][i]))
                    std_buffer = None

        del image
        
    seg_dsc, seg_asd, seg_hd, seg_dsc_m, seg_asd_m, seg_hd_m = eval(
        pd_path=result_path, gt_entries=gt_entries, label_map=cfg['label_map'], cls_num=cfg['cls_num'], 
        metric_fn=metric_fname, calc_asd=(mode != 'val'), keep_largest=False)
    
    if mode == 'val':
        print_line = 'Validation result (iter = {0:d}/{1:d}) --- DSC {2:.2f} ({3:s})%'.format(
            commu_iters+1, cfg['commu_times'], 
            seg_dsc_m*100.0, '/'.join(['%.2f']*len(seg_dsc[:,0])) % tuple(seg_dsc[:,0]*100.0))
    else:
        print_line = 'Testing results --- DSC {0:.2f} ({1:s})% --- ASD {2:.2f} ({3:s})mm --- HD {4:.2f} ({5:s})mm'.format(
            seg_dsc_m*100.0, '/'.join(['%.2f']*len(seg_dsc[:,0])) % tuple(seg_dsc[:,0]*100.0), 
            seg_asd_m, '/'.join(['%.2f']*len(seg_asd[:,0])) % tuple(seg_asd[:,0]),
            seg_hd_m, '/'.join(['%.2f']*len(seg_hd[:,0])) % tuple(seg_hd[:,0]))
    print(print_line)
    t1 = time.perf_counter()
    eval_t = t1 - t0
    print("Evaluation time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
        h=int(eval_t) // 3600, m=(int(eval_t) % 3600) // 60, s=int(eval_t) % 60))

    return seg_dsc_m, seg_dsc

def load_models(nodes, model_fname):
    for node_id in range(len(nodes)):
        nodes[node_id][0].load_state_dict(torch.load(model_fname)['local_model_{0:d}_state_dict'.format(node_id)])
        nodes[node_id][1].load_state_dict(torch.load(model_fname)['local_model_{0:d}_optimizer'.format(node_id)])
        nodes[node_id][2].load_state_dict(torch.load(model_fname)['local_model_{0:d}_scheduler'.format(node_id)])

def train():

    train_start_time = time.localtime()
    print("Start time: {start_time}\n".format(start_time=time.strftime("%Y-%m-%d %H:%M:%S", train_start_time)))
    time_stamp = time.strftime("%Y%m%d%H%M%S", train_start_time)
    
    # create directory for results storage
    store_dir = '{}/model_{}'.format(cfg['model_path'], time_stamp)
    loss_fn = '{}/loss.txt'.format(store_dir)
    val_result_path = '{}/results_val'.format(store_dir)
    os.makedirs(val_result_path, exist_ok=True)
    test_result_path = '{}/results_test'.format(store_dir)
    os.makedirs(test_result_path, exist_ok=True)

    print('Loading local data from each nodes ... \n')

    nodes, train_record, dl_val, dl_test = initialization()

    print("Training order:", train_record)

    best_val_acc = 0
    start_iter = 0
    acc_time = 0
    best_model_fn = '{0:s}/cp_commu_{1:04d}.pth.tar'.format(store_dir, 1)

    print()
    log_line = "Model: {}\nModel parameters: {}\nStart time: {}\nConfiguration:\n".format(
        nodes[0][0].module.description(), 
        sum(x.numel() for x in nodes[0][0].parameters()), 
        time.strftime("%Y-%m-%d %H:%M:%S", train_start_time))
    for cfg_key in cfg:
        log_line += ' --- {}: {}\n'.format(cfg_key, cfg[cfg_key])
    print(log_line)

    for commu_t in range(start_iter, cfg['commu_times'], 1):
        
        t0 = time.perf_counter()

        train_loss = []
        for i, [local_model, optimizer, scheduler, _, _, _] in enumerate(nodes):
            
            node_id, node_name, node_weight, dl_train = exchange_local_models(nodes, train_record, commu_t, i)
            
            print('Training ({0:d}/{1:d}) on Node: {2:s}\n'.format(commu_t+1, cfg['commu_times'], node_name))

            local_model.train()

            train_loss.append(train_local_model(local_model, optimizer, scheduler, dl_train, cfg['epoch_per_commu']))

        seg_dsc_m, seg_dsc = eval_local_model(nodes, dl_val, val_result_path, mode='val', commu_iters=commu_t)

        t1 = time.perf_counter()
        epoch_t = t1 - t0
        acc_time += epoch_t
        print("Iteration time cost: {h:>02d}:{m:>02d}:{s:>02d}\n".format(
            h=int(epoch_t) // 3600, m=(int(epoch_t) % 3600) // 60, s=int(epoch_t) % 60))

        loss_line = '{commu_iter:>04d}\t{train_loss:s}\t{seg_val_dsc:>8.6f}\t{seg_val_dsc_cls:s}'.format(
            commu_iter=commu_t+1, train_loss='\t'.join(['%8.6f']*len(train_loss)) % tuple(train_loss), 
            seg_val_dsc=seg_dsc_m, seg_val_dsc_cls='\t'.join(['%8.6f']*len(seg_dsc[:,0])) % tuple(seg_dsc[:,0])
            )
        for [_, _, scheduler, _, _, _] in nodes:
            loss_line += '\t{node_lr:>8.6f}'.format(node_lr=scheduler.get_last_lr()[0])
        loss_line += '\n'

        with open(loss_fn, 'a') as loss_file:
            loss_file.write(loss_line)

        # save best model
        if commu_t == 0 or seg_dsc_m > best_val_acc:
            # remove former best model
            if os.path.exists(best_model_fn):
                os.remove(best_model_fn)
            # save current best model
            best_val_acc = seg_dsc_m
            best_model_fn = '{0:s}/cp_commu_{1:04d}.pth.tar'.format(store_dir, commu_t+1)
            best_model_cp = {
                        'commu_iter':commu_t,
                        'acc_time':acc_time,
                        'time_stamp':time_stamp,
                        'best_val_acc':best_val_acc,
                        'best_model_filename':best_model_fn}
            for node_id, [local_model, optimizer, scheduler, _, _, _] in enumerate(nodes):
                best_model_cp['local_model_{0:d}_state_dict'.format(node_id)] = local_model.state_dict()
                best_model_cp['local_model_{0:d}_optimizer'.format(node_id)] = optimizer.state_dict()
                best_model_cp['local_model_{0:d}_scheduler'.format(node_id)] = scheduler.state_dict()
            torch.save(best_model_cp, best_model_fn)
            print('Best model (communication iteration = {}) saved.\n'.format(commu_t+1))
    
    print("Total training time: {h:>02d}:{m:>02d}:{s:>02d}\n\n".format(
            h=int(acc_time) // 3600, m=(int(acc_time) % 3600) // 60, s=int(acc_time) % 60))

    # test
    load_models(nodes, best_model_fn)
    eval_local_model(nodes, dl_test, test_result_path, mode='test', commu_iters=0)

    print("Finish time: {finish_time}\n\n".format(
            finish_time=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())))

if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg['gpu']

    train()
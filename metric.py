import os
import numpy as np
import math
# Note:
# Use itk here will cause deadlock after the first training epoch 
# when using multithread (dataloader num_workers > 0) but reason unknown
import SimpleITK as sitk
from pandas import DataFrame, read_csv
from utils import read_image

def keep_largest_component(image, largest_n=1):
    c_filter = sitk.ConnectedComponentImageFilter()
    obj_arr = sitk.GetArrayFromImage(c_filter.Execute(image))
    obj_num = c_filter.GetObjectCount()
    tmp_arr = np.zeros_like(obj_arr)

    if obj_num > 0:
        obj_vol = np.zeros(obj_num, dtype=np.int64)
        for obj_id in range(obj_num):
            tmp_arr2 = np.zeros_like(obj_arr)
            tmp_arr2[obj_arr == obj_id+1] = 1
            obj_vol[obj_id] = np.sum(tmp_arr2)

        sorted_obj_id = np.argsort(obj_vol)[::-1]
    
        for i in range(min(largest_n, obj_num)):
            tmp_arr[obj_arr == sorted_obj_id[i]+1] = 1
            
    output = sitk.GetImageFromArray(tmp_arr)
    output.SetSpacing(image.GetSpacing())
    output.SetOrigin(image.GetOrigin())
    output.SetDirection(image.GetDirection())

    return output

def cal_dsc(pd, gt):
    y = (np.sum(pd * gt) * 2 + 1) / (np.sum(pd * pd + gt * gt) + 1)
    return y

def cal_asd(a, b):
    filter1 = sitk.SignedMaurerDistanceMapImageFilter()
    filter1.SetUseImageSpacing(True)
    filter1.SetSquaredDistance(False)
    a_dist = filter1.Execute(a)

    a_dist = sitk.GetArrayFromImage(a_dist)
    a_dist = np.abs(a_dist)
    a_edge = np.zeros(a_dist.shape, a_dist.dtype)
    a_edge[a_dist == 0] = 1
    a_num = np.sum(a_edge)

    filter2 = sitk.SignedMaurerDistanceMapImageFilter()
    filter2.SetUseImageSpacing(True)
    filter2.SetSquaredDistance(False)
    b_dist = filter2.Execute(b)

    b_dist = sitk.GetArrayFromImage(b_dist)
    b_dist = np.abs(b_dist)
    b_edge = np.zeros(b_dist.shape, b_dist.dtype)
    b_edge[b_dist == 0] = 1
    b_num = np.sum(b_edge)

    a_dist[b_edge == 0] = 0.0
    b_dist[a_edge == 0] = 0.0

    #a2b_mean_dist = np.sum(b_dist) / a_num
    #b2a_mean_dist = np.sum(a_dist) / b_num
    asd = (np.sum(a_dist) + np.sum(b_dist)) / (a_num + b_num)

    return asd

def cal_hd(a, b):
    filter1 = sitk.HausdorffDistanceImageFilter()
    filter1.Execute(a, b)
    hd = filter1.GetHausdorffDistance()

    return hd

def eval(pd_path, gt_entries, label_map, cls_num, metric_fn, calc_asd=True, keep_largest=False):
    result_lines = ''
    df_fn = '{}/{}.csv'.format(pd_path, metric_fn)
    if not os.path.exists(df_fn):
        results = []
        print_line = '\n --- Start calculating metrics --- '
        print(print_line)
        result_lines += '{}\n'.format(print_line)
        for [d_name, casename, gt_fname] in gt_entries:
            gt_label = read_image(fname=gt_fname)
            gt_array = sitk.GetArrayFromImage(gt_label)
            gt_array = gt_array.astype(dtype=np.uint8)

            # map labels
            tmp_array = np.zeros_like(gt_array)
            lmap = label_map[d_name]
            tgt_labels = []
            for key in lmap:
                tmp_array[gt_array == key] = lmap[key]
                if lmap[key] not in tgt_labels:
                    tgt_labels.append(lmap[key])
            gt_array = tmp_array

            pd_fname = '{}/{}@{}.nii.gz'.format(pd_path, d_name, casename)
            pd_array = sitk.GetArrayFromImage(read_image(fname=pd_fname))

            for c in tgt_labels:
                pd = np.zeros_like(pd_array, dtype=np.uint8)
                pd[pd_array == c] = 1
                pd_im = sitk.GetImageFromArray(pd)
                pd_im.SetSpacing(gt_label.GetSpacing())
                pd_im.SetOrigin(gt_label.GetOrigin())
                pd_im.SetDirection(gt_label.GetDirection())
                if keep_largest:
                    pd_im = keep_largest_component(pd_im, largest_n=1)
                pd = sitk.GetArrayFromImage(pd_im)
                pd = pd.astype(dtype=np.uint8)
                pd = np.reshape(pd, -1)

                gt = np.zeros_like(gt_array, dtype=np.uint8)
                gt[gt_array == c] = 1
                gt_im = sitk.GetImageFromArray(gt)
                gt_im.SetSpacing(gt_label.GetSpacing())
                gt_im.SetOrigin(gt_label.GetOrigin())
                gt_im.SetDirection(gt_label.GetDirection())
                gt = np.reshape(gt, -1)

                dsc = cal_dsc(pd, gt)
                if calc_asd and np.sum(pd) > 0:
                    asd = cal_asd(pd_im, gt_im)
                    hd = cal_hd(pd_im, gt_im)
                else:
                    asd = 0
                    hd = 0
                results.append([d_name, casename, c, dsc, asd, hd])

                print_line = ' --- {0:22s}@{1:22s}@{2:d}:\t\tDSC = {3:>5.2f}%\tASD = {4:>5.2f}mm\tHD = {5:>5.2f}mm'.format(d_name, casename, c, dsc*100.0, asd, hd)
                print(print_line)
                result_lines += '{}\n'.format(print_line)
        
        df = DataFrame(results, columns=['Dataset', 'Case', 'Class', 'DSC', 'ASD', 'HD'])
        df.to_csv(df_fn)
    df = read_csv(df_fn)


    dsc = []
    asd = []
    hd = []
    dsc_mean = 0
    asd_mean = 0
    hd_mean = 0
    d_list = [d for d in set(df['Dataset'].tolist())]
    for d in d_list:
        dsc_m = df[df['Dataset'] == d]['DSC'].mean()
        dsc_v = df[df['Dataset'] == d]['DSC'].std()
        dsc.append([dsc_m, dsc_v])        
        dsc_mean += dsc_m
        asd_m = df[df['Dataset'] == d]['ASD'].mean()
        asd_v = df[df['Dataset'] == d]['ASD'].std()
        asd.append([asd_m, asd_v])
        asd_mean += asd_m
        hd_m = df[df['Dataset'] == d]['HD'].mean()
        hd_v = df[df['Dataset'] == d]['HD'].std()
        hd.append([hd_m, hd_v])
        hd_mean += hd_m
        print_line = ' --- dataset {0:s}:\tDSC = {1:.2f}({2:.2f})%\tASD = {3:.2f}({4:.2f})mm\tHD = {5:.2f}({6:.2f})mm\tN={7:d}'.format(d, dsc_m*100.0, dsc_v*100.0, asd_m, asd_v, hd_m, hd_v, len(df[df['Dataset'] == d]['DSC']))
        print(print_line)
        result_lines += '{}\n'.format(print_line)
    dsc_mean = dsc_mean / len(d_list)
    asd_mean = asd_mean / len(d_list)
    hd_mean = hd_mean / len(d_list)
    dsc = np.array(dsc)
    asd = np.array(asd)
    hd = np.array(hd)

    print_line = ' --- dataset-avg:\tDSC = {0:.2f}%\tASD = {1:.2f}mm\tHD = {2:.2f}mm'.format(dsc_mean*100.0, asd_mean, hd_mean)
    print(print_line)
    result_lines += '{}\n'.format(print_line)
    print_line = ' --- Finish calculating metrics --- \n'
    print(print_line)
    result_lines += '{}\n'.format(print_line)

    result_fn = '{}/{}.txt'.format(pd_path, metric_fn)
    with open(result_fn, 'w') as result_file:
        result_file.write(result_lines)
    
    return dsc, asd, hd, dsc_mean, asd_mean, hd_mean

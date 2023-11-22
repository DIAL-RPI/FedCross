import os
import sys
import torch
from torch.utils import data
import itk
import numpy as np
import random
import SimpleITK as sitk
from scipy import stats

def read_image(fname, imtype):
    reader = itk.ImageFileReader[imtype].New()
    reader.SetFileName(fname)
    reader.Update()
    image = reader.GetOutput()
    return image

def image_2_array(image):
    arr = itk.GetArrayFromImage(image)
    return arr

def array_2_image(arr, spacing, origin, imtype):
    image = itk.GetImageFromArray(arr)
    image.SetSpacing((spacing[0], spacing[1], spacing[2]))
    image.SetOrigin((origin[0], origin[1], origin[2]))
    cast = itk.CastImageFilter[type(image), imtype].New()
    cast.SetInput(image)
    cast.Update()
    image = cast.GetOutput()
    return image

def scan_path(dataset_name, dataset_path):
    entries = []
    if dataset_name == 'MSD':
        for f in os.listdir('{}/imagesTr'.format(dataset_path)):
            if f.startswith('prostate_') and f.endswith('.nii.gz'):
                case_name = f.split('.nii.gz')[0]
                if os.path.isfile('{}/labelsTr/{}'.format(dataset_path, f)):
                    image_name = '{}/imagesTr/{}'.format(dataset_path, f)
                    label_name = '{}/labelsTr/{}'.format(dataset_path, f)
                    entries.append([dataset_name, case_name, image_name, label_name])
    elif dataset_name == 'NCI-ISBI':
        for f in os.listdir('{}/image'.format(dataset_path)):
            if f.startswith('Prostate') and f.endswith('.nii.gz'):
                case_name = f.split('.nii.gz')[0]
                if os.path.isfile('{}/label/{}'.format(dataset_path, f)):
                    image_name = '{}/image/{}'.format(dataset_path, f)
                    label_name = '{}/label/{}'.format(dataset_path, f)
                    entries.append([dataset_name, case_name, image_name, label_name])
    elif dataset_name == 'PROMISE12':
        for f in os.listdir('{}/image'.format(dataset_path)):
            if f.startswith('Case') and f.endswith('.nii.gz'):
                case_name = f.split('.nii.gz')[0]
                if os.path.isfile('{}/label/{}'.format(dataset_path, f)):
                    image_name = '{}/image/{}'.format(dataset_path, f)
                    label_name = '{}/label/{}'.format(dataset_path, f)
                    entries.append([dataset_name, case_name, image_name, label_name])
    elif dataset_name == 'PROSTATEx':
        for f in os.listdir('{}/image'.format(dataset_path)):
            if f.startswith('ProstateX-') and f.endswith('.nii.gz'):
                case_name = f.split('.nii.gz')[0]
                if os.path.isfile('{}/label/{}'.format(dataset_path, f)):
                    image_name = '{}/image/{}'.format(dataset_path, f)
                    label_name = '{}/label/{}'.format(dataset_path, f)
                    entries.append([dataset_name, case_name, image_name, label_name])
    return entries

def create_folds(dataset_name, dataset_path, fold_name, fraction, exclude_case):
    fold_file_name = '{0:s}/data_split-{1:s}.txt'.format(sys.path[0], fold_name)
    folds = {}
    if os.path.exists(fold_file_name):
        with open(fold_file_name, 'r') as fold_file:
            strlines = fold_file.readlines()
            for strline in strlines:
                strline = strline.rstrip('\n')
                params = strline.split()
                fold_id = int(params[0])
                if fold_id not in folds:
                    folds[fold_id] = []
                folds[fold_id].append([params[1], params[2], params[3], params[4]])
    else:
        entries = []
        for [d_name, d_path] in zip(dataset_name, dataset_path):            
            entries.extend(scan_path(d_name, d_path))
        for e in entries:
            if e[0:2] in exclude_case:
                entries.remove(e)
        random.shuffle(entries)
       
        ptr = 0
        for fold_id in range(len(fraction)):
            folds[fold_id] = entries[ptr:ptr+fraction[fold_id]]
            ptr += fraction[fold_id]

        with open(fold_file_name, 'w') as fold_file:
            for fold_id in range(len(fraction)):
                for [d_name, case_name, image_path, label_path] in folds[fold_id]:
                    fold_file.write('{0:d} {1:s} {2:s} {3:s} {4:s}\n'.format(fold_id, d_name, case_name, image_path, label_path))

    folds_size = [len(x) for x in folds.values()]

    return folds, folds_size

def generate_transform(aug, min_offset, max_offset):
    if aug:
        min_rotate = -0.1 # [rad]
        max_rotate = 0.1 # [rad]
        t = itk.Euler3DTransform[itk.D].New()
        euler_parameters = t.GetParameters()
        euler_parameters = itk.OptimizerParameters[itk.D](t.GetNumberOfParameters())
        offset_x = min_offset[0] + random.random() * (max_offset[0] - min_offset[0]) # rotate
        offset_y = min_offset[1] + random.random() * (max_offset[1] - min_offset[1]) # rotate
        offset_z = min_offset[2] + random.random() * (max_offset[2] - min_offset[2]) # rotate
        rotate_x = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        rotate_y = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        rotate_z = min_rotate + random.random() * (max_rotate - min_rotate) # tranlate
        euler_parameters[0] = rotate_x # rotate
        euler_parameters[1] = rotate_y # rotate
        euler_parameters[2] = rotate_z # rotate
        euler_parameters[3] = offset_x # tranlate
        euler_parameters[4] = offset_y # tranlate
        euler_parameters[5] = offset_z # tranlate
        t.SetParameters(euler_parameters)
    else:
        offset_x = 0
        offset_y = 0
        offset_z = 0
        rotate_x = 0
        rotate_y = 0
        rotate_z = 0
        t = itk.IdentityTransform[itk.D, 3].New()
    return t, [offset_x, offset_y, offset_z, rotate_x, rotate_y, rotate_z]

def resample(image, imtype, size, spacing, origin, transform, linear, dtype):
    o_origin = image.GetOrigin()
    o_spacing = image.GetSpacing()
    o_size = image.GetBufferedRegion().GetSize()
    output = {}
    output['org_size'] = np.array(o_size, dtype=int)
    output['org_spacing'] = np.array(o_spacing, dtype=float)
    output['org_origin'] = np.array(o_origin, dtype=float)
    
    if origin is None: # if no origin point specified, center align the resampled image with the original image
        new_size = np.zeros(3, dtype=int)
        new_spacing = np.zeros(3, dtype=float)
        new_origin = np.zeros(3, dtype=float)
        for i in range(3):
            new_size[i] = size[i]
            if spacing[i] > 0:
                new_spacing[i] = spacing[i]
                new_origin[i] = o_origin[i] + o_size[i]*o_spacing[i]*0.5 - size[i]*spacing[i]*0.5
            else:
                new_spacing[i] = o_size[i] * o_spacing[i] / size[i]
                new_origin[i] = o_origin[i]
    else:
        new_size = np.array(size, dtype=int)
        new_spacing = np.array(spacing, dtype=float)
        new_origin = np.array(origin, dtype=float)

    output['size'] = new_size
    output['spacing'] = new_spacing
    output['origin'] = new_origin

    resampler = itk.ResampleImageFilter[imtype, imtype].New()
    resampler.SetInput(image)
    resampler.SetSize((int(new_size[0]), int(new_size[1]), int(new_size[2])))
    resampler.SetOutputSpacing((float(new_spacing[0]), float(new_spacing[1]), float(new_spacing[2])))
    resampler.SetOutputOrigin((float(new_origin[0]), float(new_origin[1]), float(new_origin[2])))
    resampler.SetTransform(transform)
    if linear:
        resampler.SetInterpolator(itk.LinearInterpolateImageFunction[imtype, itk.D].New())
    else:
        resampler.SetInterpolator(itk.NearestNeighborInterpolateImageFunction[imtype, itk.D].New())
    resampler.SetDefaultPixelValue(0)
    resampler.Update()
    rs_image = resampler.GetOutput()
    image_array = itk.GetArrayFromImage(rs_image)
    image_array = image_array[np.newaxis, :].astype(dtype)
    output['array'] = image_array

    return output

def zscore_normalize(x):
    y = (x - x.mean()) / x.std()
    return y

def make_onehot(input, cls):
    oh = np.repeat(np.zeros_like(input), cls+1, axis=0)
    for i in range(cls+1):
        tmp = np.zeros_like(input)
        tmp[input==i] = 1        
        oh[i,:] = tmp
    return oh

def make_flag(cls, labelmap):
    flag = np.zeros([cls, 1])
    for key in labelmap:
        flag[labelmap[key]-1,0] = 1
    return flag

def image2file(image, imtype, fname):
    writer = itk.ImageFileWriter[imtype].New()
    writer.SetInput(image)
    writer.SetFileName(fname)
    writer.Update()

def array2file(array, size, origin, spacing, imtype, fname):    
    image = itk.GetImageFromArray(array.reshape([size[2], size[1], size[0]]))
    image.SetSpacing((spacing[0], spacing[1], spacing[2]))
    image.SetOrigin((origin[0], origin[1], origin[2]))
    image2file(image, imtype=imtype, fname=fname)

# dataset of 3D image volume
# 3D volumes are resampled from and center-aligned with the original images
class Dataset(data.Dataset):
    def __init__(self, ids, rs_size, rs_spacing, rs_intensity, label_map, cls_num, aug_data):
        self.ImageType = itk.Image[itk.F, 3]
        self.LabelType = itk.Image[itk.UC, 3]
        self.ids = []
        self.rs_size = np.array(rs_size)
        self.rs_spacing = np.array(rs_spacing, dtype=np.float)
        self.rs_intensity = rs_intensity
        self.label_map = label_map
        self.cls_num = cls_num
        self.aug_data = aug_data
        self.im_cache = {}
        self.lb_cache = {}

        for [d_name, casename, image_fn, label_fn] in ids:
            reader = sitk.ImageFileReader()
            reader.SetFileName(image_fn)
            reader.ReadImageInformation()
            image_size = np.array(reader.GetSize()[:3])
            image_origin = np.array(reader.GetOrigin()[:3], dtype=np.float)
            image_spacing = np.array(reader.GetSpacing()[:3], dtype=np.float)
            image_phy_size = image_size * image_spacing
            patch_phy_size = self.rs_size*self.rs_spacing

            if not aug_data:
                patch_num = (image_phy_size/patch_phy_size).astype(int)+1
                for p_z in range(patch_num[2]):
                    for p_y in range(patch_num[1]):
                        for p_x in range(patch_num[0]):
                            patch_origin = np.zeros_like(image_origin)
                            patch_origin[0] = image_origin[0]+p_x*patch_phy_size[0]
                            patch_origin[1] = image_origin[1]+p_y*patch_phy_size[1]
                            patch_origin[2] = image_origin[2]+p_z*patch_phy_size[2]
                            patch_origin[0] = min(patch_origin[0], image_origin[0]+image_phy_size[0]-patch_phy_size[0])
                            patch_origin[1] = min(patch_origin[1], image_origin[1]+image_phy_size[1]-patch_phy_size[1])
                            patch_origin[2] = min(patch_origin[2], image_origin[2]+image_phy_size[2]-patch_phy_size[2])
                            eof = (p_x == patch_num[0]-1) & (p_y == patch_num[1]-1) & (p_z == patch_num[2]-1)
                            self.ids.append([d_name, casename, image_fn, label_fn, patch_origin, eof])
            else:
                patch_origin = np.zeros_like(image_origin)
                patch_origin = image_origin+0.5*image_phy_size-0.5*patch_phy_size
                repeat_time = 4
                for i in range(repeat_time):
                    self.ids.append([d_name, casename, image_fn, label_fn, patch_origin, i==repeat_time-1])

    
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        [d_name, casename, image_fn, label_fn, patch_origin, eof] = self.ids[index]
        
        if image_fn not in self.im_cache:
            src_image = read_image(fname=image_fn, imtype=self.ImageType)
            image_cache = {}
            image_cache['size'] = np.array(src_image.GetBufferedRegion().GetSize())
            image_cache['origin'] = np.array(src_image.GetOrigin(), dtype=np.float)
            image_cache['spacing'] = np.array(src_image.GetSpacing(), dtype=np.float)
            image_cache['array'] = zscore_normalize(image_2_array(src_image).copy())
            self.im_cache[image_fn] = image_cache
        image_cache = self.im_cache[image_fn]

        min_offset = image_cache['origin'] - patch_origin
        max_offset = image_cache['origin']+image_cache['spacing']*image_cache['size']-patch_origin-self.rs_size*self.rs_spacing

        t, _ = generate_transform(self.aug_data, min_offset, max_offset)

        src_image = array_2_image(image_cache['array'], image_cache['spacing'], image_cache['origin'], self.ImageType)

        image = resample(
                    image=src_image, imtype=self.ImageType, 
                    size=self.rs_size, spacing=self.rs_spacing, origin=patch_origin, 
                    transform=t, linear=True, dtype=np.float32)
        
        if label_fn not in self.lb_cache:
            src_label = read_image(fname=label_fn, imtype=self.LabelType)
            label_cache = {}
            label_cache['origin'] = np.array(src_label.GetOrigin(), dtype=np.float)
            label_cache['spacing'] = np.array(src_label.GetSpacing(), dtype=np.float)
            label_cache['array'] = image_2_array(src_label).copy()
            self.lb_cache[label_fn] = label_cache
        label_cache = self.lb_cache[label_fn]
        src_label = array_2_image(label_cache['array'], label_cache['spacing'], label_cache['origin'], self.LabelType)

        label = resample(
                    image=src_label, imtype=self.LabelType, 
                    size=self.rs_size, spacing=self.rs_spacing, origin=patch_origin, 
                    transform=t, linear=False, dtype=np.int64)

        tmp_array = np.zeros_like(label['array'])
        lmap = self.label_map[d_name]
        for key in lmap:
            tmp_array[label['array'] == key] = lmap[key]
        label['array'] = tmp_array
        #label_bin = make_onehot(label['array'], cls=self.cls_num)
        label_exist = make_flag(cls=self.cls_num, labelmap=self.label_map[d_name])

        image_tensor = torch.from_numpy(image['array'])
        label_tensor = torch.from_numpy(label['array'])

        output = {}
        output['data'] = image_tensor
        output['label'] = label_tensor
        output['label_exist'] = label_exist
        output['dataset'] = d_name
        output['case'] = casename
        output['label_fname'] = label_fn
        output['size'] = image['size']
        output['spacing'] = image['spacing']
        output['origin'] = image['origin']
        output['org_size'] = image['org_size']
        output['org_spacing'] = image['org_spacing']
        output['org_origin'] = image['org_origin']
        output['eof'] = eof

        return output
import numpy as np
# Note:
# Use itk here will cause deadlock after the first training epoch 
# when using multithread (dataloader num_workers > 0) but reason unknown
import SimpleITK as sitk
    
def read_image(fname):
    reader = sitk.ImageFileReader()
    reader.SetFileName(fname)
    image = reader.Execute()
    return image

def resample_array(array, size, spacing, origin, size_rs, spacing_rs, origin_rs, transform=None, linear=False):
    array = np.reshape(array, [size[2], size[1], size[0]])
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize((int(size_rs[0]), int(size_rs[1]), int(size_rs[2])))
    resampler.SetOutputSpacing((float(spacing_rs[0]), float(spacing_rs[1]), float(spacing_rs[2])))
    resampler.SetOutputOrigin((float(origin_rs[0]), float(origin_rs[1]), float(origin_rs[2])))
    if transform is not None:
        resampler.SetTransform(transform)
    else:
        resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    if linear:
        resampler.SetInterpolator(sitk.sitkLinear)
    else:
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    resampler.SetDefaultPixelValue(0)
    rs_image = resampler.Execute(image)
    rs_array = sitk.GetArrayFromImage(rs_image)

    return rs_array

def output2file(array, size, spacing, origin, fname):
    array = np.reshape(array, [size[2], size[1], size[0]])#.astype(dtype=np.uint8)
    image = sitk.GetImageFromArray(array)
    image.SetSpacing((float(spacing[0]), float(spacing[1]), float(spacing[2])))
    image.SetOrigin((float(origin[0]), float(origin[1]), float(origin[2])))

    writer = sitk.ImageFileWriter()
    writer.SetFileName(fname)
    writer.Execute(image)
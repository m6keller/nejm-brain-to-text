import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tsaug import TimeWarp, AddNoise, Convolve, Drift, Pool, Quantize
from tsaug.visualization import plot
from scipy.interpolate import CubicSpline
from scipy.stats import special_ortho_group

def gauss_smooth(inputs, device, smooth_kernel_std=2, smooth_kernel_size=100,  padding='same'):
    """
    Applies a 1D Gaussian smoothing operation with PyTorch to smooth the data along the time axis.
    Args:
        inputs (tensor : B x T x N): A 3D tensor with batch size B, time steps T, and number of features N.
                                     Assumed to already be on the correct device (e.g., GPU).
        kernelSD (float): Standard deviation of the Gaussian smoothing kernel.
        padding (str): Padding mode, either 'same' or 'valid'.
        device (str): Device to use for computation (e.g., 'cuda' or 'cpu').
    Returns:
        smoothed (tensor : B x T x N): A smoothed 3D tensor with batch size B, time steps T, and number of features N.
    """
    # Get Gaussian kernel
    inp = np.zeros(smooth_kernel_size, dtype=np.float32)
    inp[smooth_kernel_size // 2] = 1
    gaussKernel = gaussian_filter1d(inp, smooth_kernel_std)
    validIdx = np.argwhere(gaussKernel > 0.01)
    gaussKernel = gaussKernel[validIdx]
    gaussKernel = np.squeeze(gaussKernel / np.sum(gaussKernel))

    # Convert to tensor
    gaussKernel = torch.tensor(gaussKernel, dtype=torch.float32, device=device)
    gaussKernel = gaussKernel.view(1, 1, -1)  # [1, 1, kernel_size]

    # Prepare convolution
    B, T, C = inputs.shape
    inputs = inputs.permute(0, 2, 1)  # [B, C, T]
    gaussKernel = gaussKernel.repeat(C, 1, 1)  # [C, 1, kernel_size]

    # Perform convolution
    smoothed = F.conv1d(inputs, gaussKernel, padding=padding, groups=C)
    return smoothed.permute(0, 2, 1)  # [B, T, C]

def time_warping(inputs, max_warp=5):
    warp_mode = {
        0: 'stretch',
        1: 'compress'
    }

    operation_mode = np.random.choice([0,1])
    operation = warp_mode[operation_mode]

    warped = inputs.clone()
    if operation == 'stretch' or operation =='compress':
        # add random number of time points
        n_added = np.random.randint(1,max_warp)
        n_index = np.random.randint(0,warped.shape[1]-1)

        val_prev = warped[:,n_index,:]
        val_next = warped[:,n_index+1,:]
        slope = (val_next-val_prev)/(n_added+1)
        add = [(val_prev+(i+1)*slope) for i in range(n_added)]
        add = torch.stack(add, dim=1)

        warped = torch.cat([warped[:,:n_index,:], add, warped[:,n_index:,:]], dim=1)

    return warped

def permute(inputs, window_size=10):
    inputs = np.array(inputs)

    num_windows = np.floor(len(inputs[:,0,:])/window_size)

    windows = [inputs[:,(i*window_size):((i+1)*window_size),:] for i in range(int(num_windows))]

    np.random.shuffle(windows)
    
    permuted = np.concatenate(windows, axis=1)
    permuted = torch.tensor(permuted)

    return permuted

def homogeneous_scaling(inputs, scale_loc, scale_std):
    batch_size = inputs.shape[0]
    rng = np.random.default_rng()
    
    alpha = rng.normal(loc=scale_loc, scale=scale_std, size=(batch_size, 1, 1))
    
    scaled = inputs * alpha
    
    return scaled

def magnitude_warping(inputs, std, n_knots):
    batch_size, time_steps, n_features = inputs.shape
    
    warped = np.zeros_like(inputs)
    
    for b in range(batch_size):
        # knot locations
        warp_steps = np.linspace(0, time_steps - 1, num=n_knots + 2)
        
        # get values for each knot form normal distribution
        random_warps = np.random.normal(loc=1.0, scale=std, size=(n_knots + 2,))

        warper = CubicSpline(warp_steps, random_warps)
        
        orig_steps = np.arange(time_steps)
        warp_curve = warper(orig_steps)
        
        warp_curve = warp_curve[:, np.newaxis]
        
        warped[b] = inputs[b] * warp_curve
    
    return warped.to(inputs.device)

def rotation(inputs, magnitude):
    orig_shape = inputs.shape
    batch_size, time_steps, n_features = inputs.shape

    rotated = np.zeros_like(inputs)
    
    for b in range(batch_size):
        # random rotation matrix
        rotation_matrix = special_ortho_group.rvs(n_features)
        rotated[b] = inputs[b] @ rotation_matrix
    
    return rotated.to(inputs.device)

def augment(self, inputs, n_time_steps):
    speed_changes           = self.transform_args['speed_changes']
    noise_mean              = self.transform_args['noise_mean']
    noise_std               = self.transform_args['noise_std']
    conv_size               = self.transform_args['conv_size']
    drift_pts               = self.transform_args['drift_pts']
    pool_size               = self.transform_args['pool_size']
    quant_levels            = self.transform_args['quant_levels']
    quant_method            = self.transform_args['quant_method']
    homogeneous_scaling_loc = self.transform_args['homogeneous_scaling_loc']
    homogeneous_scaling_std = self.transform_args['homogeneous_scaling_std']
    mag_warp_loc            = self.transform_args['mag_warp_loc']
    n_knots                 = self.transform_args['n_knots']

    augmenter = (
        TimeWarp(n_speed_change=speed_changes)
        # + Crop(size=crop_length)
        + AddNoise(loc=noise_mean, scale=noise_std) # jittering
        + Convolve(window='hann', size=conv_size)
        + Drift(n_drift_points=drift_pts)
        # + Dropout(prob=dropout_prob)
        + Pool(kind='ave', size=pool_size)
        + Quantize(n_levels=quant_levels, how=quant_method)
        # + Resize(size=resize_size)
    )

    inputs_np = inputs.cpu().numpy()

    assert np.isfinite(inputs_np).all(), f"Input has NaN/Inf: min={inputs_np.min()}, max={inputs_np.max()}"

    # augmentation pipeline
    augmented = augmenter.augment(inputs_np)
    if not np.isfinite(augmented).all():
        print(f"Invalid after tsaug augmenter")
        print(f"min={augmented.min()}, max={augmented.max()}")
        augmented = np.nan_to_num(augmented, nan=0.0, posinf=0.0, neginf=0.0)
    
    # homogeneous scaling
    if homogeneous_scaling_std > 0:
        inputs_np = homogeneous_scaling(inputs_np, homogeneous_scaling_loc, homogeneous_scaling_std)
        if not np.isfinite(inputs_np).all():
            print(f"Invalid after homogeneous_scaling")
            print(f"min={inputs_np.min()}, max={inputs_np.max()}")
            # for debugging
            inputs_np = np.nan_to_num(inputs_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # magnitude warping
    if mag_warp_loc > 0:
        inputs_np = magnitude_warping(inputs_np, mag_warp_loc, n_knots)
        if not np.isfinite(inputs_np).all():
            print(f"Invalid after magnitude_warping")
            print(f"min={inputs_np.min()}, max={inputs_np.max()}")
            inputs_np = np.nan_to_num(inputs_np, nan=0.0, posinf=0.0, neginf=0.0)
    
    # rotation
        inputs_np = rotation(inputs_np, magnitude=1)
        if not np.isfinite(inputs_np).all():
            print(f"Invalid after rotation")
            print(f"   min={inputs_np.min()}, max={inputs_np.max()}")
            inputs_np = np.nan_to_num(inputs_np, nan=0.0, posinf=0.0, neginf=0.0)

    # check dimensions
    new_length = augmented.shape[1]
    batch_size = augmented.shape[0]
    min_length = self.model.patch_size
    
    final_length = max(new_length, min_length)
    
    new_n_time_steps = torch.full(
        (batch_size,), 
        final_length, 
        dtype=torch.int32
    )
    
    # convert to tensor
    augmented = torch.from_numpy(augmented).float().to(inputs.device)
    
    return augmented, new_n_time_steps



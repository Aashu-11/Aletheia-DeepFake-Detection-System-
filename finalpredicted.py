import argparse
import os
from tabulate import tabulate
from data_utils.face_detection import *
from deep_fake_detect.utils import *
from deep_fake_detect.DeepFakeDetectModel import *
import torchvision
from data_utils.datasets import *
import warnings
import multiprocessing
import sys
import numpy as np
import cv2
from PIL import Image
import torch.nn.functional as F
import torch
import shutil
from scipy import fftpack
from skimage.feature import local_binary_pattern

_visible_dep_warn = getattr(np, "VisibleDeprecationWarning", DeprecationWarning)
warnings.filterwarnings("ignore", category=_visible_dep_warn)


def _select_sample_paths(paths, max_samples):
    if len(paths) <= max_samples:
        return paths
    indices = np.linspace(0, len(paths) - 1, max_samples, dtype=int)
    return [paths[i] for i in indices]


def _get_all_conv_layers(module):
    """Get all convolutional layers for multi-layer Grad-CAM"""
    conv_layers = []
    for name, m in module.named_modules():
        if isinstance(m, torch.nn.Conv2d):
            conv_layers.append((name, m))
    return conv_layers


def _improved_grad_cam(model, input_tensor, smooth_samples=15, noise_sigma=0.08):
    """
    Enhanced Grad-CAM with:
    - More smoothing samples
    - Lower noise for better stability
    - Multi-layer averaging
    - Better normalization
    """
    model.eval()
    conv_layers = _get_all_conv_layers(model.encoder)
    
    if not conv_layers:
        raise RuntimeError("No Conv2d layers found for Grad-CAM.")
    
    # Use last 3 conv layers for better feature representation
    target_layers = conv_layers[-3:] if len(conv_layers) >= 3 else conv_layers
    
    all_layer_cams = []
    
    for layer_name, target_layer in target_layers:
        cams = []
        for _ in range(smooth_samples):
            activations = []
            gradients = []

            def fwd_hook(_, __, out):
                activations.append(out)

            def bwd_hook(_, __, grad_out):
                gradients.append(grad_out[0])

            handle_fwd = target_layer.register_forward_hook(fwd_hook)
            handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

            noise = torch.randn_like(input_tensor) * noise_sigma
            noisy_input = (input_tensor + noise).requires_grad_(True)
            output = model(noisy_input)
            score = torch.sigmoid(output)[0, 0]
            model.zero_grad()
            score.backward()

            handle_fwd.remove()
            handle_bwd.remove()

            if not activations or not gradients:
                continue

            act = activations[0]
            grad = gradients[0]
            
            # Use ReLU on gradients for better localization
            grad = F.relu(grad)
            weights = grad.mean(dim=(2, 3), keepdim=True)
            cam = (weights * act).sum(dim=1)
            cam = F.relu(cam)
            cam = cam.squeeze(0)
            
            # Better normalization
            cam_min = cam.min()
            cam_max = cam.max()
            if cam_max > cam_min:
                cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
            
            cams.append(cam.detach().cpu().numpy())

        if cams:
            layer_cam = np.mean(np.stack(cams, axis=0), axis=0)
            all_layer_cams.append(layer_cam)
    
    if not all_layer_cams:
        raise RuntimeError("Failed to compute Grad-CAM.")
    
    # Average across multiple layers
    final_cam = np.mean(np.stack(all_layer_cams, axis=0), axis=0)
    
    # Apply guided smoothing
    final_cam = cv2.GaussianBlur(final_cam, (3, 3), 0)
    
    return final_cam


def _save_enhanced_cam_overlay(image_path, cam, out_path, alpha=0.5):
    """Enhanced heatmap visualization with better contrast"""
    image = cv2.imread(image_path)
    if image is None:
        return
    
    h, w = image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))
    
    # Use adaptive normalization
    low = np.percentile(cam_resized, 5)
    high = np.percentile(cam_resized, 98)
    cam_resized = (cam_resized - low) / (high - low + 1e-8)
    cam_resized = np.clip(cam_resized, 0, 1)
    
    # Apply gamma correction for better visibility
    gamma = 0.8
    cam_resized = np.power(cam_resized, gamma)
    
    cam_resized = np.uint8(255 * cam_resized)
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    
    # Enhanced blending
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    
    cv2.imwrite(out_path, overlay)


def analyze_frequency_patterns(image_path):
    """
    Analyze frequency domain patterns to detect deepfakes
    Returns frequency domain features
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Apply FFT
    f_transform = fftpack.fft2(image)
    f_shift = fftpack.fftshift(f_transform)
    magnitude_spectrum = np.abs(f_shift)
    
    # Analyze high-frequency components (common in deepfakes)
    h, w = magnitude_spectrum.shape
    center_h, center_w = h // 2, w // 2
    
    # Extract high-frequency energy (outer regions)
    mask_high = np.ones((h, w))
    mask_high[center_h - h//4:center_h + h//4, center_w - w//4:center_w + w//4] = 0
    high_freq_energy = np.sum(magnitude_spectrum * mask_high) / np.sum(magnitude_spectrum)
    
    # Extract low-frequency energy (center region)
    mask_low = np.zeros((h, w))
    mask_low[center_h - h//8:center_h + h//8, center_w - w//8:center_w + w//8] = 1
    low_freq_energy = np.sum(magnitude_spectrum * mask_low) / np.sum(magnitude_spectrum)
    
    return {
        'high_freq_ratio': float(high_freq_energy),
        'low_freq_ratio': float(low_freq_energy),
        'freq_balance': float(high_freq_energy / (low_freq_energy + 1e-8))
    }


def analyze_texture_patterns(image_path):
    """
    Analyze texture patterns using Local Binary Patterns
    Deepfakes often have inconsistent textures
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Compute LBP
    radius = 3
    n_points = 8 * radius
    lbp = local_binary_pattern(image, n_points, radius, method='uniform')
    
    # Calculate histogram
    hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-8)
    
    # Calculate texture uniformity (low uniformity = potential fake)
    texture_uniformity = np.sum(hist ** 2)
    
    return {
        'texture_uniformity': float(texture_uniformity),
        'texture_entropy': float(-np.sum(hist * np.log(hist + 1e-8)))
    }


def test_time_augmentation(model, image_path, test_transform, device, n_augments=8):
    """
    Test-Time Augmentation for more robust predictions
    """
    image = Image.open(image_path).convert("RGB")
    predictions = []
    
    # Original image
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.sigmoid(output)[0, 0].item()
        predictions.append(prob)
    
    # Augmented versions
    augmentations = [
        torchvision.transforms.RandomHorizontalFlip(p=1.0),
        torchvision.transforms.ColorJitter(brightness=0.1),
        torchvision.transforms.ColorJitter(contrast=0.1),
        torchvision.transforms.ColorJitter(saturation=0.1),
        torchvision.transforms.RandomRotation(5),
    ]
    
    for aug in augmentations[:n_augments-1]:
        aug_transform = torchvision.transforms.Compose([
            aug,
            test_transform
        ])
        try:
            input_tensor = aug_transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                prob = torch.sigmoid(output)[0, 0].item()
                predictions.append(prob)
        except:
            continue
    
    # Return mean and std for confidence estimation
    return np.mean(predictions), np.std(predictions)


def predict_deepfake_enhanced(input_videofile, df_method='plain_frames', debug=False, 
                              verbose=False, return_details=False, use_tta=True):
    """
    Enhanced deepfake prediction with:
    - Test-Time Augmentation
    - Frequency domain analysis
    - Texture pattern analysis
    - Improved Grad-CAM
    - Adaptive thresholding
    """
    num_workers = multiprocessing.cpu_count() - 2
    model_params = dict()
    model_params['batch_size'] = 32
    model_params['imsize'] = 224
    model_params['encoder_name'] = 'tf_efficientnet_b0_ns'

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    vid = os.path.basename(input_videofile)[:-4]
    output_path = os.path.join("output", vid)
    plain_faces_data_path = os.path.join(output_path, "plain_frames")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(plain_faces_data_path, exist_ok=True)

    if verbose:
        print(f'Extracting faces from the video')
    extract_landmarks_from_video(input_videofile, output_path, overwrite=True)
    crop_faces_from_video(input_videofile, output_path, plain_faces_data_path, overwrite=True)

    model_path = 'final.chkpt'
    frames_path = plain_faces_data_path

    if verbose:
        print(f'Loading model and detecting DeepFakes...')
    model = DeepFakeDetectModel(frame_dim=model_params['imsize'], 
                               encoder_name=model_params['encoder_name'])
    check_point_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(check_point_dict['model_state_dict'])

    model = model.to(device)
    model.eval()

    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((model_params['imsize'], model_params['imsize'])),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]),
    ])

    data_path = os.path.join(frames_path, vid)
    test_dataset = SimpleImageFolder(root=data_path, transforms_=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=model_params['batch_size'], 
                           num_workers=num_workers, pin_memory=True)
    
    if len(test_loader) == 0:
        print('Cannot extract images. Dataloaders empty')
        if return_details:
            return None, None, None, None
        return None, None, None

    probabilities = []
    all_filenames = []
    tta_stds = []  # Track prediction uncertainty
    freq_features = []
    texture_features = []
    
    with torch.no_grad():
        for batch_id, samples in enumerate(test_loader):
            frames = samples[0].to(device)
            output = model(frames)
            class_probability = get_probability(output).to('cpu').detach().numpy()
            
            if len(class_probability) > 1:
                probabilities.extend(class_probability.squeeze())
                all_filenames.extend(samples[1])
            else:
                probabilities.append(class_probability.squeeze())
                all_filenames.append(samples[1])
    
    # Enhanced analysis with TTA and frequency/texture analysis
    if verbose:
        print('Performing enhanced analysis...')
    
    enhanced_probs = []
    for idx, (filepath, prob) in enumerate(zip(all_filenames, probabilities)):
        if use_tta and idx % 5 == 0:  # Apply TTA to every 5th frame for speed
            tta_prob, tta_std = test_time_augmentation(model, filepath, test_transform, device)
            enhanced_probs.append(tta_prob)
            tta_stds.append(tta_std)
        else:
            enhanced_probs.append(prob)
            tta_stds.append(0.0)
        
        # Analyze frequency and texture patterns
        freq_feat = analyze_frequency_patterns(filepath)
        text_feat = analyze_texture_patterns(filepath)
        
        if freq_feat:
            freq_features.append(freq_feat)
        if text_feat:
            texture_features.append(text_feat)
    
    probabilities = np.array(enhanced_probs)
    total_number_frames = len(probabilities)
    
    # Adaptive thresholding based on distribution
    prob_median = np.median(probabilities)
    prob_std = np.std(probabilities)
    
    # Adjust threshold based on data distribution
    prob_threshold_fake = max(0.5, min(0.7, prob_median + 0.5 * prob_std))
    
    # Calculate frequency-based adjustment
    if freq_features:
        avg_freq_balance = np.mean([f['freq_balance'] for f in freq_features])
        # High frequency imbalance suggests manipulation
        if avg_freq_balance > 1.5:
            prob_threshold_fake *= 0.95  # More sensitive
    
    # Calculate texture-based adjustment
    if texture_features:
        avg_texture_uniformity = np.mean([t['texture_uniformity'] for t in texture_features])
        # Low texture uniformity suggests manipulation
        if avg_texture_uniformity < 0.1:
            prob_threshold_fake *= 0.95  # More sensitive
    
    fake_frames_high_prob = probabilities[probabilities >= prob_threshold_fake]
    number_fake_frames = len(fake_frames_high_prob)
    
    if number_fake_frames == 0:
        fake_prob = 0
    else:
        fake_prob = round(sum(fake_frames_high_prob) / number_fake_frames, 4)

    real_frames_high_prob = probabilities[probabilities < prob_threshold_fake]
    number_real_frames = len(real_frames_high_prob)
    
    if number_real_frames == 0:
        real_prob = 0
    else:
        real_prob = 1 - round(sum(real_frames_high_prob) / number_real_frames, 4)

    # Adaptive fake_fraction based on confidence
    avg_uncertainty = np.mean(tta_stds) if tta_stds else 0
    fake_fraction = 0.4 if avg_uncertainty > 0.1 else 0.5

    pred = pred_strategy(number_fake_frames, number_real_frames, total_number_frames,
                        fake_fraction=fake_fraction)

    if debug:
        print(f'Adaptive threshold: {prob_threshold_fake:.4f}')
        print(f'Avg uncertainty: {avg_uncertainty:.4f}')
        print(f'Fake fraction: {fake_fraction}')
        print(f'Fake frames: {number_fake_frames}, Real frames: {number_real_frames}')
        print(f'Fake prob: {fake_prob:.4f}, Real prob: {real_prob:.4f}')

    if return_details:
        grad_cam_dir = os.path.join(output_path, "grad_cam_enhanced")
        if os.path.isdir(grad_cam_dir):
            shutil.rmtree(grad_cam_dir)
        os.makedirs(grad_cam_dir, exist_ok=True)
        
        # Generate improved heatmaps for top frames
        frame_info = list(zip(all_filenames, probabilities.tolist()))
        frame_info.sort(key=lambda x: x[1], reverse=True)
        sample_info = frame_info[:30]  # Analyze more frames
        
        if verbose:
            print('Generating enhanced Grad-CAM visualizations...')
        
        with torch.enable_grad():
            for image_path, prob in sample_info:
                try:
                    image = Image.open(image_path).convert("RGB")
                    input_tensor = test_transform(image).unsqueeze(0).to(device)
                    cam = _improved_grad_cam(model, input_tensor, smooth_samples=15, 
                                            noise_sigma=0.08)
                    out_path = os.path.join(grad_cam_dir, os.path.basename(image_path))
                    _save_enhanced_cam_overlay(image_path, cam, out_path, alpha=0.5)
                except Exception as e:
                    if debug:
                        print(f'Failed to generate CAM for {image_path}: {e}')

        details = {
            "frames_dir": data_path,
            "grad_cam_dir": grad_cam_dir,
            "probabilities": probabilities.tolist(),
            "filenames": [os.path.basename(p) for p in all_filenames],
            "grad_cam_frames": [
                {"file": os.path.basename(p), "prob": float(prob)}
                for p, prob in sample_info
            ],
            "adaptive_threshold": float(prob_threshold_fake),
            "fake_fraction": float(fake_fraction),
            "avg_uncertainty": float(avg_uncertainty),
            "freq_analysis": {
                "avg_freq_balance": float(np.mean([f['freq_balance'] for f in freq_features])) if freq_features else None,
                "avg_high_freq": float(np.mean([f['high_freq_ratio'] for f in freq_features])) if freq_features else None,
            },
            "texture_analysis": {
                "avg_uniformity": float(np.mean([t['texture_uniformity'] for t in texture_features])) if texture_features else None,
                "avg_entropy": float(np.mean([t['texture_entropy'] for t in texture_features])) if texture_features else None,
            }
        }
        return fake_prob, real_prob, pred, details

    return fake_prob, real_prob, pred


def individual_test_enhanced():
    print_line()
    debug = True
    verbose = True
    
    result = predict_deepfake_enhanced(
        args.input_videofile, 
        args.method, 
        debug=debug, 
        verbose=verbose,
        return_details=True,
        use_tta=True
    )
    
    if result[0] is None:
        print_red('Failed to detect DeepFakes')
        return

    fake_prob, real_prob, pred, details = result
    label = "REAL" if pred == 0 else "DEEP-FAKE"
    probability = real_prob if pred == 0 else fake_prob
    probability = round(probability * 100, 4)
    
    print_line()
    print(f'Enhanced Analysis Results:')
    print(f'  Adaptive Threshold: {details["adaptive_threshold"]:.4f}')
    print(f'  Fake Fraction Used: {details["fake_fraction"]:.2f}')
    print(f'  Prediction Uncertainty: {details["avg_uncertainty"]:.4f}')
    
    if details["freq_analysis"]["avg_freq_balance"]:
        print(f'  Frequency Balance: {details["freq_analysis"]["avg_freq_balance"]:.4f}')
    if details["texture_analysis"]["avg_uniformity"]:
        print(f'  Texture Uniformity: {details["texture_analysis"]["avg_uniformity"]:.4f}')
    
    print_line()
    
    if pred == 0:
        print_green(f'The video is {label}, confidence={probability}%')
    else:
        print_red(f'The video is {label}, confidence={probability}%')
    
    print_line()
    print(f'Enhanced heatmaps saved to: {details["grad_cam_dir"]}')
    print_line()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Enhanced DeepFakes Detection with Improved Accuracy')
    parser.add_argument('--input_videofile', action='store', help='Input video file')
    parser.add_argument('--method', action='store', choices=['plain_frames'],
                       default='plain_frames', help='Method type')
    parser.add_argument('--use-tta', action='store_true', default=True,
                       help='Use Test-Time Augmentation')
    
    args = parser.parse_args()
    
    if args.input_videofile is not None:
        if os.path.isfile(args.input_videofile):
            individual_test_enhanced()
        else:
            print(f'Input file not found ({args.input_videofile})')
    else:
        parser.print_help(sys.stderr)
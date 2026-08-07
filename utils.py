import os
import numpy as np
import torch
import torch.nn as nn
import json
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
import cv2
import logging
import torch.nn.functional as F



# 1. Configure Logging
def setup_logging(log_dir, log_filename="training.log"):
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, log_filename)

    logging.basicConfig(
        filename=log_path,
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )
    logger = logging.getLogger()
    return logger

# 2. Checkpoint Handling: Save and Load Checkpoints
def save_checkpoint(model, optimizer, epoch, best_mae, fold, checkpoint_dir='checkpoint', filename='checkpoint.pth'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_mae': best_mae,
        'fold': fold
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        best_mae = checkpoint['best_mae']
        fold = checkpoint['fold']
        print(f"Checkpoint loaded from {checkpoint_path}")
        return model, optimizer, epoch, best_mae, fold
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting from scratch.")
        return model, optimizer, 0, float('inf'), 1

# 3. Data Synchronization and DataLoader
def get_video_frame_rate(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return frame_rate

def resample(input_signal, target_length):
    input_signal = np.asarray(input_signal)
    return np.asarray(np.interp(np.linspace(1, input_signal.shape[0], target_length), np.linspace(1, input_signal.shape[0], input_signal.shape[0]), input_signal))

def extract_synchronized_spo2(json_path, number_of_frame):
    with open(json_path, 'r') as f:
        data = json.load(f)
    spo2_values = [item['Value']['o2saturation'] for item in data['/FullPackage']]
    return resample(spo2_values, number_of_frame)

def butter_lowpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return b, a

def extract_dc_ac_components(spatio_temporal_map, fs=15):
    b_lp, a_lp = butter_lowpass(cutoff=0.3, fs=fs, order=5)
    dc_component = filtfilt(b_lp, a_lp, spatio_temporal_map, axis=2)
    b_bp, a_bp = butter_bandpass(lowcut=0.75, highcut=2.5, fs=fs, order=5)
    ac_component = filtfilt(b_bp, a_bp, spatio_temporal_map, axis=2)
    return dc_component, ac_component


import torch.nn as nn
import torch.nn.functional as F

class TemporalFilter(nn.Module):
    def __init__(self, fs=15, low_cutoff=0.3, high_cutoff=(0.75, 2.5), order=5):
        super(TemporalFilter, self).__init__()
        self.fs = fs
        self.low_cutoff = low_cutoff
        self.high_cutoff = high_cutoff
        self.order = order

        # Define low-pass filter kernel
        b_lp, _ = butter_lowpass(low_cutoff, fs, order)
        self.register_buffer('b_lp', torch.tensor(b_lp, dtype=torch.float32).view(1, 1, -1))  # Convert to 1D convolution kernel and register as buffer

        # Define band-pass filter kernel
        b_bp, _ = butter_bandpass(high_cutoff[0], high_cutoff[1], fs, order)
        self.register_buffer('b_bp', torch.tensor(b_bp, dtype=torch.float32).view(1, 1, -1))  # Convert to 1D convolution kernel and register as buffer

    def forward(self, x_ac, x_dc):
        """
        Apply low-pass and band-pass filters to the input tensors.
        Args:
        - x_ac: Input tensor for AC filtering of shape [batch_size, time_steps]
        - x_dc: Input tensor for DC filtering of shape [batch_size, time_steps]
        Returns:
        - dc_component: Low-pass filtered signal (DC component)
        - ac_component: Band-pass filtered signal (AC component)
        """
        # Ensure the input tensors are in the correct shape
        x_ac = x_ac.unsqueeze(1)  # Shape: [batch_size, 1, time_steps]
        x_dc = x_dc.unsqueeze(1)

        # Move filters to the same device as the input tensors
        b_lp = self.b_lp.to(x_dc.device)
        b_bp = self.b_bp.to(x_ac.device)

        # Calculate padding to ensure the output size matches the input size
        padding_lp = (b_lp.shape[-1] - 1) // 2
        padding_bp = (b_bp.shape[-1] - 1) // 2

        # Apply low-pass filter
        dc_component = F.conv1d(x_dc, b_lp, padding=padding_lp)
        dc_component = dc_component.squeeze(1)  # Remove the channel dimension, back to [batch_size, time_steps]

        # Apply band-pass filter
        ac_component = F.conv1d(x_ac, b_bp, padding=padding_bp)
        ac_component = ac_component.squeeze(1)  # Remove the channel dimension, back to [batch_size, time_steps]

        # Ensure output size matches the input size
        if dc_component.size(1) != x_dc.size(2):
            dc_component = F.interpolate(dc_component.unsqueeze(1), size=(x_dc.size(2),), mode='linear', align_corners=False).squeeze(1)
        
        if ac_component.size(1) != x_ac.size(2):
            ac_component = F.interpolate(ac_component.unsqueeze(1), size=(x_ac.size(2),), mode='linear', align_corners=False).squeeze(1)

        return dc_component, ac_component
    
# 7. Plotting functions
def plot_spo2_values_all(predicted, original, epoch, save_path, filename_prefix="epoch", frame_rate=15):
    os.makedirs(save_path, exist_ok=True)
    plt.figure(figsize=(10, 5))
    time = np.arange(len(predicted)) / frame_rate
    plt.plot(time, predicted, label='Predicted SpO₂')
    plt.plot(time, original, label='Original SpO₂')
    plt.xlabel('Time (s)')
    plt.ylabel('SpO₂')
    plt.title(f'SpO₂ Prediction vs Original - Epoch {epoch}')
    plt.legend()
    plot_filename = f"{filename_prefix}_spo2_epoch_{epoch}.png"
    plt.savefig(os.path.join(save_path, plot_filename))
    plt.close()

def plot_spo2_values(predicted, original, epoch, save_path, sample_idx, filename_prefix="epoch", frame_rate=15):
    plt.figure(figsize=(10, 5))
    time = np.arange(len(predicted)) / frame_rate
    plt.plot(time, predicted, label='Predicted SpO₂')
    plt.plot(time, original, label='Original SpO₂')
    plt.xlabel('Time (s)')
    plt.ylabel('SpO₂')
    plt.title(f'SpO₂ Prediction vs Original - Epoch {epoch}, Sample {sample_idx}')
    plt.legend()
    epoch_dir = os.path.join(save_path, f"epoch_{epoch}")
    os.makedirs(epoch_dir, exist_ok=True)
    plot_filename = f"{filename_prefix}_sample_{sample_idx}.png"
    plt.savefig(os.path.join(epoch_dir, plot_filename))
    plt.close()

# 8. Training Function with Checkpoint Resuming
def train_model(train_dataloader, eval_dataloader, model, criterion, optimizer, num_epochs=50, device='cpu', plot_save_path='plots', model_save_path='models', logger=None, checkpoint_dir=None, resume=False, fold=1,fps=None):
    model.to(device)
    os.makedirs(plot_save_path, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    temporal_filter = TemporalFilter(fs=30, low_cutoff=0.3, high_cutoff=(0.75, 2.5), order=5)

    start_epoch, best_mae = 0, float('inf')
    if resume:
        checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold}_checkpoint.pth')
        model, optimizer, start_epoch, best_mae, _ = load_checkpoint(model, optimizer, checkpoint_path)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        for st_map, dc_true, ac_true, labels in train_dataloader:
            st_map,dc_true, ac_true, labels = st_map.to(device), dc_true.to(device), ac_true.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs,dc_output, ac_output = model(st_map)
            loss = criterion(outputs, labels, dc_output, dc_true, ac_output, ac_true)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_dataloader)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.3f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.3f}")

        predicted, original, epoch_mae = evaluate_model(eval_dataloader, model, device=device, epoch=epoch + 1, plot_save_path=plot_save_path)
        logger.info(f"Epoch [{epoch + 1}/{num_epochs}], Test MAE: {epoch_mae:.3f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Test MAE: {epoch_mae:.3f}")

        epoch_model_path = os.path.join(model_save_path, f"model_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), epoch_model_path)
        logger.info(f"Model weights saved: {epoch_model_path}")

        if epoch_mae < best_mae:
            best_mae = epoch_mae
            best_model_path = os.path.join(model_save_path, "best_model.pth")
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Best model weights updated: {best_model_path} with MAE: {best_mae:.3f}")
            print(f"Best model weights updated: {best_model_path} with MAE: {best_mae:.3f}")
            plot_spo2_values_all(predicted, original, epoch + 1, plot_save_path+'/best')
        plot_spo2_values_all(predicted, original, epoch + 1, plot_save_path,frame_rate=fps)
        checkpoint_filename = f'fold_{fold}_checkpoint.pth'
        save_checkpoint(model, optimizer, epoch, best_mae, fold, checkpoint_dir, filename=checkpoint_filename)
        logger.info(f"Checkpoint saved for epoch {epoch + 1}")
        print(f"Best Test MAE (Mean Absolute Error): {best_mae:.3f}")
        logger.info(f"Best Test MAE (Mean Absolute Error): {best_mae:.3f}")

    return best_mae

# 9. Evaluation function
def evaluate_model(dataloader, model, device='cpu', epoch=1, plot_save_path='plots'):
    model.eval()
    model.to(device)
    total_samples = 0
    total_loss = 0.0
    predicted_values = []
    original_values = []
    
    with torch.no_grad():
        for idx, (st_map, dc_input, ac_input, labels) in enumerate(dataloader):
            st_map, dc_input, ac_input, labels = st_map.to(device), dc_input.to(device), ac_input.to(device), labels.to(device)
            
            # Correctly calculate the number of chunks based on the temporal size (dim=3)
            num_of_frame = 300
            num_chunks = dc_input.size(3) // num_of_frame
            chunk_predictions = []
            chunk_labels = []
            
            for i in range(num_chunks):
                start_idx = i * num_of_frame
                end_idx = start_idx + num_of_frame
                
                dc_chunk = dc_input[:, :, :, start_idx:end_idx]
                ac_chunk = ac_input[:, :, :, start_idx:end_idx]
                st_chunk = st_map[:, :, :, start_idx:end_idx]
                label_chunk = labels[:, start_idx:end_idx]
                
                outputs, dc_output, ac_output = model(st_chunk)
                
                loss = nn.MSELoss()(outputs, label_chunk)
                total_loss += loss.item() * outputs.size(0)  # Multiply by the batch size to accumulate correctly
                total_samples += outputs.size(0)
                
                chunk_predictions.extend(outputs.cpu().numpy().flatten())
                chunk_labels.extend(label_chunk.cpu().numpy().flatten())
            
            predicted_values.extend(chunk_predictions)
            original_values.extend(chunk_labels)
            
            # Plot each sample separately
            # plot_spo2_values(chunk_predictions, chunk_labels, epoch, plot_save_path, sample_idx=idx)
    
    # Calculate Mean Absolute Error (MAE)
    mae = total_loss / total_samples
    print(f"Test MAE (Mean Absolute Error): {mae:.3f}")
    
    return predicted_values, original_values, mae


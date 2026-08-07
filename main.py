import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from model1 import SpO2Model
from Loss import Combined_loss, SpO2Loss
from dataset import train_dataset, test_dataset
from utils import *


# 10. K-Fold Cross-Validation with Resuming Checkpoints
def train_model_kfold(map_files, map_dir, n_splits=10, num_epochs=50, batch_size=20, device='cpu', base_plot_save_path='plots', base_model_save_path='models', resume=False, checkpoint_dir=None,fps=None):
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    logger = setup_logging(base_model_save_path)
    map_files = np.array(map_files)
    fold_mae_scores, fold = [], 1

    # Determine the starting fold if resuming
    start_fold = 1
    if resume:
        # Check if a checkpoint exists for the last fold
        for fold_idx in range(1, n_splits + 1):
            checkpoint_path = os.path.join(checkpoint_dir, f'fold_{fold_idx}_checkpoint.pth')
            if os.path.exists(checkpoint_path):
                start_fold = fold_idx
            else:
                break  # Resume from this fold onwards

    for fold in range(start_fold, n_splits + 1):
        print(f"Starting Fold {fold}/{n_splits}")
        train_indices, test_indices = list(kfold.split(map_files))[fold-1]
        train_map_files, test_map_files = map_files[train_indices], map_files[test_indices]

        plot_save_path = os.path.join(base_plot_save_path, f'fold_{fold}')
        model_save_path = os.path.join(base_model_save_path, f'fold_{fold}')

        train_data = train_dataset(train_map_files, map_dir)
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
        eval_data = test_dataset(test_map_files, map_dir)
        eval_dataloader = DataLoader(eval_data, batch_size=1, shuffle=False, num_workers=2)

        model = SpO2Model(input_channel=12)
        # model = SpO2ModelWithIntermediateFusion()

        criterion = SpO2Loss() #Combined_loss()
        # criterion_2 = CombinedMotionArtifactLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        best_mae = train_model(train_dataloader, eval_dataloader, model, criterion, optimizer, num_epochs=num_epochs, device=device, plot_save_path=plot_save_path, model_save_path=model_save_path, logger=logger, checkpoint_dir=checkpoint_dir, resume=resume, fold=fold,fps=fps)
        fold_mae_scores.append(best_mae)

    average_mae, std_mae = np.mean(fold_mae_scores), np.std(fold_mae_scores)
    print(f"Average MAE across all folds: {average_mae:.2f}, Standard Deviation: {std_mae:.2f}")
    logger.info(f"Average MAE across all folds: {average_mae:.2f}, Standard Deviation: {std_mae:.2f}")

# 11. Main script execution
if __name__ == "__main__":
    
    
    map_dir = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/SPO2_code_v2-main/VIPL-HR'
    base_plot_save_path = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/results/VIPL/VIPL_Plots_1'
    base_model_save_path = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/results/VIPL/VIPL_weight_1'
    checkpoint_dir = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/results/VIPL/VIPL_checkpoints_1'
    fps = 15   
    

    #map_dir = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/SPO2_code_v2-main/PURE/PURE_dataset'
    #base_plot_save_path = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/SPO2_code_v2-main/results/PURE_Plots_eval_new_1'
    #base_model_save_path = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/SPO2_code_v2-main/results/PURE_weight_new_1'
    #checkpoint_dir = '/home/tih_isi_9/Downloads/SPO2_code_v2-main/SPO2_code_v2-main/results/PURE_checkpoints_new_1'
    #fps = 30   
    #torch.backends.cuda.enable_mem_efficient_sdp(False)
    #torch.backends.cuda.enable_flash_sdp(False)
    #torch.backends.cuda.enable_math_sdp(True)


    # map_dir = '/media/sda1_acces/Code_SPO2/SPO2_work_new/Datasets/BH_dataset_small'
    # base_plot_save_path = '/media/sda1_acces/Code_SPO2/SPO2_work_new/BH_Plots_eval'
    # base_model_save_path = '/media/sda1_acces/Code_SPO2/SPO2_work_new/BH_weight'
    # checkpoint_dir = '/media/sda1_acces/Code_SPO2/SPO2_work_new/BH_checkpoints'
    # fps = 15   

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    map_files = sorted([f for f in os.listdir(map_dir) if f.endswith('.npz')])

    train_model_kfold(map_files, map_dir, n_splits=10, num_epochs=50, batch_size=20, device=device, base_plot_save_path=base_plot_save_path, base_model_save_path=base_model_save_path, resume=False, checkpoint_dir=checkpoint_dir,fps=fps)

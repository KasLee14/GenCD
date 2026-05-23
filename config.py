# config.py
import torch


class Config:
    # ==========================================
    # Dataset Config
    # ==========================================
    dataset_adr = "data/ASSIST2017/"
    use_time_encoding = True
    dataset_has_timestamp = True

    # ==========================================
    # Training Hyperparameters
    # ==========================================
    batch_size = 128
    epochs = 50
    lr = 1e-3
    early_stop_patience = 5
    output_file = "gencd_result.txt"
    model_path = "GenCD.snapshot"
    best_model_path = "best_GenCD_model.pt"
    gauc_min_interactions_exclusive = 100

    # ==========================================
    # Model Architecture Config
    # ==========================================
    max_seq_len = 150
    embedding_size = 256
    num_layers = 3
    num_block = 1
    dropout_ratio = 0.20

    temp = 0.2
    eps = 0.2
    lambda_cl = 0.05
    lambda_dmr = 0.05
    mask_rate = 0.2

    # ==========================================
    # System Config
    # ==========================================
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

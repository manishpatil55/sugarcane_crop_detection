"""
sequence_models.py
==================
BiLSTM with Temporal Attention for Sugarcane Detection in Uttar Pradesh.

Architecture
------------
Input: (batch, seq_len=14, input_dim=25)
  - 14 monthly composites (6 before + anchor + 7 after)
  - 25 features per timestep:
      S2 bands (10): B2, B3, B4, B5, B6, B7, B8, B8A, B11, B12
      S2 indices (10): NDVI, EVI, NDWI, LSWI, NDRE, GNDVI, SAVI, MSAVI, NBR, NDMI
      S1 bands (2): VV, VH
      SAR indices (3): RVI, RFDI, CR

Encoder: 2-layer BiLSTM (64 hidden units each direction)
Attention: Temporal attention mechanism that learns to weight specific months
           (e.g., Nov/Dec harvest vs. Jun-Sep grand growth)
Classifier: Linear(128→32) → ReLU → Dropout → Linear(32→1) → Sigmoid
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SugarcaneAttentionLSTM(nn.Module):
    """
    Bidirectional LSTM with Temporal Attention for Sugarcane Detection.

    Takes a sequence of (timesteps, features) where each timestep is a monthly
    Sentinel-1/2 composite with spectral indices.

    The temporal attention mechanism learns to focus on the most discriminative
    months for sugarcane identification — typically:
      - Nov/Dec: sugarcane maintains high NDVI while rice-wheat fields are bare
      - Jun-Sep (monsoon): SAR features dominate due to cloud cover
      - Mar-Apr: grand growth phase with peak biomass

    Parameters
    ----------
    input_dim  : number of features per timestep (default 25)
    hidden_dim : LSTM hidden dimension per direction (default 64)
    num_layers : number of stacked LSTM layers (default 2)
    dropout    : dropout probability for regularization (default 0.3)
    """

    def __init__(self, input_dim=25, hidden_dim=64, num_layers=2, dropout=0.3):
        super(SugarcaneAttentionLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # BiLSTM Encoder — captures forward/backward temporal dynamics
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # Temporal Attention Mechanism
        # Learns month-specific weights: which months are most informative
        # for separating sugarcane from confuser crops (rice, wheat, maize)
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)
            Monthly composite feature sequences.

        Returns
        -------
        torch.Tensor of shape (batch_size, 1)
            Sugarcane probability per pixel.
        """
        # lstm_out: (batch_size, seq_len, hidden_dim * 2)
        lstm_out, _ = self.lstm(x)

        # Calculate attention weights
        # attn_weights: (batch_size, seq_len, 1)
        attn_weights = self.attention(lstm_out)
        attn_weights = F.softmax(attn_weights, dim=1)

        # Apply attention to LSTM outputs
        # context_vector: (batch_size, hidden_dim * 2)
        context_vector = torch.sum(attn_weights * lstm_out, dim=1)

        # Final classification
        # out: (batch_size, 1)
        out = self.classifier(context_vector)
        return out

    def get_attention_weights(self, x):
        """
        Return attention weights for interpretability.

        Useful for understanding which months the model focuses on for
        sugarcane classification. High weights on Nov/Dec suggest the model
        relies on the harvest-vs-standing contrast.

        Parameters
        ----------
        x : torch.Tensor of shape (batch_size, seq_len, input_dim)

        Returns
        -------
        torch.Tensor of shape (batch_size, seq_len)
            Per-month attention weights (sum to 1 over seq_len).
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x)
            attn_weights = self.attention(lstm_out)
            attn_weights = F.softmax(attn_weights, dim=1)
        return attn_weights.squeeze(-1)


# ---------------------------------------------------------
# PyTorch Dataset for Temporal Data
# ---------------------------------------------------------
from torch.utils.data import Dataset


class SugarcaneTemporalDataset(Dataset):
    """
    Dataset that reshapes a wide-format DataFrame into sequences of
    shape (time_steps, features) suitable for the BiLSTM model.

    Parameters
    ----------
    dataframe    : pd.DataFrame with columns like B2_2025_03, NDVI_2025_03, etc.
    feature_bands: list of band/index prefixes to include (e.g., ["B2", "NDVI", ...])
    label_col    : column name for the binary label
    seq_len      : expected sequence length (number of monthly composites)
    """

    def __init__(self, dataframe, feature_bands=None, label_col="label", seq_len=14):
        import pandas as pd

        self.data = dataframe
        self.seq_len = seq_len
        self.label_col = label_col

        # Default feature bands for sugarcane detection (25 features)
        if feature_bands is None:
            feature_bands = [
                "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12",
                "NDVI", "EVI", "NDWI", "LSWI", "NDRE", "GNDVI", "SAVI", "MSAVI", "NBR", "NDMI",
                "VV", "VH",
                "RVI", "RFDI", "CR",
            ]
        self.feature_bands = feature_bands

        self.sequences, self.labels = self._build_sequences()

    def _build_sequences(self):
        """
        Reshape wide-format DataFrame into (N, T, F) numpy array.

        Detects time tags from column names (BAND_YYYY_MM pattern) and
        extracts feature values for each timestep.

        Returns
        -------
        sequences : np.ndarray of shape (n_samples, seq_len, n_features)
        labels    : np.ndarray of shape (n_samples,)
        """
        import pandas as pd

        # Discover time tags from column names
        tags = set()
        for col in self.data.columns:
            parts = col.split("_")
            if len(parts) >= 3:
                try:
                    y, m = int(parts[-2]), int(parts[-1])
                    if 2000 <= y <= 2100 and 1 <= m <= 12:
                        tags.add(f"{y}_{m:02d}")
                except ValueError:
                    pass
        time_tags = sorted(tags)

        if not time_tags:
            raise ValueError("No time-tagged columns found in DataFrame.")

        n_samples = len(self.data)
        n_timesteps = len(time_tags)
        n_features = len(self.feature_bands)

        sequences = np.full((n_samples, n_timesteps, n_features), np.nan, dtype=np.float32)

        for t_idx, tag in enumerate(time_tags):
            for f_idx, band in enumerate(self.feature_bands):
                col = f"{band}_{tag}"
                if col in self.data.columns:
                    sequences[:, t_idx, f_idx] = self.data[col].values.astype(np.float32)

        # Fill NaN with 0 for model compatibility
        sequences = np.nan_to_num(sequences)

        # Extract labels
        labels = self.data[self.label_col].values.astype(np.float32)

        return sequences, labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return seq, label

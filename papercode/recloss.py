"""
This file is part of the accompanying code to our manuscript:

Kratzert, F., Klotz, D., Shalev, G., Klambauer, G., Hochreiter, S., Nearing, G., "Benchmarking
a Catchment-Aware Long Short-Term Memory Network (LSTM) for Large-Scale Hydrological Modeling".
submitted to Hydrol. Earth Syst. Sci. Discussions (2019)

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import torch


class RECLoss(torch.nn.Module):
    """Calculate (batch-wise) reconstruction loss Loss.

    Each sample i is weighted by 1 / (std_i + eps)^2, where std_i is the standard deviation of the 
    discharge from the basin, to which the sample belongs.

    Parameters:
    -----------
    eps : float
        Constant, added to the weight for numerical stability and smoothing, default to 0.1
    """

    def __init__(self, eps: float = 0.1):
        super(RECLoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor):
        """Calculate the batch-wise NSE Loss function.

        Parameters
        ----------
        y_pred : torch.Tensor
            Tensor containing the network prediction.
        y_true : torch.Tensor
            Tensor containing the true discharge values

        Returns
        -------
        torch.Tenor
            The (batch-wise) NSE Loss
        """
        
        squared_error = (y_pred - y_true)**2
        return torch.mean(squared_error)
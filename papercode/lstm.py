"""
This file is part of the accompanying code to our manuscript (currently under review):
Xiang Li, Ankush Khandelwal, Xiaowei Jia, Kelly Cutler, Rahul Ghosh, Arvind Renganathan, Kshitij Tayal, Shaoming Xu, 
John Nieber, Christopher Duffy, Michael Steinbach, Vipin Kumar. 2022. 
“Regionalization in a global hydrologic deep learning model: from physical descriptors to random vectors” 
Water Resources Research (Under review). Preprint is available: https://www.essoar.org/doi/10.1002/essoar.10510083.1
A majority of this code is built on the Kratzert. et al (2019), see their github repo: https://github.com/kratzert/ealstm_regional_modeling
"""

from typing import Tuple

import torch
import torch.nn as nn


class LSTM(nn.Module):
    """Implementation of the standard LSTM.

    TODO: Include ref and LaTeX equations

    Parameters
    ----------
    input_size : int
        Number of input features
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0
    """

    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(torch.FloatTensor(input_size, 4 * hidden_size))
        self.weight_hh = nn.Parameter(torch.FloatTensor(hidden_size, 4 * hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(4 * hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 4)
        self.weight_hh.data = weight_hh_data
        nn.init.constant_(self.bias.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]
        
        Parameters
        ----------
        x : torch.Tensor
            Tensor, containing a batch of input sequences. Format must match the specified format,
            defined by the batch_first agrument.

        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len, batch_size, _ = x.size()

        h_0 = x.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(batch_size, *self.bias.size()))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) + torch.mm(x[t], self.weight_ih))
            f, i, o, g = gates.chunk(4, 1)

            c_1 = torch.sigmoid(f) * c_0 + torch.sigmoid(i) * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n

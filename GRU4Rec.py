import torch
import torch.nn as nn
import torch.nn.functional as F

from BasicModel import BasicModel


class GRU4Rec(BasicModel):

    def __init__(self, item_num, hidden_size, batch_size, seq_len=50, num_layers=3, dropout=0.1):
        super(GRU4Rec, self).__init__(item_num, hidden_size, batch_size, seq_len)
        
        self.gru = nn.GRU(
                        input_size = self.hidden_size, 
                        hidden_size = self.hidden_size, 
                        num_layers=num_layers, 
                        batch_first=True, 
                        dropout=dropout
                    )


    def forward(self, item_list, label_list, mask, device, train=True):

        item_eb = self.embeddings(item_list) # [b, s, h]
        output, fin_state = self.gru(item_eb) # [b, s, h], [num_layers, b, h]
        user_eb = fin_state[-1]
        scores = self.calculate_score(user_eb)
        
        return user_eb, scores

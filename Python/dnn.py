import torch
import torch.nn as nn
import numpy as np
import OrderedDict
from sklearn.ensemble import RandomForestClassifier

class BaseDNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, output_dim, num_layers=1, base_model=None, selected_features=None):
        super(BaseDNN, self).__init__()
        
        self.selected_features = selected_features
        if self.selected_features is not None:
            in_dim = len(self.selected_features) 

        layers = self._block(in_dim, hidden_dim, 0)

        for layer_idx in range(1, num_layers):
            layers.extend(self._block(hidden_dim, hidden_dim, layer_idx))

        self.base_model = base_model
        if base_model:
            print(len(base_model.layers))
            if type(base_model.layers[-3]) != nn.Linear:
                raise Exception("Base model must be a DNN")

            hidden_dim += base_model.layers[-3].out_features

        self.output = nn.Linear(hidden_dim, output_dim)
        self.layers = nn.Sequential(OrderedDict(layers))


    def _block(self, in_dim, out_dim, index, dropout=0.5):
        return [
            (f"linear_{index}", nn.Linear(in_dim, out_dim)),
            # (f"relu_{index}", nn.ReLU()),
            # (f"selu_{index}", nn.SELU()),
            (f"selu_{index}", nn.GELU()),
            # (f"selu_{index}", nn.SiLU()),
            (f"dropout_{index}", nn.Dropout(dropout)),
        ]

    def forward(self, inputs, features=False):
        if self.selected_features is not None:
            inputs = inputs[:, self.selected_features]         
        x = self.layers(inputs)
        
        if self.base_model:
            x = torch.cat((self.base_model(inputs, features=True), x), dim=1)
            
        if features:
            return x

        return self.output(x)

rf = RandomForestClassifier()
rf.fit(X_train, y_train)
importances = rf.feature_importances_
selected_features = np.argsort(importances)[::-1][:1000] 

# 2. Create your model with selected features
model = BaseDNN(in_dim=X_train.shape[1], hidden_dim=128, output_dim=1, 
                num_layers=3, selected_features=selected_features) 
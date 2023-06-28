import torch
from torch import nn

import timm

class ClassifierBlock(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, p_dropout=0.2):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=p_dropout),
            nn.Linear(in_features=hidden_size, out_features=num_classes)            
        )
    
    def forward(self, x):
        return self.classifier(x)

class LWTransformer(nn.Module):
    def __init__(self, base_model, num_classes, classifier_dropout=0.2, feature_only=True):
        super().__init__()
        self.num_classes = num_classes
        self.base_model = base_model
        self.cls_token = self.base_model.cls_token # -> shape = (1, 1, embed-dim)
        self.num_blocks = 12
        self.linear = nn.Linear(in_features=768, out_features=768)
        self.classifier = ClassifierBlock(input_size=1536, hidden_size=1536, num_classes=num_classes, p_dropout=classifier_dropout) # 768 + 768 = 1536 (= concat aggregated token and global token)
        self.feature_only = feature_only

    def forward(self, x):
        # Divide input image into patch embeddings 
        x = self.base_model.patch_embed(x) # -> shape = (B, 196 embeddings, 768 embed-dim) 
        cls_token = self.cls_token.expand(x.shape[0], -1, -1) # Modify batch size -> shape = (B, 1, embed-dim)
        x = torch.cat((cls_token, x), dim=1) # -> concat X + cls_token -> shape = (B, 196 + 1, embed-dim)
        # Add position embeddings
        x = x + self.base_model.pos_embed # pos_embed is also a learnable parameter
        x = self.base_model.pos_drop(x)
        
        # Feed forward through transformer blocks
        for i in range(self.num_blocks):
            x = self.base_model.blocks[i](x)
        x = self.base_model.norm(x)
        
        # Get tokens
        cls_token_out = x[:, 0]    
        local_tokens = x[:,1:]
        local_tokens = self.linear(local_tokens)
        aggregated_token = torch.sum(local_tokens, dim=1)
        feature = torch.cat((aggregated_token, cls_token_out), dim=1)
        
        if not self.feature_only:
            x = self.classifier(feature)
            return feature, x
        else:
            return feature

def make_model(config, num_classes, feature_only=True):
    base_model = timm.create_model(config.MODEL.BASE_MODEL, pretrained=True)
    base_model = base_model.to(config.MODEL.DEVICE)
    base_model.eval()
    
    model = LWTransformer(base_model, num_classes=num_classes, feature_only=feature_only, classifier_dropout=config.MODEL.DROPOUT)
    return model


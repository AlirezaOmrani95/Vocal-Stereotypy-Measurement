import torch.nn as nn
import timm


class Pretrained_Models():
    def __init__(self,model_name,class_num,input_size):
        self.model_name = model_name
        self.class_num = class_num
        self.input_size = input_size
    def get_model(self):
        cfg_file = timm.get_pretrained_cfg(self.model_name)
        cfg_file.input_size = self.input_size
        model = timm.create_model(self.model_name,pretrained_cfg=cfg_file,pretrained=True)
        model.head = nn.Linear(model.head.in_features,self.class_num)

        return model

        

if __name__ == '__main__':
    model = Pretrained_Models('xcit_tiny_24_p16_384',1,(2,64,44)).get_model()
    print(model.patch_embed.proj[2][0].weight[0,0,0,0])
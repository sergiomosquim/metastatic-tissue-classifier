import torch
from train import PCamResNet # import the model class from the train.py script

# set up the model
model = PCamResNet()
model.load_state_dict(torch.load('./models/best_model.pth', map_location='cpu'))
model.eval()

# create the dummy imput (standard PCam size is 96x96)
dummy_imput = torch.rand(1,3,96,96)

# export the model to onnx
torch.onnx.export(
    model,
    dummy_imput, 
    './models/best_model.onnx',
    verbose = True,
    input_names=['input'],
    output_names=['output'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
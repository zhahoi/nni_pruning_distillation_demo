import torch
from model import vgg

from nni.compression.pruning import L1NormPruner
from nni.compression.speedup import ModelSpeedup
from nni.compression.utils.counter import count_flops_params

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model_name = "vgg16"
    model = vgg(model_name=model_name, num_classes=2, init_weights=True).to(device)
    model.load_state_dict(torch.load('./vgg16Net.pth'))
    
    print('=================================== original model ===================================')
    print(model)
   
    # check model FLOPs and parameter counts with NNI utils
    dummy_input = torch.rand([1, 3, 224, 224]).to(device)
    flops, params, results = count_flops_params(model, dummy_input)
    print(f"FLOPs: {flops}, params: {params}")
    
     # The following `config_list` means all layers whose type is `Linear` or `Conv2d` will be pruned,
    # except the layer named `fc3`, because `fc3` is `exclude`.
    # The final sparsity ratio for each layer is 50%. The layer named `fc3` will not be pruned.
    config_list = [{
        'op_types': ['Linear', 'Conv2d'],
        'exclude_op_names': ['classifier.6'],
        'sparse_ratio': 0.5
    }]
    
    pruner = L1NormPruner(model, config_list)
    
    # show the wrapped model structure, `PrunerModuleWrapper` have wrapped the layers that configured in the config_list.
    print('=================================== wrapped model ===================================')
    print(model)
    
     # compress the model and generate the masks
    _, masks = pruner.compress()
    # show the masks sparsity
    for name, mask in masks.items():
        print(name, ' sparsity : ', '{:.2}'.format(mask['weight'].sum() / mask['weight'].numel()))
        
    # Speedup the original model with masks, note that `ModelSpeedup` requires an unwrapped model.
    # The model becomes smaller after speedup,
    # and reaches a higher sparsity ratio because `ModelSpeedup` will propagate the masks across layers.
 
    # need to unwrap the model, if the model is wrapped before speedup
    pruner.unwrap_model()
 
     # speedup the model, for more information about speedup, please refer :doc:`pruning_speedup`.
    ModelSpeedup(model, torch.rand(1, 3, 224, 224).to(device), masks).speedup_model()
    
    # the model will become real smaller after speedup
    print('=================================== pruned model ===================================')
    print(model)
    torch.save(model, 'pruned_vgg16_net.pth')
    
    
if __name__ == '__main__':
    main()
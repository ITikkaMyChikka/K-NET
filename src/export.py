import torch
import os
#import main
#os.system("main.py")
cpp = False
path_input = "/Users/adrialopezescoriza/Documents/WORK/DRIVERLESS/AMZ/code/dv_prototyping_2021/KalmanNet/Results/Simulation_"
path_output = "Export/best-model"
simulation_id = input("Select simulation ID to export model from: ")

path_model_in  = path_input  + str(simulation_id) + '/Results/best-model.pt'
path_model_out = path_output + str(simulation_id)

try:
    my_model = torch.load(path_model_in)
except:
    raise("File not found error")


my_model.eval()

if(cpp):
    script_module = torch.jit.script(my_model)
    script_module.save(path_model_out)
    print("Model successfully exported to C++ in", path_model_out+"_cpp.pt")



#traced_script_module = torch.jit.trace(my_model, example)
#traced_script_module.save(path_model_out)

# Export the model to onnx
else:
    with torch.no_grad():
        x = torch.randn(5,requires_grad=False)
        torch_out = my_model(x)
        print(torch_out)
        torch.onnx.export(my_model,               # model being run
                        x,                         # model input (or a tuple for multiple inputs)
                        path_model_out+".onnx",   # where to save the model (can be a file or file-like object)
                        export_params=True,        # store the trained parameter weights inside the model file
                        opset_version=11,          # the ONNX version to export the model to
                        do_constant_folding=True,  # whether to execute constant folding for optimization
                        input_names = ['input'],   # the model's input names
                        output_names = ['output'], # the model's output names
                        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                                        'output' : {0 : 'batch_size'}})
    print("Model successfully exported to ONNX in", path_model_out+".onnx")
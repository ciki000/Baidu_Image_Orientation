from winmltools.utils import convert_float_to_float16
from winmltools.utils import load_model, save_model
onnx_model = load_model('model_b256.onnx')
new_onnx_model = convert_float_to_float16(onnx_model)
save_model(new_onnx_model, 'model_bfp16.onnx')

# import onnxmltools
# from onnxmltools.utils.float16_converter import convert_float_to_float16

# # Update the input name and path for your ONNX model
# input_onnx_model = 'model_256.onnx'
# # Change this path to the output name and path for your float16 ONNX model
# output_onnx_model = 'model_f16.onnx'
# # Load your model
# onnx_model = onnxmltools.utils.load_model(input_onnx_model)
# # Convert tensor float type from your input ONNX model to tensor float16
# onnx_model = convert_float_to_float16(onnx_model)
# # Save as protobuf
# onnxmltools.utils.save_model(onnx_model, output_onnx_model)

# import onnx
# from onnxsim import simplify

# # load your predefined ONNX model
# model = onnx.load('model_256.onnx')
# output_onnx_model = 'model_sim.onnx'

# # convert model
# model_simp, check = simplify(model)

# assert check, "Simplified ONNX model could not be validated"
# onnxmltools.utils.save_model(model_simp, output_onnx_model)
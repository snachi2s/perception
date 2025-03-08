import os
import argparse
import tensorrt as trt

def build_engine(onnx_file, trt_file, use_fp16=False):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    #explicit batch mode
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    # Parse the ONNX model file
    with open(onnx_file, "rb") as model:
        if not parser.parse(model.read()):
            print(f"Failed to parse ONNX file: {onnx_file}")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False

    #builder configuration and set workspace memory
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB workspace

    #enable FP16 if specified and supported, otherwise, use default FP32
    if use_fp16:
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print(f"FP16 mode enabled for: {onnx_file}")
        else:
            print(f"FP16 mode not supported on this platform for: {onnx_file}. Using FP32.")
    else:
        print(f"Using FP32 mode for: {onnx_file}")

    #build
    engine = builder.build_serialized_network(network, config)
    if engine is None:
        print(f"Engine build failed for: {onnx_file}")
        return False

    #save the serialized engine
    with open(trt_file, "wb") as f:
        f.write(engine)
    print(f"TensorRT engine saved at: {trt_file}")
    return True

def convert_all_models(source_dir, target_dir, use_fp16=False):
    for root, dirs, files in os.walk(source_dir):
        relative_path = os.path.relpath(root, source_dir)
        target_subdir = os.path.join(target_dir, relative_path)
        os.makedirs(target_subdir, exist_ok=True)
        
        for file in files:
            if file.endswith(".onnx"):
                onnx_path = os.path.join(root, file)
                trt_filename = os.path.splitext(file)[0] + ".trt"
                trt_path = os.path.join(target_subdir, trt_filename)
                print(f"Converting {onnx_path} to {trt_path} ...")
                build_engine(onnx_path, trt_path, use_fp16)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert ONNX models to TensorRT engines with FP16 or FP32 precision.")
    parser.add_argument("--source_dir", type=str, default="onnx_initial_experiments",
                        help="Directory containing ONNX models.")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp32",
                        help="Target precision for conversion (fp16 or fp32).")
    args = parser.parse_args()

    #save directory
    if args.precision == "fp16":
        target_directory = "hp_tuned_FP16_tensorrt_models"
        use_fp16 = True
    else:
        target_directory = "hp_tuned_FP32_tensorrt_models"
        use_fp16 = False

    convert_all_models(args.source_dir, target_directory, use_fp16)
import tensorrt as trt
import os


# todo: modify the input part

def onnx2trt(model_version_dir, onnx_model_file, max_batch):
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    # The EXPLICIT_BATCH flag is required in order to import models using the ONNX parser
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(onnx_model_file)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        pass  # Error handling code here

    profile = builder.create_optimization_profile()
    # INPUT0可以接收[1, 2] -> [max_batch, 2]的维度
    profile.set_shape("INPUT0", [1, 2], [1, 2], [max_batch, 2])
    profile.set_shape("INPUT1", [1, 2], [1, 2], [max_batch, 2])

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    # tensorrt 8.x
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB

    # tensorrt 7.x
    # config.max_workspace_size = 1 << 20

    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    with open(os.path.join(model_version_dir, 'model.plan'), "wb") as f:
        f.write(engine_bytes)

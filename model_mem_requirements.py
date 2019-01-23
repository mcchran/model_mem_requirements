import argparse
import importlib.util

parser = argparse.ArgumentParser()
parser.add_argument("-b", "--batch_size", type=int,  help="batch size to be used", required=True)
parser.add_argument("-p", "--model_file_path", type=str, help="path_to_the_model", required=True)
parser.add_argument("-m", "--model_name", help="model_name", required=True)
args = parser.parse_args()


def get_model_memory_usage(batch_size, model):
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)

    # print(shapes_mem_count)
    # print(trainable_count)
    # print(non_trainable_count)

    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes

if __name__ == "__main__":
    print(args.model_file_path)

    spec = importlib.util.spec_from_file_location("modeldefmodule", args.model_file_path)
    foo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(foo)

    try:
        Model = getattr(foo, args.model_name)
    except:
        print(" -- Model definition has not been found!")
        exit(1)
    
    model = Model()
    print("Model requires: ", get_model_memory_usage(args.batch_size, model), "GBytes")

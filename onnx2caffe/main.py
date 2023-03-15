import argparse
import logging
from .convertCaffe import convertToCaffe, getGraph
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_model", help="Input ONNX model")
    parser.add_argument("output_prototxt", help="Output caffe prototxt")
    parser.add_argument("output_caffemodel", help="Output caffe model")
    parser.add_argument(
        "--simplify",
        help="whether to simplify onnx model",
        action="store_true",
    )
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    ## simplify onnx model
    if args.simplify:
        import onnxsim, onnx

        model_opt, check_ok = onnxsim.simplify(args.input_model)
        if check_ok:
            onnx.save(model_opt, args.input_model)
            logging.info(f"Successfully simplified onnx model: {args.input_model}")
        else:
            logging.warn("Failed to simplify onnx model")
    
    ## convert to caffe
    graph = getGraph(args.input_model)
    caffe_model = convertToCaffe(graph, args.output_prototxt, args.output_caffemodel)
    logging.info(f"Successfully export caffe to: {args.output_prototxt}")
    logging.info(f"Successfully export caffe to: {args.output_caffemodel}")

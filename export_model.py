import argparse
import sys
from model import build_EfficientPose
from tensorflow.python.keras.utils import generic_utils
from tensorflow import keras
from tensorflow.compat.v2.keras import models
from layers import (
    ClipBoxes, 
    RegressBoxes, 
    FilterDetections, 
    wBiFPNAdd, 
    BatchNormalization, 
    RegressTranslation, 
    CalculateTxTy, 
    GroupNormalization,
    CustomConcat
)
from initializers import PriorProbability


"""Export EfficientPose model as a SavedModel
"""


class FixedDropout(keras.layers.Dropout):

    def __init__(self, *args, **kwargs):
        super(FixedDropout, self).__init__(*args, **kwargs)

    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape

        symbolic_shape = keras.backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                        for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)

    def get_config(self):
        config = super(FixedDropout, self).get_config()
        return config



def build_model(weights_dir, phi, num_classes, score_threshold):
    """
    Builds an EfficientPose model and init it with a given weight file
    Args:
        phi: EfficientPose scaling hyperparameter
        num_classes: The number of classes
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        path_to_weights: Path to the weight file
        
    Returns:
        efficientpose_prediction: The EfficientPose model
        image_size: Integer image size used as the EfficientPose input resolution for the given phi

    """
    print("\nBuilding model...\n")
    _, efficientpose_prediction, _ = build_EfficientPose(phi,
                                                         num_classes = num_classes,
                                                         num_anchors = 9,
                                                         freeze_bn = True,
                                                         score_threshold = score_threshold,
                                                         num_rotation_parameters = 3,
                                                         print_architecture = False)
    
    print("\nDone!\n\nLoading weights...")
    efficientpose_prediction.load_weights(weights_dir, by_name=True)
    print("Done!")
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    
    return efficientpose_prediction, image_size


def main(args):
    model, image_size = build_model(args.weights, args.phi, 1, 0.5)
    print(model)
    print(generic_utils.get_custom_objects())
    models.save_model(model, args.export, save_format='tf', include_optimizer=False)
    # model.save(args.export, save_format='tf', signatures=signatures, include_optimizer=False)
    # print(model.get_config())
    # model.save(args.export + '.h5', save_format='h5')
    # model.save_weights(args.export + '_weights.h5', save_format='h5')

    # keras.models.load_model(args.export, custom_objects={
    #     'ClipBoxes': ClipBoxes, 
    #     'RegressBoxes': RegressBoxes, 
    #     'FilterDetections': FilterDetections, 
    #     'wBiFPNAdd': wBiFPNAdd, 
    #     'BatchNormalization': BatchNormalization, 
    #     'RegressTranslation': RegressTranslation, 
    #     'CalculateTxTy': CalculateTxTy, 
    #     'GroupNormalization': GroupNormalization,
    #     'FixedDropout': FixedDropout,
    #     'PriorProbability': PriorProbability,
    #     'CustomConcat': CustomConcat
    # })



parser = argparse.ArgumentParser(description="Export EfficientPose model.")
parser.add_argument('--weights', required=True, help='File containing weights to init the model.')
parser.add_argument('--phi', help='Hyper parameter phi.', type=int, default=0)
parser.add_argument('--export', required=True, help='Directory path to export the serialized model.')

if __name__ == '__main__':
    args = parser.parse_args(sys.argv[1:])
    print(args)

    main(args)
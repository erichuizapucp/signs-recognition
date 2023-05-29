from learning.model.legacy.opticalflow_model_builder import OpticalFlowModelBuilder
from learning.model.legacy.rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from learning.model.legacy.nsdm_builder import NSDMModelBuilder
from learning.model.legacy.nsdm_v2_builder import NSDMV2ModelBuilder
from learning.model.nsdmv3.nsdm_v3_builder import NSDMV3ModelBuilder
from learning.model.swav.swav_builder import SwAVModelBuilder

from learning.common.model_type import OPTICAL_FLOW, RGB, SWAV


def get_saved_model(model_name):
    models = {
        OPTICAL_FLOW: lambda: OpticalFlowModelBuilder(),
        RGB: lambda: RGBRecurrentModelBuilder(),
        SWAV: lambda: SwAVModelBuilder()
    }
    model = models[model_name]().load_saved_model()
    return model,


def get_opticalflow_model():
    model_builder = OpticalFlowModelBuilder()
    opticalflow_model = model_builder.build()
    return opticalflow_model,


def get_rgb_model():
    model_builder = RGBRecurrentModelBuilder()
    rgb_model = model_builder.build()
    return rgb_model,


def get_nsdm_model():
    opticalflow_model = get_saved_model(OPTICAL_FLOW)
    rgb_model = get_saved_model(RGB)

    model_builder = NSDMModelBuilder()
    nsdm_model = model_builder.build(OpticalflowModel=opticalflow_model, RGBModel=rgb_model)
    return nsdm_model,


def get_nsdm_v2_model():
    opticalflow_model = get_saved_model(OPTICAL_FLOW)
    rgb_model = get_saved_model(RGB)

    model_builder = NSDMV2ModelBuilder()
    nsdm_v2_model = model_builder.build(OpticalflowModel=opticalflow_model, RGBModel=rgb_model)
    return nsdm_v2_model,


def get_nsdm_v3_model(**kwargs):
    model_builder = NSDMV3ModelBuilder()
    model = model_builder.build(lstm_cells=kwargs['lstm_cells'],
                                embedding_size=kwargs['embedding_size'],
                                load_weights=kwargs['load_weights'],
                                swav_features_weights_path=kwargs['swav_features_weights_path'],
                                no_dense_layer1_neurons=kwargs['no_dense_layer1_neurons'],
                                no_dense_layer2_neurons=kwargs['no_dense_layer2_neurons'],
                                no_dense_layer3_neurons=kwargs['no_dense_layer3_neurons'])
    return model


def get_swav_models(**kwargs):
    model_builder = SwAVModelBuilder()
    feature_embeddings_model = model_builder.build(lstm_cells=kwargs['lstm_cells'])
    prototype_projections_model = model_builder.build2(embedding_size=kwargs['embedding_size'],
                                                       no_projection_1_neurons=kwargs['no_projection_1_neurons'],
                                                       no_projection_2_neurons=kwargs['no_projection_2_neurons'],
                                                       prototype_vector_dim=kwargs['prototype_vector_dim'],
                                                       l2_regularization_epsilon=kwargs['l2_regularization_epsilon'])

    return feature_embeddings_model, prototype_projections_model


def get_swav_features_model(**kwargs):
    model_builder = SwAVModelBuilder()
    feature_embeddings_model = model_builder.build(lstm_cells=kwargs['lstm_cells'])

    return feature_embeddings_model


def get_swav_projections_model(**kwargs):
    model_builder = SwAVModelBuilder()
    prototype_projections_model = model_builder.build2(embedding_size=kwargs['embedding_size'],
                                                       no_projection_1_neurons=kwargs['no_projection_1_neurons'],
                                                       no_projection_2_neurons=kwargs['no_projection_2_neurons'],
                                                       prototype_vector_dim=kwargs['prototype_vector_dim'],
                                                       l2_regularization_epsilon=kwargs['l2_regularization_epsilon'])
    return prototype_projections_model

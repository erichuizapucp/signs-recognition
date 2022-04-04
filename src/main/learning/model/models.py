from learning.model.legacy.opticalflow_model_builder import OpticalFlowModelBuilder
from learning.model.legacy.rgb_recurrent_model_builder import RGBRecurrentModelBuilder
from learning.model.legacy.nsdm_builder import NSDMModelBuilder
from learning.model.legacy.nsdm_v2_builder import NSDMV2ModelBuilder
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


def get_swav_models():
    model_builder = SwAVModelBuilder()

    feature_embeddings_model = model_builder.build()
    prototype_projections_model = model_builder.build2()

    return feature_embeddings_model, prototype_projections_model

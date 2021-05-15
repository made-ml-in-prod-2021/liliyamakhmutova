from dataclasses import dataclass

import yaml
from marshmallow_dataclass import class_schema

from .feature_params import FeatureParams
from .split_params import SplittingParams
from .train_params import TrainingParams


@dataclass()
class TrainingPipelineParams:
    input_data_path: str
    output_model_path: str
    output_transformer_path: str
    metric_path: str
    pretrained_model_path: str
    predictions_path: str
    splitting_params: SplittingParams
    train_params: TrainingParams
    feature_params: FeatureParams


TrainingPipelineParamsSchema = class_schema(TrainingPipelineParams)


def read_training_pipeline_params(path: str) -> TrainingPipelineParams:
    with open(path, "r") as input_stream:
        schema = TrainingPipelineParamsSchema()
        return schema.load(yaml.safe_load(input_stream))

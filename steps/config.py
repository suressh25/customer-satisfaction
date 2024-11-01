from zenml.steps import BaseParameters


class ModelNameConfig(BaseParameters):
    name: str = "LinearRegression"

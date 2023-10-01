from vish.model.decomposed.entity import TransformerData


def debug_dict(d: dict[str, TransformerData]):
    for key, value in d.items():
        print("\tKey:", key)
        print(
            "\t\tValue -> Data Shape: ",
            value.data.shape,
            "DType:",
            value.data.dtype,
        )
        print(
            "\t\tValue -> Label Shape: ",
            value.labels.shape,
            "DType:",
            value.labels.dtype,
        )

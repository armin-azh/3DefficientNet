from ._model import efficient_net


def efficient_net_b0(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.0,
                         default_resolution=224,
                         dropout_rate=0.2,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b0",
                         **kwargs)


def efficient_net_b1(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.0,
                         depth_coefficient=1.1,
                         default_resolution=240,
                         dropout_rate=0.2,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b1",
                         **kwargs)


def efficient_net_b2(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.1,
                         depth_coefficient=1.2,
                         default_resolution=260,
                         dropout_rate=0.3,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b2",
                         **kwargs)


def efficient_net_b3(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.2,
                         depth_coefficient=1.4,
                         default_resolution=300,
                         dropout_rate=0.3,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b3",
                         **kwargs)


def efficient_net_b4(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.4,
                         depth_coefficient=1.8,
                         default_resolution=380,
                         dropout_rate=0.4,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b4",
                         **kwargs)


def efficient_net_b5(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.6,
                         depth_coefficient=2.2,
                         default_resolution=456,
                         dropout_rate=0.4,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b5",
                         **kwargs)


def efficient_net_b6(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=1.8,
                         depth_coefficient=2.6,
                         default_resolution=528,
                         dropout_rate=0.5,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b6",
                         **kwargs)


def efficient_net_b7(input_shape=None, classes=1000, **kwargs):
    return efficient_net(width_coefficient=2.0,
                         depth_coefficient=3.1,
                         default_resolution=600,
                         dropout_rate=0.5,
                         input_shape=input_shape,
                         classes=classes,
                         model_name="efficient_net_b7",
                         **kwargs)
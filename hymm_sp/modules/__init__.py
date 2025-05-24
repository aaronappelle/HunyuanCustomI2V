from .models import HYVideoDiffusionTransformer, HUNYUAN_VIDEO_CONFIG
from .models import HYVideoDiffusionTransformerI2V

def load_model(args, in_channels, out_channels, factor_kwargs):
    if args.i2v_mode:
        model = HYVideoDiffusionTransformerI2V(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
    else:
        model = HYVideoDiffusionTransformer(
            args,
            in_channels=in_channels,
            out_channels=out_channels,
            **HUNYUAN_VIDEO_CONFIG[args.model],
            **factor_kwargs,
        )
    return model

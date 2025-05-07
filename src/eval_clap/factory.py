from .model import SIMPLE_CLAP, SLIDE_CLAP
from .clap import MS_CLAP_2023, MS_CLAP_2022, M2D_CLAP, LAION_CLAP

def create_model(model_name):
    scorer_name, clap_name = model_name.split('-')
    scorer_class = SIMPLE_CLAP if scorer_name == 'SIMPLE' else SLIDE_CLAP

    if clap_name == 'MS_CLAP_2022':
        clap = MS_CLAP_2022(use_local_model=True)
        return scorer_class(
            window_size=5.0,
            hop_size=1.0,
            clap=clap
        )
    elif clap_name == 'MS_CLAP_2023':
        clap = MS_CLAP_2023(use_local_model=True)
        return scorer_class(
            window_size=7.0,
            hop_size=1.0,
            clap=clap
        )
    elif clap_name == 'M2D_CLAP':
        clap = M2D_CLAP()
        return scorer_class(
            window_size=10.0,
            hop_size=1.0,
            clap=clap
        )
    elif clap_name == 'LAION_CLAP':
        clap = LAION_CLAP()
        return scorer_class(
            window_size=7.0,
            hop_size=1.0,
            clap=clap
        )
    else:
        raise ValueError(f"Unknown model name: {model_name}")
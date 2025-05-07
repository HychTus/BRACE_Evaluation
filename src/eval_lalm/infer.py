from .factory import create_model


def inference_single(audio_path, prompt, model):
    output = model.inference_single(audio_path, prompt)
    return output


def inference_batch(audio_path, prompt, model_name):
    raise NotImplementedError


if __name__ == '__main__':
    pass
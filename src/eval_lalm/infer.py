from .factory import create_model
from ..utils import PROJECT_DIR

def inference_single(audio_path, prompt, model):
    # NOTE: 可以先直接输出 caption_0 进行测试
    # import random
    # randint 生成随机数为闭区间（同时测试错误处理是否正常）
    # return f'caption_{random.randint(0, 2)}'

    output = model.inference_single(audio_path, prompt)
    return output


def inference_batch(audio_path, prompt, model_name):
    raise NotImplementedError


# 保留 unit test 的代码方便使用
def test_inference_single(model_name):
    sample_audio_path = f'{PROJECT_DIR}/sample_audio.wav'
    model = create_model(model_name)

    print(inference_single(
        audio_path=sample_audio_path,
        prompt='Describe the audio.',
        model=model
    ))

    print(inference_single(
        audio_path=sample_audio_path,
        prompt='What is the audio about?',
        model=model
    ))


if __name__ == '__main__':
    # 不同 model 使用的环境不同，可能要分别测试
    # test_inference_single('GAMA')
    # test_inference_single('LTU')
    # test_inference_single('LTU_AS')
    # test_inference_single('SALMON')
    # test_inference_single('AF2-1.5B')
    # test_inference_single('AF2-0.5B')
    # test_inference_single('AF2-3B')
    pass
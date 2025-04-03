from .factory import create_model

def inference_single(audio_path, prompt, model_name):
    # NOTE: 可以先直接输出 caption_0 进行测试
    # import random
    # randint 生成随机数为闭区间（同时测试错误处理是否正常）
    # return f'caption_{random.randint(0, 2)}'

    model = create_model(model_name) # 工厂函数
    output = model.inference_single(audio_path, prompt)
    return output


def inference_batch(audio_path, prompt, model_name):
    raise NotImplementedError


def test_inference_single(model_name):
    # 保留 unit test 的代码方便使用
    sample_audio_path = '/mnt/data/lh/chy/BRACE_Eval/sample_audio.wav'

    print(inference_single(
        audio_path=sample_audio_path,
        prompt='Describe the audio.',
        model_name=model_name
    ))

    print(inference_single(
        audio_path=sample_audio_path,
        prompt='What is the audio about?',
        model_name=model_name
    ))

if __name__ == '__main__':
    # test_inference_single('GAMA')
    # test_inference_single('LTU')
    # test_inference_single('AF2-1.5B')
    pass
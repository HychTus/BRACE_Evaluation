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
    # 测试单个推理
    print(inference_single(
        audio_path='/mnt/public/data/lh/chy/GAMA/sample_audio.wav',
        prompt='Describe the audio.',
        model_name=model_name
    ))

    print(inference_single(
        audio_path='/mnt/public/data/lh/chy/GAMA/sample_audio.wav',
        prompt='What is the audio about?',
        model_name=model_name
    ))

if __name__ == '__main__':
    test_inference_single('GAMA')
    # test_inference_single('LTU')
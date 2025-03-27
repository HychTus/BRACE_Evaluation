from vllm import LLM, SamplingParams

# 初始化模型 (请根据实际模型名称调整)
model_name = "/mnt/public/data/lh/models/Qwen2.5-7B-Instruct" 

# 初始化LLM
llm = LLM(
    model=model_name,
    trust_remote_code=True,
    tensor_parallel_size=1  # 根据GPU数量调整
)

# 设置生成参数
sampling_params = SamplingParams(
    temperature=0.8,
    top_p=0.9,
    max_tokens=256
)

# 定义测试prompt
prompts = [
    "给我解释量子计算的基本原理",
    "用Python写一个快速排序算法",
    "写一首关于秋天的五言绝句"
]

# 生成文本
outputs = llm.generate(prompts, sampling_params)

# 打印结果
for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated text: {output.outputs[0].text}\n{'-'*50}")
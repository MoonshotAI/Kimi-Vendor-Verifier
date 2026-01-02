from inspect_ai import eval
from aime2025 import aime2025
from mmmu_pro_vision import mmmu_pro_v
from ocr_bench import ocrbench


if __name__ == "__main__":
    # ocrbench 评测
    eval(
        [ocrbench],
        ["openai-api/kimi/{your_model_id}"],
        temperature=1.0,
        max_tokens=10000,
        max_connections=1000,
        model_args={"stream": True},
        extra_body={
            "thinking": {"type": "disabled"}, # 思考模式关闭
            # "thinking": {"type": "enabled"}, # 思考模式开启
        },
        retry_on_error=3,
        fail_on_error=True
    )

    # # mmmu_pro_vision 评测
    # eval(
    #     [mmmu_pro_v],
    #     ["openai-api/kimi/{your_model_id}"],
    #     temperature=1.0,
    #     max_tokens=100000,
    #     max_connections=1000,
    #     model_args={"stream": True},
    #     extra_body={
    #         "thinking": {"type": "disabled"}, # 思考模式关闭
    #         # "thinking": {"type": "enabled"}, # 思考模式开启
    #     },
    #     retry_on_error=3,
    #     fail_on_error=True
    # )

    # # aime2025 测32次计算得分
    # eval(
    #     [aime2025()],
    #     #   替换为自己需要测试的model_id
    #     ["openai-api/kimi/{your_model_id}"],
    #     temperature=0.6, # 非思考模式温度
    #     # temperature=1.0, # 思考模式温度
    #     max_tokens=100000,
    #     max_connections=1000,
    #     epochs=32,
    #     model_args={"stream": True},
    #     extra_body={
    #         "thinking": {"type": "disabled"}, # 思考模式关闭
    #         # "thinking": {"type": "enabled"}, # 思考模式开启
    #     },
    #     retry_on_error=3,
    #     fail_on_error=True
    # )

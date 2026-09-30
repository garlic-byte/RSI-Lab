"""检查手写注意力：与 PyTorch 对比输出、输入梯度、参数梯度和整网接入。"""

import copy

import torch
from torch import nn

from config.train_config import ModelConfig
from model.velocity_transformer import VelocityTransformer, manual_multi_head_attention


def check_attention(batch: int, tokens: int, width: int, heads: int, bias: bool) -> None:
    """使用相同权重和输入，检查单层数值及反向传播是否等价。"""
    reference = nn.MultiheadAttention(width, heads, dropout=0., bias=bias, batch_first=True).double()
    candidate = copy.deepcopy(reference)
    # 非连续输入也应能使用；不能假定 transpose 后的张量可直接 view。
    inputs = torch.randn(batch, width, tokens, dtype=torch.float64).transpose(1, 2)
    x_ref = inputs.detach().requires_grad_(True)
    x_manual = inputs.detach().requires_grad_(True)
    expected, _ = reference(x_ref, x_ref, x_ref, need_weights=False)
    actual = manual_multi_head_attention(x_manual, candidate)
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
    probe = torch.randn_like(expected)
    (expected * probe).sum().backward()
    (actual * probe).sum().backward()
    torch.testing.assert_close(x_manual.grad, x_ref.grad, atol=1e-9, rtol=1e-7)
    for name, parameter in candidate.named_parameters():
        torch.testing.assert_close(parameter.grad, dict(reference.named_parameters())[name].grad,
                                   atol=1e-9, rtol=1e-7, msg=lambda message: f"{name}: {message}")
    print(f"PASS: B={batch}, T={tokens}, D={width}, H={heads}, bias={bias}")


def check_model() -> None:
    """确认切换后整网训练/eval 等价，且不同 batch 样本之间不互相影响。"""
    config = ModelConfig(model_type="transformer", transformer_dim=16, num_heads=4,
                         hidden_dim=32, num_layers=2)
    reference = VelocityTransformer(config).double()
    candidate = copy.deepcopy(reference)
    candidate.set_manual_attention(True)
    points = torch.randn(5, 2, dtype=torch.float64)
    times = torch.rand(5, 1, dtype=torch.float64)
    labels = torch.tensor([0, 9, 19, 59, 99])
    for training in (True, False):
        reference.train(training)
        candidate.train(training)
        with torch.no_grad():
            expected = reference(points, times, labels)
            actual = candidate(points, times, labels)
            separate = torch.cat([candidate(points[i:i+1], times[i:i+1], labels[i:i+1])
                                  for i in range(len(labels))])
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-7)
        torch.testing.assert_close(actual, separate, atol=1e-9, rtol=1e-7)
    print("PASS: full model, train/eval, batch independence")


def main() -> None:
    """在 CPU 上运行小规模确定性检查，不启动训练或修改模型文件。"""
    torch.manual_seed(42)
    torch.set_num_threads(2)
    try:
        for case in [(2, 3, 16, 4, True), (1, 1, 8, 1, True),
                     (3, 7, 24, 3, True), (2, 5, 16, 4, False)]:
            check_attention(*case)
        check_model()
    except NotImplementedError as error:
        raise SystemExit(f"尚未完成练习：{error}") from None
    print("所有检查通过。还可以把实现发给我，继续检查代码结构和可读性。")


if __name__ == "__main__":
    main()

import platform

import torch


def check_mps_support():
    """检查 MPS 支持情况"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"macOS version: {platform.mac_ver()[0]}")

    print("\nMPS 支持检查:")
    print(f"MPS 是否可用: {torch.backends.mps.is_available()}")
    print(f"MPS 是否已构建: {torch.backends.mps.is_built()}")

    if torch.backends.mps.is_available():
        # 测试 MPS 功能
        try:
            device = torch.device("mps")
            x = torch.ones(5, device=device)
            print("\nMPS 测试成功！可以使用 MPS 进行训练。")
            print(f"测试张量: {x}")
        except Exception as e:
            print(f"\nMPS 测试失败: {e}")
    else:
        print("\nMPS 不可用，将使用 CPU 进行训练。")
        if not torch.backends.mps.is_built():
            print("原因: PyTorch 未启用 MPS 支持进行构建")
        else:
            print("原因: 可能是 macOS 版本低于 12.3 或设备不支持 MPS")


if __name__ == "__main__":
    check_mps_support()

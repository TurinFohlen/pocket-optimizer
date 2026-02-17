from __future__ import annotations
import numpy as np


# ================================
# SERVICE COMPONENT TEMPLATE
# ================================

#1️⃣ 注册信息（需配合 registry 使用时取消注释）
 @registry.register(
     name='service.data.tensor_array_translator',
    type_='service',
     signature='translate(data: np.ndarray) -> np.ndarray'
 

class TensorArrayTranslatorService:
    """
    组件名称：
        TensorArrayTranslator

    组件职责：
        双向转换 ndarray 维度
        张量（ndim >= 3）→ 二维数组
        二维数组（ndim == 2）→ 张量
        一维数组（ndim == 1）→ (1, n)

    输入格式：
        numpy.ndarray（任意 ndim）

    输出格式：
        numpy.ndarray（维度已变换）

    不做什么：
        不修改数值
        不做归一化
        不做类型推断
        不做语义判断
        不做复杂封装
    """

    # ================================
    # 单例模式（不可修改）
    # ================================
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True

    # ================================
    # 核心 API
    # ================================
    def translate(self, data: np.ndarray) -> np.ndarray:
        """
        功能说明：
            根据输入维度双向转换

        参数：
            data: numpy.ndarray

        返回：
            numpy.ndarray（维度已变换）
        """
        if data.ndim >= 3:
            return data.reshape(data.shape[0], -1)
        elif data.ndim == 2:
            return np.expand_dims(data, axis=0)
        else:
            return data.reshape(1, data.shape[0])


# ================================
# 全局实例
# ================================
tensor_array_translator = TensorArrayTranslatorService()


# ================================
# 模块级快捷函数（供直接调用）
# ================================
def translate(data: np.ndarray) -> np.ndarray:
    """
    如果是张量（ndim >= 3）→ 转数组
    如果是数组（ndim == 2）→ 转张量
    如果是一维（ndim == 1）→ reshape 为 (1, n)
    """
    return tensor_array_translator.translate(data)


# ================================
# 示例测试
# ================================
if __name__ == '__main__':
    # 张量 → 数组：(2, 3, 4) → (2, 12)
    tensor = np.ones((2, 3, 4))
    result = translate(tensor)
    assert result.shape == (2, 12), f"期望 (2,12)，实际 {result.shape}"
    print(f"[PASS] 张量→数组: {tensor.shape} → {result.shape}")

    # 数组 → 张量：(3, 4) → (1, 3, 4)
    array_2d = np.ones((3, 4))
    result = translate(array_2d)
    assert result.shape == (1, 3, 4), f"期望 (1,3,4)，实际 {result.shape}"
    print(f"[PASS] 数组→张量: {array_2d.shape} → {result.shape}")

    # 一维 → (1, n)：(5,) → (1, 5)
    array_1d = np.ones((5,))
    result = translate(array_1d)
    assert result.shape == (1, 5), f"期望 (1,5)，实际 {result.shape}"
    print(f"[PASS] 一维→行矩阵: {array_1d.shape} → {result.shape}")

    # 单例验证
    s1 = TensorArrayTranslatorService()
    s2 = TensorArrayTranslatorService()
    assert s1 is s2
    print("[PASS] 单例模式验证通过")

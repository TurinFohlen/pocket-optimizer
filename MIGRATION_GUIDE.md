# 组件迁移模板指南

## 标准组件结构

### 算法组件模板

```python
import numpy as np
from typing import List, Tuple
from registry import registry


@registry.register(
    name='algorithm.{algorithm_name}',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class {AlgorithmName}:
    def __init__(self, source, config=None):
        self.source = source
        self.config = config
        
    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        best_point = None
        best_value = -np.inf
        
        return best_point, best_value
```

### 测量源组件模板

```python
import numpy as np
from registry import registry


@registry.register(
    name='source.{source_name}',
    type_='source',
    signature='measure(point: np.ndarray, n_samples: int) -> float'
)
class {SourceName}:
    def __init__(self):
        pass
    
    def measure(self, point: np.ndarray, n_samples: int = 5) -> float:
        return 0.0
```

### 后处理组件模板

```python
from typing import List
from registry import registry


@registry.register(
    name='processor.{processor_name}',
    type_='processor',
    signature='process(history: List) -> Dict'
)
class {ProcessorName}:
    def __init__(self):
        pass
    
    def process(self, history: List) -> dict:
        return {}
```

## 迁移步骤

### 步骤1：识别原方法

原代码：
```python
class EnhancedUniversalOptimizer:
    def _genetic_algorithm_implementation(self, population_size, generations):
        # 原实现
        pass
```

### 步骤2：提取为独立类

```python
@registry.register(
    name='algorithm.genetic',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class GeneticAlgorithm:
    def __init__(self, source):
        self.source = source
        self.population_size = 20
        self.generations = 15
    
    def optimize(self, bounds):
        pass
```

### 步骤3：替换测量调用

原代码：
```python
value = self.measure(point, "genetic")
```

新代码：
```python
value = self.source.measure(point, n_samples=5)
```

### 步骤4：标准化返回值

所有算法必须返回：
```python
return (best_point, best_value)
```

## 实际示例：遗传算法迁移

### 原实现（简化版）

```python
class EnhancedUniversalOptimizer:
    def _genetic_algorithm_implementation(self, population_size, generations):
        bounds = self.config.param_bounds
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        population = np.random.uniform(lower, upper, size=(population_size, n_dims))
        
        for gen in range(generations):
            fitness = []
            for ind in population:
                eval_point = self.measure(ind, "genetic")
                fitness.append(eval_point.value)
            
        return best_point, best_value
```

### 迁移后

```python
import numpy as np
from typing import List, Tuple
from registry import registry


@registry.register(
    name='algorithm.genetic',
    type_='algorithm',
    signature='optimize(bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]'
)
class GeneticAlgorithm:
    def __init__(self, source):
        self.source = source
        self.population_size = 20
        self.generations = 15
    
    def optimize(self, bounds: List[Tuple[float, float]]) -> Tuple[np.ndarray, float]:
        n_dims = len(bounds)
        lower = np.array([b[0] for b in bounds])
        upper = np.array([b[1] for b in bounds])
        
        population = np.random.uniform(lower, upper, size=(self.population_size, n_dims))
        
        best_point = None
        best_value = -np.inf
        
        for gen in range(self.generations):
            fitness = []
            for ind in population:
                value = self.source.measure(ind, n_samples=5)
                fitness.append(value)
                
                if value > best_value:
                    best_value = value
                    best_point = ind.copy()
            
            fitness = np.array(fitness)
            
        return best_point, best_value
```

## 关键修改点

1. **装饰器注册**：添加 `@registry.register()`
2. **构造函数**：接收 `source` 参数
3. **测量调用**：`self.measure()` → `self.source.measure()`
4. **参数传递**：通过构造函数或 `optimize()` 参数
5. **返回值**：统一为 `(np.ndarray, float)` 元组

## 依赖自动检测原理

注册系统会扫描源码中的以下模式：
- `self.source.{component_name}`
- `source.{component_name}`
- 其他组件名称的直接引用

因此只要在代码中引用其他组件，依赖关系会自动建立。

## 配置传递策略

### 方式1：协调器传递（推荐）

```python
class Algorithm:
    def __init__(self, source):
        self.source = source
    
    def optimize(self, bounds):
        pass
```

协调器调用时传递bounds。

### 方式2：构造时传递

```python
class Algorithm:
    def __init__(self, source, config):
        self.source = source
        self.config = config
    
    def optimize(self, bounds):
        bounds = bounds or self.config.param_bounds
```

更灵活但增加耦合。

## 测试检查清单

- [ ] 添加 `@registry.register()` 装饰器
- [ ] 构造函数接收 `source` 参数
- [ ] 所有测量调用改为 `self.source.measure()`
- [ ] `optimize()` 方法签名正确
- [ ] 返回值为 `(best_point, best_value)`
- [ ] 运行后 `components.yaml` 自动更新
- [ ] 依赖关系正确记录
- [ ] 能通过协调器成功调用

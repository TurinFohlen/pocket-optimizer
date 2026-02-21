"""
JSON 翻译器服务（轻量版）
注册名: service.translator.json
依赖: chompjs (已安装)
"""

import json
from typing import Any, Union
from registry import registry

try:
    import chompjs
    HAS_CHOMP = True
except ImportError:
    HAS_CHOMP = False


@registry.register(
    name='service.translator.json',
    type_='service',
    signature='json_to_python(json_input: Union[str, dict], **kwargs) -> Any'
)
class JSONTranslatorService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def json_to_python(self,
                      json_input: Union[str, dict, bytes],
                      **kwargs) -> Any:
        if isinstance(json_input, dict):
            return json_input
        if isinstance(json_input, bytes):
            json_input = json_input.decode('utf-8')
        try:
            return json.loads(json_input)
        except json.JSONDecodeError:
            if HAS_CHOMP:
                try:
                    return chompjs.parse_js_object(json_input, **kwargs)
                except Exception as e:
                    raise ValueError(f"chompjs 清洗失败: {e}")
            else:
                raise ImportError("请安装 chompjs: pip install chompjs")

    def python_to_json(self, obj: Any, **kwargs) -> str:
        return json.dumps(obj, default=str, **kwargs)

    def to_file(self, obj: Any, filepath: str, **kwargs):
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(self.python_to_json(obj, **kwargs))
        print(f"✅ JSON已保存: {filepath}")

    def from_file(self, filepath: str, **kwargs) -> Any:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return self.json_to_python(content, **kwargs)

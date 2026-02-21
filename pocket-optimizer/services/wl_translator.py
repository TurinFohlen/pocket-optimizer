from registry import registry
#from FoxySheep import if2python(老版本Python可用)

@registry.register(
    name='service.translator.wl',
    type_='service',
    signature='wl_to_python(expr: str) -> str'
)
class WolframTranslatorService:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def wl_to_python(self, wl_expr: str) -> str:
        """将 Wolfram Language 表达式翻译为 Python 代码字符串"""
        try:
            return if2python(wl_expr)
        except Exception as e:
            print(f"⚠️ 翻译失败: {e}")
            return None
    
    def wl_to_value(self, wl_expr: str, local_dict=None):
        """翻译并立即求值"""
        py_code = self.wl_to_python(wl_expr)
        if py_code:
            try:
                return eval(py_code, local_dict or {})
            except Exception as e:
                print(f"⚠️ 求值失败: {e}")
        return None
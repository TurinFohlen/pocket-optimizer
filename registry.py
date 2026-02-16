from dataclasses import dataclass, field, asdict
from typing import Dict, List, Callable, Any, Tuple, Optional
import inspect
import yaml
import os
import re
from pathlib import Path


@dataclass
class ComponentSpec:
    name: str
    type: str
    signature: str
    dependencies: List[str] = field(default_factory=list)
    registration_order: int = 0
    source_file: str = ""
    class_name: str = ""


class Registry:
    def __init__(self, config_path: str = "components.yaml"):
        self.config_path = config_path
        self.components: Dict[str, ComponentSpec] = {}
        self.registration_counter = 0
        self.component_instances: Dict[str, Any] = {}
        self._load_existing()
    
    def _load_existing(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    if data and 'components' in data:
                        for comp_data in data['components']:
                            spec = ComponentSpec(**comp_data)
                            self.components[spec.name] = spec
                            if spec.registration_order >= self.registration_counter:
                                self.registration_counter = spec.registration_order + 1
            except Exception as e:
                pass
    
    def _analyze_dependencies(self, func_or_class) -> List[str]:
        deps = []
        
        if inspect.isclass(func_or_class) and hasattr(func_or_class, 'required_source'):
            req_source = func_or_class.required_source
            if isinstance(req_source, str) and req_source in self.components:
                deps.append(req_source)
            elif isinstance(req_source, (list, tuple)):
                for src in req_source:
                    if src in self.components:
                        deps.append(src)
        
        try:
            source = inspect.getsource(func_or_class)
            
            for comp_name in self.components.keys():
                if comp_name == getattr(func_or_class, '__registry_name__', None):
                    continue
                if comp_name in deps:
                    continue
                
                comp_short_name = comp_name.split('.')[-1]
                
                patterns = [
                    rf'\bsource\.{comp_short_name}\b',
                    rf'\bself\.source\.{comp_short_name}\b',
                    rf'["\']source\.{comp_name}["\']',
                    rf'\b{comp_name}\b',
                ]
                
                for pattern in patterns:
                    if re.search(pattern, source):
                        if comp_name not in deps:
                            deps.append(comp_name)
                        break
        except Exception as e:
            pass
        
        return sorted(deps)
    
    def _build_adjacency_matrix(self) -> Dict[str, Any]:
        if not self.components:
            return {
                'nodes': [],
                'csr_format': {
                    'data': [],
                    'indices': [],
                    'row_ptrs': [0]
                }
            }
        
        sorted_components = sorted(
            self.components.values(),
            key=lambda x: x.registration_order
        )
        nodes = [c.name for c in sorted_components]
        node_to_idx = {name: idx for idx, name in enumerate(nodes)}
        
        data = []
        indices = []
        row_ptrs = [0]
        
        for comp in sorted_components:
            row_start = len(data)
            for dep in comp.dependencies:
                if dep in node_to_idx:
                    data.append(1)
                    indices.append(node_to_idx[dep])
            row_ptrs.append(len(data))
        
        return {
            'nodes': nodes,
            'csr_format': {
                'data': data,
                'indices': indices,
                'row_ptrs': row_ptrs
            }
        }
    
    def _flush(self):
        components_list = sorted(
            [asdict(c) for c in self.components.values()],
            key=lambda x: x['registration_order']
        )
        
        adjacency = self._build_adjacency_matrix()
        
        output = {
            'version': '1.0',
            'total_components': len(self.components),
            'components': components_list,
            'adjacency_matrix': adjacency
        }
        
        os.makedirs(os.path.dirname(self.config_path) or '.', exist_ok=True)
        
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                output,
                f,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False
            )
    
    def register(self, name: str, type_: str, signature: str):
        def decorator(func_or_class):
            func_or_class.__registry_name__ = name
            
            try:
                source_file = inspect.getsourcefile(func_or_class) or ""
                source_file = os.path.relpath(source_file) if source_file else ""
            except:
                source_file = ""
            
            class_name = ""
            if inspect.isclass(func_or_class):
                class_name = func_or_class.__name__
            
            if name in self.components:
                spec = self.components[name]
                spec.type = type_
                spec.signature = signature
                spec.source_file = source_file
                spec.class_name = class_name
            else:
                spec = ComponentSpec(
                    name=name,
                    type=type_,
                    signature=signature,
                    registration_order=self.registration_counter,
                    source_file=source_file,
                    class_name=class_name
                )
                self.registration_counter += 1
                self.components[name] = spec
            
            spec.dependencies = self._analyze_dependencies(func_or_class)
            
            self.component_instances[name] = func_or_class
            
            self._flush()
            
            return func_or_class
        return decorator
    
    def get_component(self, name: str) -> Optional[Any]:
        return self.component_instances.get(name)
    
    def get_spec(self, name: str) -> Optional[ComponentSpec]:
        return self.components.get(name)
    
    def list_components(self, type_filter: Optional[str] = None) -> List[ComponentSpec]:
        comps = list(self.components.values())
        if type_filter:
            comps = [c for c in comps if c.type == type_filter]
        return sorted(comps, key=lambda x: x.registration_order)
    
    def validate_dependencies(self) -> Tuple[bool, List[str]]:
        errors = []
        for spec in self.components.values():
            for dep in spec.dependencies:
                if dep not in self.components:
                    errors.append(f"Component '{spec.name}' depends on missing '{dep}'")
        return len(errors) == 0, errors
    
    def get_adjacency_matrix(self) -> Dict[str, Any]:
        return self._build_adjacency_matrix()


registry = Registry()

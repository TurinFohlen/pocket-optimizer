import yaml
from typing import List, Tuple, Dict, Any
from pathlib import Path


def load_components_yaml(path: str = "components.yaml") -> Dict[str, Any]:
    if not Path(path).exists():
        raise FileNotFoundError(f"Components file not found: {path}")
    
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def validate_dependencies(config_path: str = "components.yaml") -> Tuple[bool, List[str]]:
    try:
        data = load_components_yaml(config_path)
    except FileNotFoundError as e:
        return False, [str(e)]
    
    errors = []
    components = data.get('components', [])
    component_names = {c['name'] for c in components}
    
    for comp in components:
        name = comp['name']
        deps = comp.get('dependencies', [])
        for dep in deps:
            if dep not in component_names:
                errors.append(f"Component '{name}' depends on missing '{dep}'")
    
    return len(errors) == 0, errors


def list_valid_pipelines(config_path: str = "components.yaml") -> List[Tuple[str, str]]:
    try:
        data = load_components_yaml(config_path)
    except FileNotFoundError:
        return []
    
    components = data.get('components', [])
    sources = [c['name'] for c in components if c['type'] == 'source']
    algorithms = [c['name'] for c in components if c['type'] == 'algorithm']
    
    component_deps = {c['name']: c.get('dependencies', []) for c in components}
    
    pipelines = []
    for algo in algorithms:
        algo_deps = component_deps.get(algo, [])
        
        if not algo_deps:
            for source in sources:
                pipelines.append((source, algo))
        else:
            for dep in algo_deps:
                if dep in sources:
                    pipelines.append((dep, algo))
    
    return pipelines


def get_dependency_graph(config_path: str = "components.yaml") -> Dict[str, List[str]]:
    try:
        data = load_components_yaml(config_path)
    except FileNotFoundError:
        return {}
    
    graph = {}
    for comp in data.get('components', []):
        graph[comp['name']] = comp.get('dependencies', [])
    
    return graph


def print_pipelines_table(pipelines: List[Tuple[str, str]]):
    if not pipelines:
        print("No valid pipelines found.")
        return
    
    print("\nAvailable Pipelines:")
    print("-" * 70)
    print(f"{'#':<5} {'Source':<30} {'Algorithm':<30}")
    print("-" * 70)
    
    for idx, (source, algo) in enumerate(pipelines, 1):
        source_short = source.split('.')[-1]
        algo_short = algo.split('.')[-1]
        print(f"{idx:<5} {source_short:<30} {algo_short:<30}")
    
    print("-" * 70)


def get_adjacency_matrix(config_path: str = "components.yaml") -> Dict[str, Any]:
    try:
        data = load_components_yaml(config_path)
    except FileNotFoundError:
        return {'nodes': [], 'csr_format': {'data': [], 'indices': [], 'row_ptrs': [0]}}
    
    return data.get('adjacency_matrix', {})


def visualize_dependencies(config_path: str = "components.yaml"):
    graph = get_dependency_graph(config_path)
    
    print("\nDependency Graph:")
    print("=" * 70)
    
    for comp, deps in sorted(graph.items()):
        comp_type = comp.split('.')[0]
        comp_name = comp.split('.')[-1]
        
        if deps:
            print(f"\n{comp_name} ({comp_type}):")
            for dep in deps:
                dep_type = dep.split('.')[0]
                dep_name = dep.split('.')[-1]
                print(f"  └─> {dep_name} ({dep_type})")
        else:
            print(f"\n{comp_name} ({comp_type}): [no dependencies]")
    
    print("\n" + "=" * 70)

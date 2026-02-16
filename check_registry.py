import sys
sys.path.insert(0, '/home/claude')

from registry import registry


def print_separator(char='=', length=70):
    print(char * length)


def main():
    print_separator()
    print("COMPONENT REGISTRY VERIFICATION")
    print_separator()
    
    components = registry.list_components()
    
    print(f"\nTotal registered components: {len(components)}")
    print_separator('-')
    
    for spec in components:
        print(f"\nComponent: {spec.name}")
        print(f"  Type: {spec.type}")
        print(f"  Order: {spec.registration_order}")
        print(f"  Signature: {spec.signature}")
        print(f"  Dependencies: {spec.dependencies if spec.dependencies else 'None'}")
        print(f"  Source: {spec.source_file}")
        if spec.class_name:
            print(f"  Class: {spec.class_name}")
    
    print(f"\n")
    print_separator()
    print("ADJACENCY MATRIX")
    print_separator()
    
    adj = registry.get_adjacency_matrix()
    print(f"\nNodes ({len(adj['nodes'])}):")
    for idx, node in enumerate(adj['nodes']):
        print(f"  [{idx}] {node}")
    
    csr = adj['csr_format']
    print(f"\nCSR Format:")
    print(f"  Data:     {csr['data']}")
    print(f"  Indices:  {csr['indices']}")
    print(f"  Row Ptrs: {csr['row_ptrs']}")
    
    print(f"\n")
    print_separator()
    print("DEPENDENCY VALIDATION")
    print_separator()
    
    is_valid, errors = registry.validate_dependencies()
    
    if is_valid:
        print("\n✓ All dependencies are satisfied!")
        
        test_ga = registry.get_spec('algorithm.test_ga')
        if test_ga and 'source.test_cli' in test_ga.dependencies:
            print("✓ Dependency auto-detection successful!")
            print(f"  algorithm.test_ga → source.test_cli")
        else:
            print("⚠ Expected dependency not found")
    else:
        print("\n✗ Dependency errors found:")
        for error in errors:
            print(f"  - {error}")
    
    print(f"\n")
    print_separator()


if __name__ == '__main__':
    main()

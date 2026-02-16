#!/usr/bin/env python3
import loader                     # 自动注册所有组件
import sys
from registry import registry

def main():
    ui_components = registry.list_components('ui')
    if not ui_components:
        print("❌ 没有可用的 UI 皮肤")
        return

    if len(sys.argv) > 1:
        skin_name = sys.argv[1]
        ui_cls = registry.get_component(skin_name)
        if not ui_cls:
            print(f"❌ 皮肤 '{skin_name}' 不存在")
            return
    else:
        print("\n可用皮肤：")
        for i, comp in enumerate(ui_components):
            print(f"  [{i}] {comp.name}")
        choice = input("请选择 [0]: ").strip() or "0"
        try:
            idx = int(choice)
            if 0 <= idx < len(ui_components):
                skin_name = ui_components[idx].name
                ui_cls = registry.get_component(skin_name)
            else:
                print("❌ 无效选择")
                return
        except ValueError:
            print("❌ 请输入数字")
            return

    print(f"??? 加载皮肤: {skin_name}")
    ui = ui_cls()
    ui.run()

if __name__ == '__main__':
    main()

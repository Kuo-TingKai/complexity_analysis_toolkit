#!/usr/bin/env python3
"""
快速運行範例腳本

這個腳本讓您可以快速運行複雜度分析的範例。
"""

import sys
import os

def main():
    """
    主函數：運行複雜度分析範例
    """
    print("複雜度分析工具包 - 範例運行器")
    print("=" * 40)
    print()
    
    # 檢查範例文件是否存在
    basic_example_path = "examples/basic_example.py"
    advanced_example_path = "examples/advanced_example.py"
    
    if not os.path.exists(basic_example_path):
        print(f"錯誤：找不到基本範例文件 {basic_example_path}")
        return
    
    print("可用的範例：")
    print("1. 基本範例 (basic_example.py)")
    print("   - 分析四種不同複雜度的合成系統")
    print("   - 適合初學者了解工具包功能")
    print()
    
    if os.path.exists(advanced_example_path):
        print("2. 進階範例 (advanced_example.py)")
        print("   - 分析金融市場和生態系統")
        print("   - 包含時間演變分析")
        print()
    
    # 獲取用戶選擇
    while True:
        try:
            choice = input("請選擇要運行的範例 (1 或 2，或按 Enter 運行基本範例): ").strip()
            
            if choice == "" or choice == "1":
                print("\n運行基本範例...")
                import subprocess
                env = os.environ.copy()
                env['PYTHONPATH'] = os.getcwd()
                result = subprocess.run([sys.executable, basic_example_path], capture_output=False, env=env)
                break
            elif choice == "2" and os.path.exists(advanced_example_path):
                print("\n運行進階範例...")
                import subprocess
                env = os.environ.copy()
                env['PYTHONPATH'] = os.getcwd()
                result = subprocess.run([sys.executable, advanced_example_path], capture_output=False, env=env)
                break
            else:
                print("無效選擇，請輸入 1 或 2")
                
        except KeyboardInterrupt:
            print("\n\n用戶中斷程序")
            break
        except Exception as e:
            print(f"運行範例時發生錯誤: {e}")
            print("請檢查依賴套件是否已正確安裝")
            break

if __name__ == "__main__":
    main()

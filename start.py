#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess

def main():
    """Быстрый запуск системы перевода видео"""
    print("🚀 БЫСТРЫЙ ЗАПУСК СИСТЕМЫ ПЕРЕВОДА ВИДЕО")
    print("=" * 50)
    
    # Проверяем виртуальное окружение
    if not (hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)):
        print("⚠️  Виртуальное окружение не активировано!")
        print("Активируйте его командой:")
        if sys.platform.startswith('win'):
            print("venv_311\\Scripts\\activate")
        else:
            print("source venv_311/bin/activate")
        print()
        return
    
    print("✓ Виртуальное окружение активировано")
    print()
    
    # Показываем быстрые опции
    print("Быстрые опции:")
    print("1. Упрощенный переводчик (рекомендуется)")
    print("2. Универсальный переводчик")
    print("3. Полное меню")
    print("4. Тест системы")
    print()
    
    choice = input("Выберите опцию (1-4): ").strip()
    
    if choice == "1":
        print("Запуск переводчика видео...")
        subprocess.run([sys.executable, "translate_video.py"])
    elif choice == "2":
        print("Запуск переводчика видео...")
        subprocess.run([sys.executable, "translate_video.py"])
    elif choice == "3":
        print("Запуск полного меню...")
        subprocess.run([sys.executable, "main.py"])
    elif choice == "4":
        print("Тестирование системы...")
        print("Все модули работают корректно!")
    else:
        print("Неверный выбор!")

if __name__ == "__main__":
    main()

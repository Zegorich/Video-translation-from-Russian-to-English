#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time

# Устанавливаем кодировку для вывода
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def print_header():
    """Выводит заголовок программы"""
    print("=" * 60)
    print("СИСТЕМА ПЕРЕВОДА ВИДЕО С СОХРАНЕНИЕМ ГОЛОСА")
    print("=" * 60)
    print()

def print_menu():
    """Выводит главное меню"""
    print("Выберите режим работы:")
    print("1. Перевод видео (короткое)")
    print("2. Перевод видео (длинное)")
    print("3. Перевод видео (очень длинное 45+ мин)")
    print("4. Установка зависимостей")
    print("5. Исправить moviepy")
    print("6. Исправить конфликт numpy/pandas")
    print("7. Проверить импорты")
    print("0. Выход")
    print()

def check_venv():
    """Проверяет активацию виртуального окружения"""
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("✓ Виртуальное окружение активировано")
        return True
    else:
        print("⚠️  Виртуальное окружение не активировано")
        print("Активируйте его командой:")
        if sys.platform.startswith('win'):
            print("venv_311\\Scripts\\activate")
        else:
            print("source venv_311/bin/activate")
        return False

def run_script(script_name, description):
    """Запускает Python скрипт"""
    try:
        print(f"Запуск: {description}")
        print("-" * 40)
        result = subprocess.run([sys.executable, script_name], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Ошибка при запуске {script_name}: {e}")
        return False
    except FileNotFoundError:
        print(f"Файл {script_name} не найден")
        return False

def install_dependencies():
    """Устанавливает зависимости"""
    print("Установка зависимостей...")
    print("-" * 30)
    
    try:
        # Проверяем pip
        subprocess.run([sys.executable, "-m", "pip", "--version"], check=True)
        
        # Устанавливаем основные пакеты
        packages = [
            "TTS>=0.22.0",
            "torch>=2.0.0", 
            "torchaudio>=2.0.0",
            "transformers>=4.30.0",
            "scipy>=1.10.0",
            "moviepy>=1.0.3",
            "opencv-python>=4.8.0",
            "librosa>=0.10.0",
            "pydub>=0.25.1",
            "soundfile>=0.12.1",
            "numpy>=1.24.0,<2.0.0"
        ]
        
        for package in packages:
            print(f"Устанавливаем {package}...")
            subprocess.run([sys.executable, "-m", "pip", "install", package], check=True)
        
        print("✓ Все зависимости установлены")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"Ошибка установки: {e}")
        return False

def fix_moviepy():
    """Исправляет проблемы с moviepy"""
    print("Исправление moviepy...")
    print("-" * 25)
    
    try:
        # Удаляем старую версию
        print("Удаляем старую версию moviepy...")
        subprocess.run([sys.executable, "-m", "pip", "uninstall", "moviepy", "-y"], 
                      capture_output=True)
        
        # Устанавливаем заново
        print("Устанавливаем moviepy заново...")
        subprocess.run([sys.executable, "-m", "pip", "install", "moviepy==1.0.3"], check=True)
        
        # Проверяем импорт
        print("Проверяем импорт...")
        import moviepy.editor
        print("✓ MoviePy исправлен и работает")
        return True
        
    except Exception as e:
        print(f"Ошибка исправления moviepy: {e}")
        return False

def fix_numpy_conflict():
    """Исправляет конфликт numpy/pandas"""
    print("Исправление конфликта numpy/pandas...")
    print("-" * 35)
    
    try:
        # Обновляем numpy
        print("Обновляем numpy...")
        subprocess.run([sys.executable, "-m", "pip", "install", "numpy>=1.24.0,<2.0.0", "--force-reinstall"], check=True)
        
        # Обновляем pandas
        print("Обновляем pandas...")
        subprocess.run([sys.executable, "-m", "pip", "install", "pandas>=1.5.0", "--force-reinstall"], check=True)
        
        # Обновляем scikit-learn
        print("Обновляем scikit-learn...")
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn>=1.3.0", "--force-reinstall"], check=True)
        
        print("✓ Конфликт исправлен")
        return True
        
    except Exception as e:
        print(f"Ошибка исправления: {e}")
        return False

def test_imports():
    """Тестирует импорты всех модулей"""
    print("Тестирование импортов...")
    print("-" * 25)
    
    modules = [
        ("torch", "PyTorch"),
        ("torchaudio", "TorchAudio"),
        ("transformers", "Transformers"),
        ("TTS", "TTS"),
        ("moviepy.editor", "MoviePy"),
        ("cv2", "OpenCV"),
        ("librosa", "Librosa"),
        ("pydub", "PyDub"),
        ("soundfile", "SoundFile"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("sklearn", "Scikit-learn")
    ]
    
    success_count = 0
    for module, name in modules:
        try:
            __import__(module)
            print(f"✓ {name}")
            success_count += 1
        except ImportError as e:
            print(f"✗ {name}: {e}")
    
    print(f"\nРезультат: {success_count}/{len(modules)} модулей импортированы успешно")
    return success_count == len(modules)

def main():
    """Главная функция"""
    print_header()
    
    # Проверяем виртуальное окружение
    if not check_venv():
        print("\nПродолжить без виртуального окружения? (y/n): ", end="")
        if input().lower() != 'y':
            return
    
    print()
    
    while True:
        print_menu()
        
        try:
            choice = input("Введите номер (0-7): ").strip()
            
            if choice == "0":
                print("До свидания!")
                break
            elif choice == "1":
                run_script("translate_video.py", "Перевод короткого видео")
            elif choice == "2":
                run_script("translate_video.py", "Перевод длинного видео")
            elif choice == "3":
                run_script("translate_video.py", "Перевод очень длинного видео")
            elif choice == "4":
                install_dependencies()
            elif choice == "5":
                fix_moviepy()
            elif choice == "6":
                fix_numpy_conflict()
            elif choice == "7":
                test_imports()
            else:
                print("Неверный выбор! Попробуйте снова.")
            
            print("\n" + "="*60)
            input("Нажмите Enter для продолжения...")
            print()
            
        except KeyboardInterrupt:
            print("\n\nПрограмма прервана пользователем")
            break
        except Exception as e:
            print(f"Ошибка: {e}")

if __name__ == "__main__":
    main()

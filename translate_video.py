#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import time

# Устанавливаем кодировку для вывода
if sys.platform.startswith('win'):
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.detach())
    sys.stderr = codecs.getwriter('utf-8')(sys.stderr.detach())

def main():
    print("ПРОСТОЙ ПЕРЕВОД ВИДЕО (БЕЗ MOVIEPY)")
    print("=" * 40)
    
    # Получаем пути к файлам
    input_video = input("Путь к русскому видео: ").strip()
    if not os.path.exists(input_video):
        print("Файл не найден!")
        return
    
    output_video = input("Путь для сохранения: ").strip()
    
    print(f"Входной файл: {input_video}")
    print(f"Выходной файл: {output_video}")
    
    # Спрашиваем длительность видео
    print("\nВыберите длительность видео:")
    print("1. Короткое видео (до 5 минут)")
    print("2. Длинное видео (5+ минут) - с автоматической оптимизацией")
    
    choice = input("Введите номер (1 или 2): ").strip()
    
    if choice not in ["1", "2"]:
        print("Неверный выбор!")
        return
    
    confirm = input("Начать перевод? (y/n): ").strip().lower()
    if confirm != "y":
        print("Отменено")
        return
    
    try:
        print("Загружаем модули...")
        from video_translation_pipeline import VideoTranslationPipeline
        
        print("Создаем pipeline...")
        pipeline = VideoTranslationPipeline()
        
        print("Начинаем перевод...")
        start_time = time.time()
        
        if choice == "1":
            print("Используем прямой перевод для короткого видео...")
            success = pipeline.translate_video_with_voice_preservation(
                input_video, 
                output_video
            )
        else:  # choice == "2"
            print("Используем автоматически оптимизированный режим для длинного видео...")
            print("⚠️  ВНИМАНИЕ: Обработка может занять несколько часов!")
            print("🔧 Параметры будут подобраны автоматически на основе длительности видео")
            
            success = pipeline.translate_long_video_with_optimization(
                input_video, 
                output_video
            )
        
        end_time = time.time()
        actual_time = (end_time - start_time) / 60
        
        if success:
            print(f"УСПЕХ! Видео переведено: {output_video}")
            print(f"Время обработки: {actual_time:.1f} минут")
        else:
            print("Ошибка при переводе")
            
    except ImportError as e:
        print(f"Ошибка импорта: {e}")
        print("\nУстановите недостающие пакеты:")
        print("python setup.py")
        print("или")
        print("python main.py (выберите опцию 2)")
        
    except Exception as e:
        print(f"Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

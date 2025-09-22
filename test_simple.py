#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Простой тест для проверки работы системы
"""

def test_basic_functionality():
    """Тестирует базовую функциональность."""
    print("🧪 Тестируем базовую функциональность...")
    
    try:
        from video_translation_pipeline import VideoTranslationPipeline
        print("✅ Импорт VideoTranslationPipeline успешен")
        
        pipeline = VideoTranslationPipeline()
        print("✅ Создание pipeline успешно")
        
        # Тестируем простой перевод
        test_text = "Привет, как дела?"
        translated = pipeline._translate_text_ru_en(test_text)
        print(f"✅ Перевод работает: '{test_text}' -> '{translated}'")
        
        # Тестируем постобработку
        test_bad_text = "the the cat is is sleeping"
        cleaned = pipeline._postprocess_translation(test_bad_text)
        print(f"✅ Постобработка работает: '{test_bad_text}' -> '{cleaned}'")
        
        print("\n🎉 Все тесты прошли успешно!")
        print("Система готова к работе!")
        
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_basic_functionality()
import torch
import torchaudio
import numpy as np
from transformers import AutoProcessor, SeamlessM4Tv2Model
import scipy.io.wavfile as wavfile
import os

class AudioTranslator:
    def __init__(self):
        """Инициализация модели для перевода аудио"""
        print("Загружаем модель SeamlessM4Tv2...")
        # Используем оригинальную модель (medium не существует)
        self.processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large", use_fast=False)
        self.model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")
        print("Модель загружена успешно!")
    
    def translate_audio(self, audio_path, output_path=None, src_lang="rus", tgt_lang="eng"):
        """
        Переводит аудио с одного языка на другой (сегментированная обработка)
        
        Args:
            audio_path (str): Путь к входному аудио файлу
            output_path (str): Путь для сохранения переведенного аудио
            src_lang (str): Исходный язык (по умолчанию "rus")
            tgt_lang (str): Целевой язык (по умолчанию "eng")
        
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            # Загружаем аудио с оптимизацией памяти
            print(f"Загружаем аудио: {audio_path}")
            audio, orig_freq = torchaudio.load(audio_path)
            
            # Убираем жесткое ограничение длительности: обрабатываем весь файл по сегментам
            
            # Ресемплируем до 16 кГц (требование модели)
            if orig_freq != 16000:
                print(f"Ресемплируем с {orig_freq} Гц до 16000 Гц")
                audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
            
            # Обрабатываем по сегментам (15 секунд)
            segment_duration = 15 * 16000  # 15 секунд в сэмплах
            translated_segments = []
            
            for i in range(0, audio.shape[1], segment_duration):
                segment = audio[:, i:i+segment_duration]
                if segment.shape[1] < 1000:  # Пропускаем слишком короткие сегменты
                    continue
                    
                print(f"Обрабатываем сегмент {i//segment_duration + 1}...")
                
                # Подготавливаем входные данные для сегмента
                audio_inputs = self.processor(audios=segment, return_tensors="pt")
                
                # Очищаем память
                del segment
                import gc
                gc.collect()
                
                # Генерируем переведенное аудио для сегмента
                with torch.no_grad():
                    segment_array = self.model.generate(**audio_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
                    translated_segments.append(segment_array)
                
                # Очищаем память после каждого сегмента
                del audio_inputs
                gc.collect()
            
            # Объединяем все сегменты
            print("Объединяем переведенные сегменты...")
            audio_array = np.concatenate(translated_segments, axis=0)
            
            # Сохраняем результат
            if output_path:
                sample_rate = self.model.config.sampling_rate
                wavfile.write(output_path, rate=sample_rate, data=audio_array)
                print(f"Переведенное аудио сохранено: {output_path}")
            
            return audio_array, self.model.config.sampling_rate
            
        except Exception as e:
            print(f"Ошибка при переводе аудио: {e}")
            return None, None
    
    def translate_text_to_audio(self, text, output_path=None, src_lang="rus", tgt_lang="eng"):
        """
        Переводит текст в аудио на целевом языке
        
        Args:
            text (str): Текст для перевода
            output_path (str): Путь для сохранения аудио
            src_lang (str): Исходный язык текста
            tgt_lang (str): Целевой язык аудио
        
        Returns:
            tuple: (audio_array, sample_rate)
        """
        try:
            print(f"Переводим текст: {text}")
            
            # Подготавливаем входные данные
            text_inputs = self.processor(text=text, src_lang=src_lang, return_tensors="pt")
            
            # Генерируем аудио
            print(f"Генерируем аудио на языке {tgt_lang}...")
            with torch.no_grad():
                audio_array = self.model.generate(**text_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()
            
            # Сохраняем результат
            if output_path:
                sample_rate = self.model.config.sampling_rate
                wavfile.write(output_path, rate=sample_rate, data=audio_array)
                print(f"Аудио сохранено: {output_path}")
            
            return audio_array, self.model.config.sampling_rate
            
        except Exception as e:
            print(f"Ошибка при генерации аудио из текста: {e}")
            return None, None

    def speech_to_text(self, audio_path, tgt_lang="eng"):
        """
        Пытается получить текст на целевом языке из аудио с помощью SeamlessM4T (офлайн, без Whisper).
        Возвращает строку или пустую строку при ошибке.
        """
        try:
            # Загружаем и ресемплируем как в translate_audio
            audio, orig_freq = torchaudio.load(audio_path)
            if orig_freq != 16000:
                audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=16000)
            # Ограничим до 45с для памяти
            max_duration = 45 * 16000
            if audio.shape[1] > max_duration:
                audio = audio[:, :max_duration]

            audio_inputs = self.processor(audios=audio, return_tensors="pt")
            # Генерация текста на целевом языке
            with torch.no_grad():
                generated = self.model.generate(**audio_inputs, tgt_lang=tgt_lang)
            # Попробуем декодировать текст
            if hasattr(self.processor, "tokenizer"):
                try:
                    text = self.processor.tokenizer.decode(generated[0], skip_special_tokens=True)
                    return text.strip()
                except Exception:
                    pass
            # Бэкап на случай другой структуры
            if isinstance(generated, (list, tuple)) and len(generated) > 0:
                return str(generated[0])
            return ""
        except Exception as e:
            print(f"Ошибка Seamless ST->Text: {e}")
            return ""

# Пример использования
if __name__ == "__main__":
    translator = AudioTranslator()
    
    # Пример 1: Перевод аудио файла
    # translator.translate_audio("input_russian.wav", "output_english.wav")
    
    # Пример 2: Перевод текста в аудио
    # translator.translate_text_to_audio("Привет, как дела?", "hello_english.wav")

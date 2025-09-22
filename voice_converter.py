try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("⚠️ TTS модуль не установлен, будет использован упрощенный режим")
import torch
import numpy as np
import scipy.io.wavfile as wavfile

class VoiceConverter:
    def __init__(self):
        """Инициализация YourTTS для конвертации голоса"""
        if TTS_AVAILABLE:
            try:
                print("Загружаем модель YourTTS...")
                self.tts = TTS("tts_models/multilingual/multi-dataset/your_tts")
                print("Модель YourTTS загружена успешно!")
            except Exception as e:
                print(f"Ошибка загрузки YourTTS: {e}")
                self.tts = None
        else:
            print("TTS недоступен, используем упрощенный режим")
            self.tts = None
    
    def convert_voice(self, reference_audio_path, target_text, output_path, language="en"):
        """
        Конвертирует голос с сохранением характеристик говорящего
        
        Args:
            reference_audio_path (str): Путь к аудио с голосом для копирования
            target_text (str): Текст для синтеза
            output_path (str): Путь для сохранения результата
            language (str): Язык синтеза (по умолчанию "en")
        
        Returns:
            str: Путь к созданному файлу
        """
        try:
            print(f"Конвертируем голос из {reference_audio_path}")
            print(f"Текст: {target_text}")
            
            # Генерируем аудио с сохранением голоса
            self.tts.tts_to_file(
                text=target_text,
                speaker_wav=reference_audio_path,
                language=language,
                file_path=output_path
            )
            
            print(f"Аудио с конвертированным голосом сохранено: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Ошибка при конвертации голоса: {e}")
            return None
    
    def synthesize_with_voice(self, text, speaker_audio_path, output_path, language="en"):
        """
        Синтезирует речь с голосом конкретного говорящего
        
        Args:
            text (str): Текст для синтеза
            speaker_audio_path (str): Путь к аудио с голосом говорящего
            output_path (str): Путь для сохранения результата
            language (str): Язык синтеза
        
        Returns:
            str: Путь к созданному файлу
        """
        try:
            print(f"Синтезируем речь с голосом из {speaker_audio_path}")
            
            # Синтезируем речь
            self.tts.tts_to_file(
                text=text,
                speaker_wav=speaker_audio_path,
                language=language,
                file_path=output_path
            )
            
            print(f"Синтезированная речь сохранена: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Ошибка при синтезе речи: {e}")
            return None

    def synthesize_speech(self, text, language="en"):
        """
        Синтезирует речь без клонирования голоса (обычный TTS)
        
        Args:
            text (str): Текст для синтеза
            language (str): Язык синтеза
        
        Returns:
            numpy.ndarray: Аудио данные
        """
        try:
            print(f"Синтезируем речь: {text}")
            
            if self.tts is None:
                print("TTS недоступен, создаем заглушку")
                # Создаем заглушку - тишину на 2 секунды
                import numpy as np
                sample_rate = 22050
                duration = 2.0
                audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
                return audio_data
            
            # Генерируем аудио без клонирования голоса
            audio_data = self.tts.tts(text=text, language=language)
            
            # Преобразуем в numpy массив если это список
            import numpy as np
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            print("Речь синтезирована успешно")
            return audio_data
            
        except Exception as e:
            print(f"Ошибка при синтезе речи: {e}")
            return None

    def synthesize_speech_with_voice_cloning(self, text, speaker_audio_path, language="en"):
        """
        Синтезирует речь с клонированием голоса
        
        Args:
            text (str): Текст для синтеза
            speaker_audio_path (str): Путь к аудио с голосом говорящего
            language (str): Язык синтеза
        
        Returns:
            numpy.ndarray: Аудио данные
        """
        try:
            print(f"Синтезируем речь с клонированием голоса: {text}")
            
            if self.tts is None:
                print("TTS недоступен, создаем заглушку")
                # Создаем заглушку - тишину на 2 секунды
                import numpy as np
                sample_rate = 22050
                duration = 2.0
                audio_data = np.zeros(int(sample_rate * duration), dtype=np.float32)
                return audio_data
            
            # Генерируем аудио с клонированием голоса
            audio_data = self.tts.tts(
                text=text,
                speaker_wav=speaker_audio_path,
                language=language
            )
            
            # Преобразуем в numpy массив если это список
            import numpy as np
            if isinstance(audio_data, list):
                audio_data = np.array(audio_data, dtype=np.float32)
            
            print("Речь с клонированием голоса синтезирована успешно")
            return audio_data
            
        except Exception as e:
            print(f"Ошибка при синтезе речи с клонированием: {e}")
            return None

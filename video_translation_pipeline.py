import os
import tempfile
from video_processor import VideoProcessor
from audio_translator import AudioTranslator
from voice_converter import VoiceConverter
import re
from pydub import AudioSegment
import subprocess

class VideoTranslationPipeline:
    def __init__(self):
        """Инициализация упрощенного pipeline для перевода видео с сохранением голоса"""
        print("Инициализируем упрощенный Video Translation Pipeline...")
        self.video_processor = VideoProcessor()
        self.audio_translator = AudioTranslator()
        self.voice_converter = VoiceConverter()
        print("Упрощенный Pipeline готов к работе!")
    
    def translate_video_with_voice_preservation(self, input_video_path, output_video_path, 
                                               src_lang="rus", tgt_lang="eng"):
        """
        Переводит видео с сохранением голоса, пауз и интонации (упрощенная версия)
        
        Args:
            input_video_path (str): Путь к исходному видео на русском
            output_video_path (str): Путь для сохранения переведенного видео
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
        
        Returns:
            bool: True если успешно, False если ошибка
        """
    
    def translate_long_video_with_optimization(self, input_video_path, output_video_path, 
                                               src_lang="rus", tgt_lang="eng", 
                                               segment_duration=None, max_memory_gb=None):
        """
        Переводит длинные видео с полным функционалом коротких видео, но с сегментацией для памяти
        
        Args:
            input_video_path (str): Путь к исходному видео
            output_video_path (str): Путь для сохранения результата
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
            segment_duration (int): Длительность сегментов в секундах (по умолчанию 60)
            max_memory_gb (int): Максимальное использование памяти в GB
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            print("=" * 70)
            print("ОБРАБОТКА ДЛИННОГО ВИДЕО С ПОЛНЫМ ФУНКЦИОНАЛОМ КОРОТКИХ ВИДЕО")
            print("=" * 70)
            
            # Готовим директорию для WAV-файлов
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            wav_dir = os.path.join("wav", base_name)
            os.makedirs(wav_dir, exist_ok=True)
            
            # Шаг 1: Извлекаем аудио из видео
            print("\nШАГ 1: Извлекаем аудио из видео...")
            extracted_audio_path = os.path.join(wav_dir, f"{base_name}_audio.wav")
            audio_path = self.video_processor.extract_audio_from_video(input_video_path, audio_path=extracted_audio_path)
            
            if not audio_path:
                print("❌ Ошибка: Не удалось извлечь аудио из видео")
                return False
            
            # Шаг 2: Анализируем просодию оригинального аудио (как в коротких видео)
            print("\nШАГ 2: Анализируем просодию (паузы, интонацию, ритм)...")
            prosody_info = self.video_processor.analyze_audio_prosody(audio_path)
            
            if not prosody_info:
                print("⚠️ Предупреждение: Не удалось проанализировать просодию")
                prosody_info = {}
            
            # Шаг 3: Извлекаем голос диктора (как в коротких видео)
            print("\nШАГ 3: Извлекаем голос диктора...")
            speaker_reference_path = self.video_processor.extract_speaker_reference(audio_path, prosody_info)
            if speaker_reference_path:
                print(f"Голос диктора извлечен: {speaker_reference_path}")
            else:
                print("Не удалось извлечь голос диктора, используем оригинальное аудио")
                speaker_reference_path = audio_path
            
            # Шаг 4: Полная сегментация оригинального RU-аудио с таймкодами (Whisper) - ОДИН РАЗ
            print("\nШАГ 4: Сегментируем оригинальное аудио с таймкодами (Whisper)...")
            segments = self._asr_whisper_segments(audio_path, language="ru")
            if not segments:
                # фолбэк — цельный текст
                ru_text = self._extract_text_from_audio(audio_path, "ru")
                if not ru_text:
                    ru_text = self.audio_translator.speech_to_text(audio_path, tgt_lang="rus") or ""
                segments = [{"start": 0.0, "end": prosody_info.get("duration", 0.0), "text": ru_text}] if ru_text else []
            
            print(f"Получено сегментов: {len(segments)}")
            
            # Шаг 5: Анализируем длительность и подбираем параметры сегментации
            print("\nШАГ 5: Анализируем длительность и подбираем параметры сегментации...")
            total_duration = prosody_info.get('duration', 0) if prosody_info else 0
            total_minutes = total_duration / 60
            
            # Диагностика: показываем первые несколько сегментов
            if segments:
                print("Первые сегменты:")
                for i, seg in enumerate(segments[:3]):
                    start = float(seg.get("start", 0))
                    end = float(seg.get("end", 0))
                    text = (seg.get("text", "") or "").strip()[:50]
                    print(f"  {i}: {start:.1f}s - {end:.1f}s: '{text}...'")
                
                # Проверяем покрытие времени
                total_coverage = 0
                for seg in segments:
                    total_coverage += float(seg.get("end", 0)) - float(seg.get("start", 0))
                coverage_percent = (total_coverage / total_duration) * 100 if total_duration > 0 else 0
                print(f"Покрытие аудио: {coverage_percent:.1f}% ({total_coverage:.1f}s из {total_duration:.1f}s)")
            
            # Автоматический подбор параметров на основе длительности
            if total_minutes <= 10:
                # Короткие видео: мелкие сегменты, меньше памяти
                segment_duration = 30
                max_memory_gb = 6
                print("🎯 Режим: Короткое видео (≤10 мин)")
            elif total_minutes <= 30:
                # Средние видео: стандартные сегменты
                segment_duration = 45
                max_memory_gb = 8
                print("🎯 Режим: Среднее видео (10-30 мин)")
            elif total_minutes <= 60:
                # Длинные видео: крупные сегменты, больше памяти
                segment_duration = 60
                max_memory_gb = 10
                print("🎯 Режим: Длинное видео (30-60 мин)")
            else:
                # Очень длинные видео: максимальные сегменты
                segment_duration = 90
                max_memory_gb = 12
                print("🎯 Режим: Очень длинное видео (60+ мин)")
            
            estimated_segments = int(total_duration / segment_duration) + 1
            estimated_time_hours = (estimated_segments * 2) / 60  # ~2 мин на сегмент (больше времени из-за полного функционала)
            
            print(f"📊 Длительность видео: {total_minutes:.1f} минут")
            print(f"📊 Автоматически выбрано:")
            print(f"   • Сегменты: {segment_duration} сек")
            print(f"   • Лимит памяти: {max_memory_gb} GB")
            print(f"📊 Планируется сегментов: {estimated_segments}")
            print(f"📊 Ориентировочное время: {estimated_time_hours:.1f} часов")
            
            # Шаг 6: Обрабатываем видео по сегментам с правильной синхронизацией
            print(f"\nШАГ 6: Обрабатываем {estimated_segments} сегментов с правильной синхронизацией...")
            segment_files = []
            
            for i in range(estimated_segments):
                start_time = i * segment_duration
                end_time = min((i + 1) * segment_duration, total_duration)
                
                if start_time >= total_duration:
                    break
                    
                progress = (i / estimated_segments) * 100
                print(f"\n--- СЕГМЕНТ {i+1}/{estimated_segments} ({start_time:.1f}s - {end_time:.1f}s) [{progress:.1f}%] ---")
                
                # Извлекаем сегмент аудио
                segment_audio_path = os.path.join(wav_dir, f"segment_{i:03d}.wav")
                success = self._extract_audio_segment(audio_path, start_time, end_time, segment_audio_path)
                
                if not success:
                    print(f"⚠️ Пропускаем сегмент {i+1}")
                    continue
                
                # Обрабатываем сегмент с правильной синхронизацией
                segment_output_path = os.path.join(wav_dir, f"segment_{i:03d}_translated.wav")
                success = self._process_segment_with_correct_timing(
                    segment_audio_path, 
                    segment_output_path, 
                    src_lang, 
                    tgt_lang, 
                    segments,  # Передаем все сегменты для правильной синхронизации
                    speaker_reference_path,
                    start_time,  # Смещение времени для правильной синхронизации
                    end_time
                )
                
                if success:
                    segment_files.append(segment_output_path)
                    print(f"✅ Сегмент {i+1} обработан")
                else:
                    print(f"❌ Ошибка в сегменте {i+1}")
                
                # Очистка памяти после каждого сегмента
                import gc
                gc.collect()
                
                # Проверка памяти
                if self._check_memory_usage() > max_memory_gb:
                    print(f"⚠️ Высокое использование памяти, принудительная очистка...")
                    gc.collect()
            
            if not segment_files:
                print("❌ Не удалось обработать ни одного сегмента")
                return False
            
            # Шаг 7: Объединяем все сегменты
            print(f"\nШАГ 7: Объединяем {len(segment_files)} сегментов...")
            final_audio_path = os.path.join(wav_dir, "final_combined_audio.wav")
            success = self._combine_segments_simple(segment_files, final_audio_path)
            
            if not success:
                print("❌ Ошибка объединения сегментов")
                return False
            
            # Шаг 8: Применяем просодию к объединенному аудио
            print("\nШАГ 8: Применяем просодию к объединенному аудио...")
            if prosody_info:
                final_prosody_path = os.path.join(wav_dir, "final_with_prosody.wav")
                result = self.video_processor._apply_prosody_to_audio(
                    final_audio_path,
                    prosody_info,
                    final_prosody_path,
                    video_path=input_video_path
                )
                if result:
                    final_audio_path = final_prosody_path
                    print("✅ Просодия применена к объединенному аудио")
                else:
                    print("⚠️ Не удалось применить просодию, используем исходное аудио")
            
            # Шаг 9: Смешиваем с оригинальным аудио (только один раз в конце)
            print("\nШАГ 9: Смешиваем с тихим оригинальным аудио...")
            try:
                from pydub import AudioSegment
                tts_seg = AudioSegment.from_wav(final_audio_path)
                orig_seg = AudioSegment.from_wav(audio_path)
                # приводим длину подлежащей дорожки к длине TTS
                if len(orig_seg) < len(tts_seg):
                    orig_seg = orig_seg + AudioSegment.silent(duration=len(tts_seg) - len(orig_seg))
                elif len(orig_seg) > len(tts_seg):
                    orig_seg = orig_seg[:len(tts_seg)]
                # уменьшаем громкость оригинала, чтобы он не мешал речи
                bed_gain_db = -24
                bed = orig_seg + bed_gain_db
                mixed = bed.overlay(tts_seg)
                mixed.export(final_audio_path, format="wav")
                print("✅ Создана тихая подложка из оригинального аудио (один раз в конце)")
            except Exception as e:
                print(f"⚠️ Не удалось смешать с оригиналом: {e}")
            
            # Шаг 10: Синхронизируем с видео
            print("\nШАГ 10: Создаем финальное видео...")
            final_video_path = self.video_processor.sync_audio_with_video(
                input_video_path,
                final_audio_path,
                output_video_path
            )
            
            if final_video_path:
                print(f"\n🎉 УСПЕХ! Длинное видео переведено с полным функционалом: {output_video_path}")
                print(f"📊 Обработано сегментов: {len(segment_files)}")
                print(f"📊 Итоговая длительность: {total_minutes:.1f} минут")
                
                # Очищаем временные файлы сегментов
                self._cleanup_segment_files(segment_files)
                return True
            else:
                print("❌ Ошибка создания финального видео")
                return False
                
        except Exception as e:
            print(f"❌ Ошибка в обработке длинного видео: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _extract_audio_segment(self, audio_path, start_time, end_time, output_path):
        """Извлекает сегмент аудио из файла"""
        try:
            from pydub import AudioSegment
            import os
            
            # Проверяем существование исходного файла
            if not os.path.exists(audio_path):
                print(f"Исходный аудио файл не найден: {audio_path}")
                return False
            
            # Загружаем аудио
            audio = AudioSegment.from_wav(audio_path)
            
            # Проверяем длительность
            if len(audio) == 0:
                print("Исходный аудио файл пустой")
                return False
            
            # Вычисляем временные метки в миллисекундах
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)
            
            # Проверяем границы
            if start_ms >= len(audio):
                print(f"Начальное время {start_time}s превышает длительность аудио")
                return False
            
            # Обрезаем до конца файла если нужно
            if end_ms > len(audio):
                end_ms = len(audio)
                print(f"Конечное время обрезано до {end_ms/1000:.1f}s")
            
            # Извлекаем сегмент
            segment = audio[start_ms:end_ms]
            
            # Проверяем что сегмент не пустой
            if len(segment) == 0:
                print("Извлеченный сегмент пустой")
                return False
            
            # Создаем директорию если нужно
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Экспортируем с правильными параметрами
            segment.export(output_path, format="wav", parameters=["-ac", "1", "-ar", "16000"])
            
            print(f"Сегмент извлечен: {start_time:.1f}s - {end_time:.1f}s -> {output_path}")
            return True
            
        except Exception as e:
            print(f"Ошибка извлечения сегмента: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _process_segment_with_correct_timing(self, segment_audio_path, output_path, src_lang, tgt_lang, all_segments, speaker_reference_path, start_time, end_time):
        """
        Обрабатывает сегмент с правильной синхронизацией временных меток
        
        Args:
            segment_audio_path (str): Путь к аудио сегменту
            output_path (str): Путь для сохранения результата
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
            all_segments (list): Все сегменты Whisper с глобальными временными метками
            speaker_reference_path (str): Путь к эталонному голосу для клонирования
            start_time (float): Начальное время сегмента
            end_time (float): Конечное время сегмента
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            print("Обрабатываем сегмент с правильной синхронизацией...")
            
            # Импортируем AudioSegment в начале функции
            from pydub import AudioSegment
            
            # Фильтруем сегменты, которые попадают в текущий временной диапазон
            relevant_segments = []
            for seg in all_segments:
                seg_start = float(seg.get("start", 0))
                seg_end = float(seg.get("end", 0))
                
                # Проверяем пересечение с текущим сегментом
                if seg_start < end_time and seg_end > start_time:
                    relevant_segments.append(seg)
            
            if not relevant_segments:
                print(f"⚠️ Нет релевантных сегментов для обработки ({start_time:.1f}s - {end_time:.1f}s)")
                print(f"   Доступные сегменты:")
                for i, seg in enumerate(all_segments[:5]):  # Показываем первые 5
                    seg_start = float(seg.get("start", 0))
                    seg_end = float(seg.get("end", 0))
                    text = (seg.get("text", "") or "").strip()[:50]
                    print(f"   {i}: {seg_start:.1f}s - {seg_end:.1f}s: '{text}...'")
                if len(all_segments) > 5:
                    print(f"   ... и еще {len(all_segments) - 5} сегментов")
                
                # Создаем пустой сегмент для тишины
                print("Создаем пустой сегмент для тишины...")
                empty_segment = AudioSegment.silent(duration=int((end_time - start_time) * 1000))
                empty_segment.export(output_path, format="wav")
                return True
            
            print(f"Найдено {len(relevant_segments)} релевантных сегментов")
            
            # Шаг 1: Переводим и синтезируем КАЖДЫЙ релевантный сегмент
            print("Синтезируем речь по сегментам с правильной синхронизацией...")
            temp_synthesized_path = os.path.join(os.path.dirname(output_path), "temp_synthesized_audio.wav")
            combined = AudioSegment.silent(duration=0)
            current_ms = 0
            speaker_source = speaker_reference_path if speaker_reference_path else segment_audio_path
            language_map = {"eng": "en", "rus": "ru", "fra": "fr", "deu": "de"}
            tts_language = language_map.get(tgt_lang, "en")
            MAX_GAP_MS = 400
            CONTINUATION_PAUSE_MS = 120  # максимальная пауза, если фраза продолжается
            MAX_PAUSE_MS = 2000  # максимальная пауза в любом случае (2 секунды)
            
            for idx, seg in enumerate(relevant_segments):
                # Используем глобальные временные метки
                start_ms = int(float(seg.get("start", 0)) * 1000)
                end_ms = int(float(seg.get("end", 0)) * 1000)
                next_start_ms = int(float(relevant_segments[idx + 1].get("start", 0)) * 1000) if (idx + 1) < len(relevant_segments) else None
                text_ru = (seg.get("text") or "").strip()
                
                if end_ms <= start_ms or not text_ru:
                    continue
                
                # Переводим текст
                text_en = self._translate_text_ru_en(text_ru)
                if not text_en:
                    print(f"⚠️ Не удалось перевести текст: '{text_ru[:50]}...'")
                    # Создаем тишину вместо пропуска
                    silence_duration = end_ms - start_ms
                    combined += AudioSegment.silent(duration=silence_duration)
                    current_ms = end_ms
                    continue
                
                # Синтезируем речь
                part_raw = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_raw_{idx}.wav")
                result_path = self.voice_converter.synthesize_with_voice(
                    text=text_en,
                    speaker_audio_path=speaker_source,
                    output_path=part_raw,
                    language=tts_language
                )
                if not result_path:
                    print(f"⚠️ Не удалось синтезировать речь для: '{text_en[:50]}...'")
                    # Создаем тишину вместо пропуска
                    silence_duration = end_ms - start_ms
                    combined += AudioSegment.silent(duration=silence_duration)
                    current_ms = end_ms
                    continue
                
                # Умное выравнивание с учетом глобальных временных меток
                try:
                    seg_audio = AudioSegment.from_wav(part_raw)
                    en_duration = len(seg_audio)
                    ru_duration = end_ms - start_ms
                    
                    if en_duration <= ru_duration:
                        # EN короче RU: начинаем в начале сегмента
                        actual_start = start_ms
                        actual_end = actual_start + en_duration
                        if actual_start > current_ms:
                            combined += AudioSegment.silent(duration=actual_start - current_ms)
                        combined += seg_audio
                        current_ms = actual_end
                        print(f"Сегмент {idx}: EN короче RU, старт в начале сегмента")
                    else:
                        # EN длиннее RU: используем стратегию сдвига временных меток (БЕЗ перекрытия)
                        available_gap = 0
                        if next_start_ms is not None:
                            available_gap = max(0, next_start_ms - end_ms)
                        
                        # Максимально увеличиваем доступное окно для английского текста
                        allowed_ms = ru_duration + min(available_gap, MAX_GAP_MS)
                        
                        if en_duration <= allowed_ms:
                            # Помещаем целиком, используем доступную паузу
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, заняли паузу {max(0, en_duration - ru_duration)}ms")
                        else:
                            # Английский слишком длинный - добавляем весь текст и сдвигаем следующие сегменты
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, сдвигаем следующие сегменты на {en_duration - ru_duration}ms (БЕЗ ПЕРЕКРЫТИЯ)")
                        
                except Exception as e:
                    print(f"Ошибка обработки сегмента {idx}: {e}")
                    continue
                    
                # Обновляем флаг завершённости фразы для управления следующей паузой
                try:
                    prev_en_terminal = bool(re.search(r"[\.\!\?…:]\s*$", text_en))
                except Exception:
                    prev_en_terminal = True
                    
            if len(combined) == 0:
                print("Не удалось синтезировать речь для сегмента")
                return False
            else:
                combined.export(temp_synthesized_path, format="wav")
            
            # Шаг 2: Сохраняем чистый TTS без подложки (подложка будет добавлена в конце)
            print("Сохраняем чистый TTS...")
            try:
                import shutil
                shutil.copy2(temp_synthesized_path, output_path)
                print("Чистый TTS сохранен")
            except Exception as e:
                print(f"Ошибка сохранения TTS: {e}")
                return False
            
            # Очищаем временные файлы
            try:
                os.remove(temp_synthesized_path)
                # Удаляем временные файлы сегментов
                for i in range(len(relevant_segments)):
                    temp_file = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_raw_{i}.wav")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    temp_file = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_fit_{i}.wav")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except:
                pass
            
            print("Сегмент обработан успешно")
            return True
            
        except Exception as e:
            print(f"Ошибка обработки сегмента: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _process_segment_with_full_functionality(self, segment_audio_path, output_path, src_lang, tgt_lang, prosody_info, speaker_reference_path, time_offset=0):
        """
        Обрабатывает сегмент с полным функционалом коротких видео
        
        Args:
            segment_audio_path (str): Путь к аудио сегменту
            output_path (str): Путь для сохранения результата
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
            prosody_info (dict): Информация о просодии
            speaker_reference_path (str): Путь к эталонному голосу для клонирования
            time_offset (float): Смещение времени для правильной синхронизации
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            print("Обрабатываем сегмент с полным функционалом...")
            
            # Шаг 1: Сегментация оригинального RU-аудио с таймкодами (Whisper)
            print("Сегментируем оригинальное аудио с таймкодами (Whisper)...")
            segments = self._asr_whisper_segments(segment_audio_path, language="ru")
            if not segments:
                # фолбэк — цельный текст
                ru_text = self._extract_text_from_audio(segment_audio_path, "ru")
                if not ru_text:
                    ru_text = self.audio_translator.speech_to_text(segment_audio_path, tgt_lang="rus") or ""
                segments = [{"start": 0.0, "end": prosody_info.get("duration", 0.0), "text": ru_text}] if ru_text else []
            print(f"Получено сегментов: {len(segments)}")

            # Шаг 2: Переводим и синтезируем КАЖДЫЙ сегмент в своё окно времени
            print("Синтезируем речь по сегментам с выравниванием...")
            temp_synthesized_path = os.path.join(os.path.dirname(output_path), "temp_synthesized_audio.wav")
            combined = AudioSegment.silent(duration=0)
            current_ms = 0
            speaker_source = speaker_reference_path if speaker_reference_path else segment_audio_path
            language_map = {"eng": "en", "rus": "ru", "fra": "fr", "deu": "de"}
            tts_language = language_map.get(tgt_lang, "en")
            MAX_GAP_MS = 400
            CONTINUATION_PAUSE_MS = 120  # максимальная пауза, если фраза продолжается
            MAX_PAUSE_MS = 2000  # максимальная пауза в любом случае (2 секунды)
            
            for idx, seg in enumerate(segments):
                # Корректируем временные метки с учетом смещения
                start_ms = int((float(seg.get("start", 0)) + time_offset) * 1000)
                end_ms = int((float(seg.get("end", 0)) + time_offset) * 1000)
                next_start_ms = int((float(segments[idx + 1].get("start", 0)) + time_offset) * 1000) if (idx + 1) < len(segments) else None
                text_ru = (seg.get("text") or "").strip()
                if end_ms <= start_ms or not text_ru:
                    continue
                    
                # реальная пауза до сегмента
                if start_ms > current_ms:
                    gap = start_ms - current_ms
                    # Если предыдущая фраза НЕ была завершена пунктуацией, считаем продолжением и ограничиваем паузу
                    if not locals().get("prev_en_terminal", True):
                        gap = min(gap, CONTINUATION_PAUSE_MS)
                    # Ограничиваем максимальную паузу в любом случае
                    gap = min(gap, MAX_PAUSE_MS)
                    combined += AudioSegment.silent(duration=gap)
                    current_ms += gap  # Обновляем current_ms на основе реально добавленной паузы
                    
                # перевод сегмента
                text_en = self._translate_text_ru_en(text_ru)
                if not text_en:
                    continue
                    
                part_raw = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_raw_{idx}.wav")
                result_path = self.voice_converter.synthesize_with_voice(
                    text=text_en,
                    speaker_audio_path=speaker_source,
                    output_path=part_raw,
                    language=tts_language
                )
                if not result_path:
                    continue
                
                # Умное выравнивание: избегаем замедления, предпочитаем сдвиг
                try:
                    seg_audio = AudioSegment.from_wav(part_raw)
                    en_duration = len(seg_audio)
                    ru_duration = end_ms - start_ms
                    
                    if en_duration <= ru_duration:
                        # EN короче RU: размещаем относительно current_ms для синхронизации
                        combined += seg_audio
                        current_ms += en_duration
                        print(f"Сегмент {idx}: EN короче RU, размещен относительно current_ms")
                    else:
                        # EN длиннее RU: используем стратегию сдвига временных меток (БЕЗ перекрытия)
                        available_gap = 0
                        if next_start_ms is not None:
                            available_gap = max(0, next_start_ms - end_ms)
                        
                        # Максимально увеличиваем доступное окно для английского текста
                        allowed_ms = ru_duration + min(available_gap, MAX_GAP_MS)
                        
                        if en_duration <= allowed_ms:
                            # Помещаем целиком, используем доступную паузу
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, заняли паузу {max(0, en_duration - ru_duration)}ms")
                        else:
                            # Английский слишком длинный - добавляем весь текст и сдвигаем следующие сегменты
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, сдвигаем следующие сегменты на {en_duration - ru_duration}ms (БЕЗ ПЕРЕКРЫТИЯ)")
                        
                except Exception as e:
                    print(f"Ошибка обработки сегмента {idx}: {e}")
                    continue
                    
                # Обновляем флаг завершённости фразы для управления следующей паузой
                try:
                    prev_en_terminal = bool(re.search(r"[\.\!\?…:]\s*$", text_en))
                except Exception:
                    prev_en_terminal = True
                    
            if len(combined) == 0:
                print("Не удалось синтезировать речь для сегмента")
                return False
            else:
                combined.export(temp_synthesized_path, format="wav")
            
            # Шаг 3: Применяем просодическую информацию для точного сохранения ритма
            print("Применяем просодию (паузы, интонацию)...")
            final_audio_path = temp_synthesized_path
            if prosody_info:
                final_audio_path = os.path.join(os.path.dirname(output_path), "temp_final_audio.wav")
                result = self.video_processor._apply_prosody_to_audio(
                    temp_synthesized_path,
                    prosody_info,
                    final_audio_path,
                    video_path=None  # Для сегментов не используем точную синхронизацию
                )
                if not result:
                    print("Предупреждение: Не удалось применить просодию, используем исходное аудио")
                    final_audio_path = temp_synthesized_path
            
            # Шаг: Сохраняем чистый TTS без подложки (подложка будет добавлена в конце)
            print("Сохраняем чистый TTS...")
            try:
                import shutil
                shutil.copy2(final_audio_path, output_path)
                print("Чистый TTS сохранен")
            except Exception as e:
                print(f"Ошибка сохранения TTS: {e}")
                return False
            
            # Очищаем временные файлы
            try:
                os.remove(temp_synthesized_path)
                if final_audio_path != temp_synthesized_path:
                    os.remove(final_audio_path)
                # Удаляем временные файлы сегментов
                for i in range(len(segments)):
                    temp_file = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_raw_{i}.wav")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    temp_file = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_fit_{i}.wav")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except:
                pass
            
            print("Сегмент обработан успешно")
            return True
            
        except Exception as e:
            print(f"Ошибка обработки сегмента: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _process_segment_as_short_video(self, segment_audio_path, output_path, src_lang, tgt_lang, prosody_info, speaker_reference_path=None):
        """
        Обрабатывает сегмент как короткое видео - использует тот же алгоритм что и для коротких видео
        
        Args:
            segment_audio_path (str): Путь к аудио сегменту
            output_path (str): Путь для сохранения результата
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
            prosody_info (dict): Информация о просодии
            speaker_reference_path (str): Путь к эталонному голосу для клонирования
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            print("Обрабатываем сегмент как короткое видео...")
            
            # Шаг 1: Сегментация оригинального RU-аудио с таймкодами (Whisper)
            print("Сегментируем оригинальное аудио с таймкодами (Whisper)...")
            segments = self._asr_whisper_segments(segment_audio_path, language="ru")
            if not segments:
                # фолбэк — цельный текст
                ru_text = self._extract_text_from_audio(segment_audio_path, "ru")
                if not ru_text:
                    ru_text = self.audio_translator.speech_to_text(segment_audio_path, tgt_lang="rus") or ""
                segments = [{"start": 0.0, "end": prosody_info.get("duration", 0.0), "text": ru_text}] if ru_text else []
            print(f"Получено сегментов: {len(segments)}")

            # Шаг 2: Переводим и синтезируем КАЖДЫЙ сегмент в своё окно времени
            print("Синтезируем речь по сегментам с выравниванием...")
            temp_synthesized_path = os.path.join(os.path.dirname(output_path), "temp_synthesized_audio.wav")
            combined = AudioSegment.silent(duration=0)
            current_ms = 0
            speaker_source = speaker_reference_path if speaker_reference_path else segment_audio_path
            language_map = {"eng": "en", "rus": "ru", "fra": "fr", "deu": "de"}
            tts_language = language_map.get(tgt_lang, "en")
            MAX_GAP_MS = 400
            CONTINUATION_PAUSE_MS = 120  # максимальная пауза, если фраза продолжается
            MAX_PAUSE_MS = 2000  # максимальная пауза в любом случае (2 секунды)
            
            for idx, seg in enumerate(segments):
                start_ms = int(float(seg.get("start", 0)) * 1000)
                end_ms = int(float(seg.get("end", 0)) * 1000)
                next_start_ms = int(float(segments[idx + 1].get("start", 0)) * 1000) if (idx + 1) < len(segments) else None
                text_ru = (seg.get("text") or "").strip()
                if end_ms <= start_ms or not text_ru:
                    continue
                    
                # реальная пауза до сегмента
                if start_ms > current_ms:
                    gap = start_ms - current_ms
                    # Если предыдущая фраза НЕ была завершена пунктуацией, считаем продолжением и ограничиваем паузу
                    if not locals().get("prev_en_terminal", True):
                        gap = min(gap, CONTINUATION_PAUSE_MS)
                    # Ограничиваем максимальную паузу в любом случае
                    gap = min(gap, MAX_PAUSE_MS)
                    combined += AudioSegment.silent(duration=gap)
                    current_ms += gap  # Обновляем current_ms на основе реально добавленной паузы
                    
                # перевод сегмента
                text_en = self._translate_text_ru_en(text_ru)
                if not text_en:
                    continue
                    
                part_raw = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_raw_{idx}.wav")
                result_path = self.voice_converter.synthesize_with_voice(
                    text=text_en,
                    speaker_audio_path=speaker_source,
                    output_path=part_raw,
                    language=tts_language
                )
                if not result_path:
                    continue
                
                # Умное выравнивание: избегаем замедления, предпочитаем сдвиг
                try:
                    seg_audio = AudioSegment.from_wav(part_raw)
                    en_duration = len(seg_audio)
                    ru_duration = end_ms - start_ms
                    
                    if en_duration <= ru_duration:
                        # EN короче RU: размещаем относительно current_ms для синхронизации
                        combined += seg_audio
                        current_ms += en_duration
                        print(f"Сегмент {idx}: EN короче RU, размещен относительно current_ms")
                    else:
                        # EN длиннее RU: используем стратегию сдвига временных меток (БЕЗ перекрытия)
                        available_gap = 0
                        if next_start_ms is not None:
                            available_gap = max(0, next_start_ms - end_ms)
                        
                        # Максимально увеличиваем доступное окно для английского текста
                        allowed_ms = ru_duration + min(available_gap, MAX_GAP_MS)
                        
                        if en_duration <= allowed_ms:
                            # Помещаем целиком, используем доступную паузу
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, заняли паузу {max(0, en_duration - ru_duration)}ms")
                        else:
                            # Английский слишком длинный - добавляем весь текст и сдвигаем следующие сегменты
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, сдвигаем следующие сегменты на {en_duration - ru_duration}ms (БЕЗ ПЕРЕКРЫТИЯ)")
                        
                except Exception as e:
                    print(f"Ошибка обработки сегмента {idx}: {e}")
                    continue
                    
                # Обновляем флаг завершённости фразы для управления следующей паузой
                try:
                    prev_en_terminal = bool(re.search(r"[\.\!\?…:]\s*$", text_en))
                except Exception:
                    prev_en_terminal = True
                    
            if len(combined) == 0:
                print("Не удалось синтезировать речь для сегмента")
                return False
            else:
                combined.export(temp_synthesized_path, format="wav")
            
            # Шаг 3: Применяем просодическую информацию для точного сохранения ритма
            print("Применяем просодию (паузы, интонацию)...")
            final_audio_path = temp_synthesized_path
            if prosody_info:
                final_audio_path = os.path.join(os.path.dirname(output_path), "temp_final_audio.wav")
                result = self.video_processor._apply_prosody_to_audio(
                    temp_synthesized_path,
                    prosody_info,
                    final_audio_path,
                    video_path=None  # Для сегментов не используем точную синхронизацию
                )
                if not result:
                    print("Предупреждение: Не удалось применить просодию, используем исходное аудио")
                    final_audio_path = temp_synthesized_path
            
            # Шаг: Сохраняем чистый TTS без подложки (подложка будет добавлена в конце)
            print("Сохраняем чистый TTS...")
            try:
                import shutil
                shutil.copy2(final_audio_path, output_path)
                print("Чистый TTS сохранен")
            except Exception as e:
                print(f"Ошибка сохранения TTS: {e}")
                return False
            
            # Очищаем временные файлы
            try:
                os.remove(temp_synthesized_path)
                if final_audio_path != temp_synthesized_path:
                    os.remove(final_audio_path)
                # Удаляем временные файлы сегментов
                for i in range(len(segments)):
                    temp_file = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_raw_{i}.wav")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                    temp_file = os.path.join(os.path.dirname(output_path), f"temp_tts_seg_fit_{i}.wav")
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
            except:
                pass
            
            print("Сегмент обработан успешно")
            return True
            
        except Exception as e:
            print(f"Ошибка обработки сегмента: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _combine_segments_with_full_functionality(self, segment_files, output_path, original_audio_path, prosody_info):
        """
        Объединяет сегменты с полным функционалом (как в коротких видео)
        
        Args:
            segment_files (list): Список путей к файлам сегментов
            output_path (str): Путь для сохранения результата
            original_audio_path (str): Путь к оригинальному аудио
            prosody_info (dict): Информация о просодии
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            from pydub import AudioSegment
            
            print("Объединяем сегменты с полным функционалом...")
            
            # Объединяем все сегменты
            combined = AudioSegment.silent(duration=0)
            
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    segment = AudioSegment.from_wav(segment_file)
                    combined += segment
            
            if len(combined) == 0:
                print("❌ Нет аудио для объединения")
                return False
            
            # Применяем просодию если доступна
            if prosody_info:
                print("Применяем просодию к объединенному аудио...")
                temp_prosody_path = output_path.replace(".wav", "_temp_prosody.wav")
                combined.export(temp_prosody_path, format="wav")
                
                # Применяем просодию
                result = self.video_processor._apply_prosody_to_audio(
                    temp_prosody_path,
                    prosody_info,
                    output_path,
                    video_path=None  # Для длинных видео не используем точную синхронизацию
                )
                
                if result and os.path.exists(output_path):
                    # Удаляем временный файл
                    try:
                        os.remove(temp_prosody_path)
                    except:
                        pass
                else:
                    # Фолбэк: используем исходное объединенное аудио
                    combined.export(output_path, format="wav")
            else:
                combined.export(output_path, format="wav")
            
            # Смешиваем с оригинальным аудио если доступно
            if original_audio_path and os.path.exists(original_audio_path):
                print("Смешиваем с тихим оригинальным аудио...")
                try:
                    final_audio = AudioSegment.from_wav(output_path)
                    original_audio = AudioSegment.from_wav(original_audio_path)
                    
                    # Подгоняем длину оригинального аудио под финальное
                    if len(original_audio) < len(final_audio):
                        original_audio = original_audio + AudioSegment.silent(duration=len(final_audio) - len(original_audio))
                    elif len(original_audio) > len(final_audio):
                        original_audio = original_audio[:len(final_audio)]
                    
                    # Уменьшаем громкость оригинала
                    bed_gain_db = -24
                    bed = original_audio + bed_gain_db
                    mixed = bed.overlay(final_audio)
                    
                    # Сохраняем смешанное аудио
                    mixed.export(output_path, format="wav")
                    print("Создана тихая подложка из оригинального аудио")
                    
                except Exception as e:
                    print(f"Не удалось смешать с оригиналом: {e}")
            
            print("✅ Сегменты объединены с полным функционалом")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка объединения сегментов: {e}")
            return False

    def _combine_segments_simple(self, segment_files, output_path):
        """
        Простое объединение сегментов без сложной обработки
        
        Args:
            segment_files (list): Список путей к файлам сегментов
            output_path (str): Путь для сохранения результата
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            from pydub import AudioSegment
            
            print("Объединяем сегменты...")
            combined = AudioSegment.silent(duration=0)
            
            for i, segment_file in enumerate(segment_files):
                if os.path.exists(segment_file):
                    print(f"Добавляем сегмент {i+1}/{len(segment_files)}")
                    segment = AudioSegment.from_wav(segment_file)
                    combined += segment
                else:
                    print(f"⚠️ Сегмент {i+1} не найден: {segment_file}")
            
            if len(combined) == 0:
                print("❌ Нет аудио для объединения")
                return False
            
            # Сохраняем объединенное аудио
            combined.export(output_path, format="wav")
            print(f"✅ Объединено {len(segment_files)} сегментов")
            return True
            
        except Exception as e:
            print(f"❌ Ошибка объединения сегментов: {e}")
            return False

    def _translate_segment(self, segment_audio_path, output_path, src_lang, tgt_lang, original_video_path=None, speaker_reference_path=None):
        """Переводит отдельный сегмент аудио с полным функционалом"""
        try:
            # Шаг 1: Извлекаем текст из русского аудио (Whisper)
            print("Обрабатываем сегмент 1...")
            russian_text = self._extract_text_from_audio(segment_audio_path, "ru")
            
            if not russian_text:
                print("Не удалось извлечь русский текст, используем упрощенный перевод")
                # Фолбэк: используем SeamlessM4T напрямую
                translated_audio, sample_rate = self.audio_translator.translate_audio(
                    segment_audio_path, src_lang=src_lang, tgt_lang=tgt_lang
                )
                if translated_audio is None:
                    return False
                import scipy.io.wavfile as wavfile
                wavfile.write(output_path, rate=sample_rate, data=translated_audio)
                return True
            
            # Шаг 2: Переводим текст с русского на английский
            print("Обрабатываем сегмент 2...")
            english_text = self._translate_text(russian_text, src_lang="ru", tgt_lang="en")
            
            if not english_text:
                print("Не удалось перевести текст")
                return False
            
            # Шаг 3: Синтезируем английскую речь с клонированием голоса
            print("Обрабатываем сегмент 3...")
            synthesized_audio = None
            
            if speaker_reference_path and os.path.exists(speaker_reference_path):
                try:
                    # Используем YourTTS с клонированием голоса
                    synthesized_audio = self.voice_converter.synthesize_speech_with_voice_cloning(
                        english_text, 
                        speaker_reference_path, 
                        language="en"
                    )
                    print("Использован голос диктора для синтеза")
                except Exception as e:
                    print(f"Ошибка клонирования голоса: {e}, используем обычный TTS")
                    synthesized_audio = None
            else:
                print("Голос диктора недоступен, используем обычный TTS")
                synthesized_audio = None
            
            if synthesized_audio is None:
                # Фолбэк: обычный TTS без клонирования
                try:
                    synthesized_audio = self.voice_converter.synthesize_speech(english_text, language="en")
                    print("Использован обычный TTS")
                except Exception as e:
                    print(f"Ошибка обычного TTS: {e}")
                    return False
            
            # Шаг 4: Сохраняем результат
            import scipy.io.wavfile as wavfile
            wavfile.write(output_path, rate=22050, data=synthesized_audio)
            
            return True
            
        except Exception as e:
            print(f"Ошибка перевода сегмента: {e}")
            return False
    
    def _combine_segments(self, segment_files, output_path, original_audio_path=None, prosody_info=None):
        """Объединяет сегменты в один файл с применением просодии и смешиванием"""
        try:
            from pydub import AudioSegment
            
            # Объединяем все сегменты
            combined = AudioSegment.silent(duration=0)
            
            for segment_file in segment_files:
                if os.path.exists(segment_file):
                    segment = AudioSegment.from_wav(segment_file)
                    combined += segment
            
            # Применяем просодию если доступна
            if prosody_info:
                print("Применяем просодию к объединенному аудио...")
                temp_prosody_path = output_path.replace(".wav", "_temp_prosody.wav")
                combined.export(temp_prosody_path, format="wav")
                
                # Применяем просодию
                result = self.video_processor._apply_prosody_to_audio(
                    temp_prosody_path,
                    prosody_info,
                    output_path,
                    video_path=None  # Для длинных видео не используем точную синхронизацию
                )
                
                if result and os.path.exists(output_path):
                    # Удаляем временный файл
                    try:
                        os.remove(temp_prosody_path)
                    except:
                        pass
                else:
                    # Фолбэк: используем исходное объединенное аудио
                    combined.export(output_path, format="wav")
            else:
                combined.export(output_path, format="wav")
            
            # Смешиваем с оригинальным аудио если доступно
            if original_audio_path and os.path.exists(original_audio_path):
                print("Смешиваем с тихим оригинальным аудио...")
                try:
                    final_audio = AudioSegment.from_wav(output_path)
                    original_audio = AudioSegment.from_wav(original_audio_path)
                    
                    # Подгоняем длину оригинального аудио под финальное
                    if len(original_audio) < len(final_audio):
                        original_audio = original_audio + AudioSegment.silent(duration=len(final_audio) - len(original_audio))
                    elif len(original_audio) > len(final_audio):
                        original_audio = original_audio[:len(final_audio)]
                    
                    # Уменьшаем громкость оригинала
                    bed_gain_db = -24
                    bed = original_audio + bed_gain_db
                    mixed = bed.overlay(final_audio)
                    
                    # Сохраняем смешанное аудио
                    mixed.export(output_path, format="wav")
                    print("Создана тихая подложка из оригинального аудио")
                    
                except Exception as e:
                    print(f"Не удалось смешать с оригиналом: {e}")
            
            return True
            
        except Exception as e:
            print(f"Ошибка объединения сегментов: {e}")
            return False
    
    def _check_memory_usage(self):
        """Проверяет использование памяти"""
        try:
            import psutil
            process = psutil.Process()
            memory_gb = process.memory_info().rss / (1024**3)
            return memory_gb
        except ImportError:
            return 0  # Если psutil не установлен, возвращаем 0
        except Exception:
            return 0
    
    def _cleanup_segment_files(self, segment_files):
        """Очищает временные файлы сегментов"""
        for file_path in segment_files:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    def translate_video_with_voice_preservation(self, input_video_path, output_video_path, 
                                               src_lang="rus", tgt_lang="eng"):
        """
        Переводит видео с сохранением голоса, пауз и интонации (упрощенная версия)
        
        Args:
            input_video_path (str): Путь к исходному видео на русском
            output_video_path (str): Путь для сохранения переведенного видео
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        try:
            print("=" * 60)
            print("НАЧИНАЕМ ПЕРЕВОД ВИДЕО С СОХРАНЕНИЕМ ГОЛОСА (УПРОЩЕННАЯ ВЕРСИЯ)")
            print("=" * 60)
            
            # Готовим директорию для WAV-файлов
            base_name = os.path.splitext(os.path.basename(input_video_path))[0]
            wav_dir = os.path.join("wav", base_name)
            os.makedirs(wav_dir, exist_ok=True)
            
            # Шаг 1: Извлекаем аудио из видео
            print("\nШАГ 1: Извлекаем аудио из видео...")
            extracted_audio_path = os.path.join(wav_dir, f"{base_name}_audio.wav")
            audio_path = self.video_processor.extract_audio_from_video(input_video_path, audio_path=extracted_audio_path)
            
            if not audio_path:
                print(" Ошибка: Не удалось извлечь аудио из видео")
                return False
            
            # Шаг 2: Анализируем просодию оригинального аудио
            print("\nШАГ 2: Анализируем просодию (паузы, интонацию, ритм)...")
            prosody_info = self.video_processor.analyze_audio_prosody(audio_path)
            
            if not prosody_info:
                print(" Предупреждение: Не удалось проанализировать просодию")
                prosody_info = {}
            
            # Шаг 3: Переводим аудио
            print("\nШАГ 3: Переводим аудио...")
            translated_audio, sample_rate = self.audio_translator.translate_audio(
                audio_path, 
                src_lang=src_lang, 
                tgt_lang=tgt_lang
            )
            
            if translated_audio is None:
                print(" Ошибка: Не удалось перевести аудио")
                return False
            
            # Сохраняем переведенное аудио
            temp_translated_path = os.path.join(wav_dir, "temp_translated_audio.wav")
            import scipy.io.wavfile as wavfile
            wavfile.write(temp_translated_path, rate=sample_rate, data=translated_audio)
            
            # Шаг 4: Сегментация оригинального RU-аудио с таймкодами (Whisper)
            print("\nШАГ 4: Сегментируем оригинальное аудио с таймкодами (Whisper)...")
            segments = self._asr_whisper_segments(audio_path, language="ru")
            if not segments:
                # фолбэк — цельный текст
                ru_text = self._extract_text_from_audio(audio_path, "ru")
                if not ru_text:
                    ru_text = self.audio_translator.speech_to_text(audio_path, tgt_lang="rus") or ""
                segments = [{"start": 0.0, "end": prosody_info.get("duration", 0.0), "text": ru_text}] if ru_text else []
            print(f"Получено сегментов: {len(segments)}")

            # Шаг 5: Переводим и синтезируем КАЖДЫЙ сегмент в своё окно времени
            print("\nШАГ 5: Синтезируем речь по сегментам с выравниванием...")
            temp_synthesized_path = os.path.join(wav_dir, "temp_synthesized_audio.wav")
            combined = AudioSegment.silent(duration=0)
            current_ms = 0
            speaker_ref = self.video_processor.extract_speaker_reference(audio_path, prosody_info)
            speaker_source = speaker_ref if speaker_ref else audio_path
            language_map = {"eng": "en", "rus": "ru", "fra": "fr", "deu": "de"}
            tts_language = language_map.get(tgt_lang, "en")
            MAX_GAP_MS = 400
            CONTINUATION_PAUSE_MS = 120  # максимальная пауза, если фраза продолжается
            MAX_PAUSE_MS = 2000  # максимальная пауза в любом случае (2 секунды)
            for idx, seg in enumerate(segments):
                start_ms = int(float(seg.get("start", 0)) * 1000)
                end_ms = int(float(seg.get("end", 0)) * 1000)
                next_start_ms = int(float(segments[idx + 1].get("start", 0)) * 1000) if (idx + 1) < len(segments) else None
                text_ru = (seg.get("text") or "").strip()
                if end_ms <= start_ms or not text_ru:
                    continue
                # реальная пауза до сегмента
                if start_ms > current_ms:
                    gap = start_ms - current_ms
                    # Если предыдущая фраза НЕ была завершена пунктуацией, считаем продолжением и ограничиваем паузу
                    if not locals().get("prev_en_terminal", True):
                        gap = min(gap, CONTINUATION_PAUSE_MS)
                    # Ограничиваем максимальную паузу в любом случае
                    gap = min(gap, MAX_PAUSE_MS)
                    combined += AudioSegment.silent(duration=gap)
                    current_ms += gap  # Обновляем current_ms на основе реально добавленной паузы
                # перевод сегмента
                text_en = self._translate_text_ru_en(text_ru)
                if not text_en:
                    continue
                part_raw = os.path.join(wav_dir, f"temp_tts_seg_raw_{idx}.wav")
                result_path = self.voice_converter.synthesize_with_voice(
                    text=text_en,
                    speaker_audio_path=speaker_source,
                    output_path=part_raw,
                    language=tts_language
                )
                if not result_path:
                    continue
                
                # Умное выравнивание: избегаем замедления, предпочитаем сдвиг
                try:
                    seg_audio = AudioSegment.from_wav(part_raw)
                    en_duration = len(seg_audio)
                    ru_duration = end_ms - start_ms
                    
                    if en_duration <= ru_duration:
                        # EN короче RU: размещаем относительно current_ms для синхронизации
                        combined += seg_audio
                        current_ms += en_duration
                        print(f"Сегмент {idx}: EN короче RU, размещен относительно current_ms")
                    else:
                        # EN длиннее RU: используем стратегию сдвига временных меток (БЕЗ перекрытия)
                        available_gap = 0
                        if next_start_ms is not None:
                            available_gap = max(0, next_start_ms - end_ms)
                        allowed_ms = ru_duration + available_gap
                        if en_duration <= allowed_ms:
                            # Помещаем целиком, съедаем часть/всю паузу до следующего сегмента
                            combined += seg_audio
                            current_ms += en_duration
                            print(f"Сегмент {idx}: EN длиннее, заняли паузу {max(0, en_duration - ru_duration)}ms")
                        else:
                            # Нужно сжатие до allowed_ms (сохранение высоты через atempo)
                            part_fit = os.path.join(wav_dir, f"temp_tts_seg_fit_{idx}.wav")
                            ok = self._stretch_wav_to_duration(part_raw, allowed_ms, part_fit)
                            if ok:
                                try:
                                    seg_audio_fit = AudioSegment.from_wav(part_fit)
                                    combined += seg_audio_fit
                                    current_ms += allowed_ms
                                    print(f"Сегмент {idx}: EN > окно, сжатие на {en_duration - allowed_ms}ms")
                                except Exception:
                                    # НЕ ОБРЕЗАЕМ! Добавляем весь английский текст
                                    combined += seg_audio
                                    current_ms += en_duration
                            else:
                                # Фолбэк: без сжатия добавим и ограничим дрейф (редкий случай)
                                combined += seg_audio
                                current_ms = start_ms + en_duration
                        
                except Exception as e:
                    print(f"Ошибка обработки сегмента {idx}: {e}")
                    continue
                # Обновляем флаг завершённости фразы для управления следующей паузой
                try:
                    prev_en_terminal = bool(re.search(r"[\.\!\?…:]\s*$", text_en))
                except Exception:
                    prev_en_terminal = True
            if len(combined) == 0:
                temp_synthesized_path = temp_translated_path
            else:
                combined.export(temp_synthesized_path, format="wav")
            
            # Шаг 6: Применяем просодическую информацию для точного сохранения ритма
            print("\nШАГ 6: Применяем просодию (паузы, интонацию)...")
            final_audio_path = temp_synthesized_path
            if prosody_info:
                final_audio_path = os.path.join(wav_dir, "temp_final_audio.wav")
                result = self.video_processor._apply_prosody_to_audio(
                    temp_synthesized_path,
                    prosody_info,
                    final_audio_path,
                    video_path=input_video_path
                )
                if not result:
                    print("Предупреждение: Не удалось применить просодию, используем исходное аудио")
                    final_audio_path = temp_synthesized_path
            
            # Шаг 7: Добавляем тихую подложку оригинального аудио под синтез
            print("\nШАГ 7: Микшируем тихую оригинальную дорожку под TTS...")
            try:
                tts_seg = AudioSegment.from_wav(final_audio_path)
                orig_seg = AudioSegment.from_wav(audio_path)
                # приводим длину подлежащей дорожки к длине TTS
                if len(orig_seg) < len(tts_seg):
                    orig_seg = orig_seg + AudioSegment.silent(duration=len(tts_seg) - len(orig_seg))
                elif len(orig_seg) > len(tts_seg):
                    orig_seg = orig_seg[:len(tts_seg)]
                # уменьшаем громкость оригинала, чтобы он не мешал речи
                bed_gain_db =  -24
                bed = orig_seg + bed_gain_db
                mixed = bed.overlay(tts_seg)
                mixed_audio_path = os.path.join(wav_dir, "temp_mixed_audio.wav")
                mixed.export(mixed_audio_path, format="wav")
                used_audio_path = mixed_audio_path
                print("Сформирована подложка оригинала под TTS")
            except Exception as e:
                print(f"Не удалось смешать оригинал с TTS: {e}. Используем чистый TTS.")
                used_audio_path = final_audio_path

            # Шаг 8: Синхронизируем новое аудио с видео
            print("\nШАГ 8: Синхронизируем аудио с видео...")
            final_video_path = self.video_processor.sync_audio_with_video(
                input_video_path,
                used_audio_path,
                output_video_path
            )
            
            if final_video_path:
                print(f"\n УСПЕХ! Переведенное видео с сохранением голоса: {output_video_path}")
                
                # Очищаем временные файлы
                self._cleanup_temp_files([audio_path, temp_translated_path, temp_synthesized_path])
                
                return True
            else:
                print(" Ошибка: Не удалось создать финальное видео")
                return False
                
        except Exception as e:
            print(f" Ошибка в pipeline: {e}")
            return False
    
    def _cleanup_temp_files(self, file_paths):
        """Очищает временные файлы"""
        for file_path in file_paths:
            if os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except:
                    pass
    
    def _extract_text_from_audio(self, audio_path, language):
        """
        Извлекает текст из аудио с помощью ASR (Automatic Speech Recognition)
        
        Args:
            audio_path (str): Путь к аудио файлу
            language (str): Язык аудио
        
        Returns:
            str: Извлеченный текст
        """
        try:
            print(f"Извлекаем текст из аудио: {audio_path}")
            
            # Используем Whisper для ASR (если доступен)
            try:
                import whisper
                model = whisper.load_model("base")
                result = model.transcribe(audio_path, language=language)
                return result["text"].strip()
            except ImportError:
                print("Whisper не установлен — текст не получен")
                return ""
            except Exception as e:
                print(f"Ошибка Whisper: {e}")
                return ""
                    
        except Exception as e:
            print(f"Ошибка при извлечении текста: {e}")
            return ""

    def _translate_text(self, text, src_lang="ru", tgt_lang="en"):
        """
        Переводит текст с одного языка на другой
        
        Args:
            text (str): Текст для перевода
            src_lang (str): Исходный язык
            tgt_lang (str): Целевой язык
        
        Returns:
            str: Переведенный текст
        """
        try:
            if not text or not text.strip():
                return ""
            
            # Используем MarianMT для перевода текста
            from transformers import MarianMTModel, MarianTokenizer
            
            # Маппинг языков для MarianMT
            lang_map = {
                "ru": "ru",
                "en": "en", 
                "rus": "ru",
                "eng": "en"
            }
            
            src_code = lang_map.get(src_lang, src_lang)
            tgt_code = lang_map.get(tgt_lang, tgt_lang)
            
            model_name = f"Helsinki-NLP/opus-mt-{src_code}-{tgt_code}"
            
            try:
                tokenizer = MarianTokenizer.from_pretrained(model_name)
                model = MarianMTModel.from_pretrained(model_name)
                
                # Переводим текст
                inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
                translated = model.generate(**inputs, max_length=512, num_beams=4, early_stopping=True)
                translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
                
                return translated_text.strip()
                
            except Exception as e:
                print(f"Ошибка перевода текста: {e}")
                return text  # Возвращаем оригинальный текст в случае ошибки
                
        except Exception as e:
            print(f"Ошибка в _translate_text: {e}")
            return text

    def _split_into_sentences(self, text: str):
        try:
            if not text:
                return []
            # Простая разметка по пунктуации. Можно заменить на nltk/Punkt при наличии.
            parts = re.split(r"(?<=[\.!?])\s+", text.strip())
            # Чистим пустые хвосты и артефакты
            sentences = [p.strip() for p in parts if p and p.strip()]
            # Защита от зацикливаний: убираем длинные повторы одинаковых слов в конце
            cleaned = []
            for s in sentences:
                # Если в конце 3+ одинаковых слов подряд — подрежем
                tokens = s.split()
                if len(tokens) > 6:
                    tail = tokens[-6:]
                    if len(set(tail)) <= 2:
                        s = " ".join(tokens[:-3])
                cleaned.append(s)
            return cleaned
        except Exception:
            return [text] if text else []

    def _asr_whisper_segments(self, audio_path: str, language: str = "ru"):
        """Возвращает список сегментов Whisper: [{start, end, text}]"""
        try:
            import whisper
            model = whisper.load_model("base")
            result = model.transcribe(audio_path, language=language)
            segs = []
            for s in result.get("segments", []) or []:
                segs.append({
                    "start": float(s.get("start", 0.0)),
                    "end": float(s.get("end", 0.0)),
                    "text": (s.get("text") or "").strip()
                })
            return segs
        except Exception as e:
            print(f"Ошибка Whisper segments: {e}")
            return []

    def _stretch_wav_to_duration(self, input_wav: str, target_ms: int, output_wav: str) -> bool:
        """Подгоняет длительность WAV под target_ms ТОЛЬКО через добавление пауз между словами (без изменения скорости речи)."""
        try:
            seg = AudioSegment.from_wav(input_wav)
            cur_ms = len(seg)
            if cur_ms == 0:
                return False
            if abs(cur_ms - target_ms) <= 40:
                seg.export(output_wav, format="wav")
                return True
            
            # Если нужно растянуть аудио (английский короче русского)
            if cur_ms < target_ms:
                return self._add_pauses_between_words(input_wav, target_ms, output_wav)
            
            # Если английский длиннее русского - НЕ СЖИМАЕМ! Просто экспортируем как есть
            # Это гарантирует, что весь смысл будет сохранен
            print(f"⚠️ Английский текст длиннее ({cur_ms}ms > {target_ms}ms), НЕ СЖИМАЕМ для сохранения смысла")
            seg.export(output_wav, format="wav")
            return True
            
        except Exception as e:
            print(f"Ошибка stretch (без изменения скорости): {e}")
            try:
                seg = AudioSegment.from_wav(input_wav)
                # Всегда экспортируем оригинал без изменений
                seg.export(output_wav, format="wav")
                return True
            except Exception:
                return False

    def _add_pauses_between_words(self, input_wav: str, target_duration: int, output_wav: str) -> bool:
        """Добавляет паузы между словами для растягивания аудио БЕЗ изменения скорости речи."""
        try:
            from pydub import AudioSegment
            import numpy as np
            
            # Загружаем аудио
            audio = AudioSegment.from_wav(input_wav)
            current_duration = len(audio)
            
            # Вычисляем, сколько времени нужно добавить
            additional_time = target_duration - current_duration
            
            # Если нужно добавить менее 50ms, просто экспортируем
            if additional_time < 50:
                audio.export(output_wav, format="wav")
                return True
            
            # Конвертируем в numpy для анализа
            samples = np.array(audio.get_array_of_samples())
            if audio.channels == 2:
                samples = samples.reshape((-1, 2))
            
            # Находим тихие участки (паузы между словами)
            # Используем энергию для определения пауз
            if audio.channels == 2:
                energy = np.sqrt(np.mean(samples**2, axis=1))
            else:
                energy = np.abs(samples)
            
            # Сглаживаем энергию для лучшего определения пауз
            window_size = int(0.05 * audio.frame_rate)  # 50ms окно
            if window_size > 0:
                energy_smooth = np.convolve(energy, np.ones(window_size)/window_size, mode='same')
            else:
                energy_smooth = energy
            
            # Находим тихие участки (ниже 15% от максимальной энергии)
            threshold = np.max(energy_smooth) * 0.15
            quiet_regions = energy_smooth < threshold
            
            # Находим границы тихих участков
            quiet_starts = []
            quiet_ends = []
            in_quiet = False
            
            for i, is_quiet in enumerate(quiet_regions):
                if is_quiet and not in_quiet:
                    quiet_starts.append(i)
                    in_quiet = True
                elif not is_quiet and in_quiet:
                    quiet_ends.append(i)
                    in_quiet = False
            
            if in_quiet:
                quiet_ends.append(len(quiet_regions))
            
            # Добавляем паузы в тихие участки
            if quiet_starts and quiet_ends:
                # Вычисляем среднюю длину паузы для добавления
                avg_pause_length = additional_time // len(quiet_starts)
                avg_pause_length = max(30, min(avg_pause_length, 150))  # От 30ms до 150ms
                
                # Создаем новое аудио с добавленными паузами
                new_audio = AudioSegment.silent(duration=0)
                last_end = 0
                
                for start, end in zip(quiet_starts, quiet_ends):
                    # Добавляем аудио до паузы
                    if start > last_end:
                        segment = audio[last_end:start]
                        new_audio += segment
                    
                    # Добавляем расширенную паузу
                    pause_duration = end - start + avg_pause_length
                    pause_duration = min(pause_duration, 200)  # Максимум 200ms
                    pause = AudioSegment.silent(duration=pause_duration)
                    new_audio += pause
                    
                    last_end = end
                
                # Добавляем оставшееся аудио
                if last_end < len(audio):
                    segment = audio[last_end:]
                    new_audio += segment
                
                # Если все еще не достигли целевой длительности, добавляем паузу в конце
                if len(new_audio) < target_duration:
                    final_pause = target_duration - len(new_audio)
                    new_audio += AudioSegment.silent(duration=final_pause)
                
                new_audio.export(output_wav, format="wav")
                print(f"✅ Добавлены паузы между словами: +{additional_time}ms (скорость речи не изменена)")
                return True
            else:
                # Если не нашли паузы, просто добавляем паузу в конце
                final_pause = target_duration - current_duration
                audio += AudioSegment.silent(duration=final_pause)
                audio.export(output_wav, format="wav")
                print(f"✅ Добавлена пауза в конце: +{final_pause}ms (скорость речи не изменена)")
                return True
                
        except Exception as e:
            print(f"Ошибка добавления пауз: {e}")
            return False
                
    def _translate_text_ru_en(self, text_ru: str) -> str:
        """Локальный перевод RU->EN через MarianMT (офлайн). Возвращает английский текст или пустую строку."""
        try:
            if not text_ru or not text_ru.strip():
                return ""
            
            # Кэшируем модель и токенизатор в классе
            if not hasattr(self, '_ru_en_tokenizer') or not hasattr(self, '_ru_en_model'):
                print("Загружаем модель перевода RU->EN...")
                from transformers import MarianMTModel, MarianTokenizer
                import time
                
                model_name = "Helsinki-NLP/opus-mt-ru-en"
                
                # Повторные попытки с экспоненциальной задержкой
                max_retries = 5
                for attempt in range(max_retries):
                    try:
                        print(f"Попытка {attempt + 1}/{max_retries} загрузки модели...")
                        self._ru_en_tokenizer = MarianTokenizer.from_pretrained(model_name)
                        self._ru_en_model = MarianMTModel.from_pretrained(model_name)
                        print("✅ Модель перевода загружена успешно")
                        break
                    except Exception as e:
                        if "429" in str(e) or "rate limit" in str(e).lower():
                            wait_time = 2 ** attempt  # 1, 2, 4, 8, 16 секунд
                            print(f"⚠️ Rate limit, ждем {wait_time} секунд...")
                            time.sleep(wait_time)
                        else:
                            print(f"❌ Ошибка загрузки модели: {e}")
                            return ""
                else:
                    print("❌ Не удалось загрузить модель после всех попыток")
                    return ""
            
            # Переводим текст
            batch = self._ru_en_tokenizer([text_ru], return_tensors="pt", truncation=True)
            import torch
            with torch.no_grad():
                gen = self._ru_en_model.generate(**batch, max_new_tokens=2048)
            en = self._ru_en_tokenizer.batch_decode(gen, skip_special_tokens=True)
            result = en[0].strip() if en else ""
            
            # Простая постобработка для улучшения качества
            result = self._postprocess_translation(result)
            return result
            
        except Exception as e:
            print(f"Ошибка локального перевoda RU->EN: {e}")
            return ""
    
    def _postprocess_translation(self, text: str) -> str:
        """Простая постобработка переведенного текста для улучшения качества."""
        if not text:
            return text
            
        # Убираем лишние пробелы
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Исправляем частые ошибки перевода (дублирование слов)
        corrections = {
            r'\bthe the\b': 'the',
            r'\ba a\b': 'a',
            r'\ban an\b': 'an',
            r'\bis is\b': 'is',
            r'\bthat that\b': 'that',
            r'\bof of\b': 'of',
            r'\band and\b': 'and',
            r'\bor or\b': 'or',
            r'\bto to\b': 'to',
            r'\bin in\b': 'in',
            r'\bfor for\b': 'for',
            r'\bwith with\b': 'with',
            r'\bby by\b': 'by',
            r'\bfrom from\b': 'from',
            r'\bat at\b': 'at',
            r'\bon on\b': 'on',
        }
        
        for pattern, replacement in corrections.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        # Убираем повторяющиеся слова в конце предложений
        text = re.sub(r'(\w+)\s+\1\s*([.!?])', r'\1\2', text)
        
        return text

# Пример использования
if __name__ == "__main__":
    pipeline = VideoTranslationPipeline()
    
    # Пример: Перевод видео с сохранением голоса
    # pipeline.translate_video_with_voice_preservation(
    #     "input_video.mp4", 
    #     "output_video_english.mp4"
    # )

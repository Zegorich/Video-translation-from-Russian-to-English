import os
import tempfile
import numpy as np
import librosa
import soundfile as sf
from pydub import AudioSegment
from pydub.silence import detect_nonsilent

class VideoProcessor:
    def __init__(self):
        """Инициализация упрощенного процессора видео"""
        print("Инициализируем упрощенный процессор видео...")
    
    def extract_audio_from_video(self, video_path, audio_path=None):
        """
        Извлекает аудио из видео файла (упрощенная версия)
        
        Args:
            video_path (str): Путь к видео файлу
            audio_path (str): Путь для сохранения аудио (опционально)
        
        Returns:
            str: Путь к извлеченному аудио файлу
        """
        try:
            print(f"Извлекаем аудио из видео: {video_path}")
            
            if audio_path is None:
                audio_path = video_path.rsplit(".", 1)[0] + "_audio.wav"
            
            # Используем pydub для извлечения аудио
            video = AudioSegment.from_file(video_path)
            
            # Экспортируем как WAV
            video.export(audio_path, format="wav")
            
            print(f"Аудио извлечено: {audio_path}")
            return audio_path
            
        except Exception as e:
            print(f"Ошибка при извлечении аудио: {e}")
            print("Попробуйте использовать ffmpeg напрямую:")
            print(f"ffmpeg -i \"{video_path}\" \"{audio_path}\"")
            return None
    
    def analyze_audio_prosody(self, audio_path):
        """
        Анализирует просодию аудио (паузы, интонацию, ритм)
        
        Args:
            audio_path (str): Путь к аудио файлу
        
        Returns:
            dict: Словарь с информацией о просодии
        """
        try:
            print("Анализируем просодию аудио...")
            
            # Загружаем аудио
            y, sr = librosa.load(audio_path, sr=None)
            
            # Анализируем паузы
            audio_segment = AudioSegment.from_wav(audio_path)
            nonsilent_ranges = detect_nonsilent(audio_segment, min_silence_len=100, silence_thresh=-40)
            
            # Анализируем интонацию (F0)
            f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"))
            
            # Анализируем ритм
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            
            # Анализируем энергию
            energy = librosa.feature.rms(y=y)[0]
            
            # Анализируем спектральные характеристики
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            
            prosody_info = {
                "sample_rate": sr,
                "duration": len(y) / sr,
                "nonsilent_ranges": nonsilent_ranges,
                "f0": f0,
                "voiced_flag": voiced_flag,
                "tempo": tempo,
                "beats": beats,
                "energy": energy,
                "spectral_centroids": spectral_centroids,
                "audio_length": len(y)
            }
            
            print("Анализ просодии завершен")
            return prosody_info
            
        except Exception as e:
            print(f"Ошибка при анализе просодии: {e}")
            return None
    
    def extract_speaker_reference(self, audio_path, prosody_info=None, min_duration_ms=4000, max_duration_ms=10000):
        """
        Извлекает чистый эталонный голосовой сегмент для копирования тембра.
        Предпочтение отдается первому достаточно длинному непрерывному участку речи.
        
        Args:
            audio_path (str): Путь к исходному аудио
            prosody_info (dict|None): Результат analyze_audio_prosody с 'nonsilent_ranges'
            min_duration_ms (int): Минимальная длительность эталонного сегмента
            max_duration_ms (int): Максимальная длительность эталонного сегмента
        
        Returns:
            str|None: Путь к сохраненному эталонному сегменту или None
        """
        try:
            audio = AudioSegment.from_wav(audio_path)
            nonsilent_ranges = None
            if prosody_info and 'nonsilent_ranges' in prosody_info:
                nonsilent_ranges = prosody_info['nonsilent_ranges']
            if not nonsilent_ranges:
                nonsilent_ranges = detect_nonsilent(audio, min_silence_len=100, silence_thresh=-40)

            if not nonsilent_ranges:
                return None

            # Выбираем самый длинный участок речи для максимально чистого эталона
            selected = max(nonsilent_ranges, key=lambda r: r[1] - r[0])

            start, end = selected
            duration = end - start
            # Приводим длительность к желаемому окну: предпочитаем более длинный эталон
            if duration > max_duration_ms:
                center = start + duration // 2
                half = max_duration_ms // 2
                start = max(0, center - half)
                end = start + max_duration_ms
            elif duration < min_duration_ms:
                # Если участок короче минимума, расширяем окно симметрично при возможности
                deficit = min_duration_ms - duration
                extend_left = deficit // 2
                extend_right = deficit - extend_left
                start = max(0, start - extend_left)
                end = end + extend_right

            ref_segment = audio[start:end]
            # сохраняем рядом с исходным WAV (который теперь лежит в wav/<basename>/)
            ref_path = os.path.splitext(audio_path)[0] + "_ref.wav"
            ref_segment.export(ref_path, format="wav")
            print(f"Эталонный голосовой сегмент сохранен: {ref_path} ({end-start}ms)")
            return ref_path
        except Exception as e:
            print(f"Не удалось извлечь эталонный сегмент: {e}")
            return None
    
    def sync_audio_with_video(self, video_path, new_audio_path, output_path):
        """
        Синхронизирует новое аудио с видео (упрощенная версия)
        
        Args:
            video_path (str): Путь к оригинальному видео
            new_audio_path (str): Путь к новому аудио
            output_path (str): Путь для сохранения результата
        """
        try:
            print("Синхронизируем аудио с видео...")
            
            # Используем ffmpeg для замены аудио
            import subprocess
            
            cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", new_audio_path,
                "-map", "0:v:0",
                "-map", "1:a:0",
                "-c:v", "copy",
                "-c:a", "aac",
                "-ar", "48000",
                "-ac", "2",
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            
            if result.returncode == 0:
                print(f"Видео с новым аудио сохранено: {output_path}")
                return output_path
            else:
                print(f"Ошибка ffmpeg: {result.stderr}")
                return None
            
        except Exception as e:
            print(f"Ошибка при синхронизации: {e}")
            print("Убедитесь, что ffmpeg установлен и доступен в PATH")
            return None
    
    def _apply_prosody_to_audio(self, synthesized_audio_path, prosody_info, output_path, video_path=None):
        """
        Применяет просодию (паузы, интонацию) к синтезированному аудио
        
        Args:
            synthesized_audio_path (str): Путь к синтезированному аудио
            prosody_info (dict): Информация о просодии оригинального аудио
            output_path (str): Путь для сохранения результата
        
        Returns:
            str: Путь к обработанному аудио
        """
        try:
            print("Применяем просодию к синтезированному аудио...")
            
            # Загружаем синтезированное аудио
            synthesized_audio = AudioSegment.from_wav(synthesized_audio_path)
            
            # Применяем точную просодию для максимального сохранения оригинала
            if prosody_info:
                # Целевая длительность (мс): приоритет — длительность исходного ВИДЕО
                original_ms = int(prosody_info.get('duration', 0) * 1000)
                if video_path:
                    probed_ms = self._probe_video_duration_ms(video_path)
                    if probed_ms:
                        original_ms = probed_ms
                        print(f"Целевая длительность по видео: {original_ms}ms")
                
                # Применяем паузы на основе оригинального аудио (включая начальную тишину)
                if 'nonsilent_ranges' in prosody_info and len(prosody_info['nonsilent_ranges']) > 0:
                    nonsilent_ranges = prosody_info['nonsilent_ranges']
                    
                    # 1) Начальная тишина до первой речи
                    first_start = nonsilent_ranges[0][0]
                    if first_start > 50:
                        synthesized_audio = AudioSegment.silent(duration=int(first_start)) + synthesized_audio
                        print(f"Добавлена начальная тишина: {first_start}ms")
                    
                    # 2) Межфразовые паузы в конце
                    if len(nonsilent_ranges) > 1:
                        total_pause_time = 0
                        for i in range(len(nonsilent_ranges) - 1):
                            pause_start = nonsilent_ranges[i][1]
                            pause_end = nonsilent_ranges[i + 1][0]
                            pause_duration = pause_end - pause_start
                            if pause_duration > 50:
                                total_pause_time += pause_duration
                        if total_pause_time > 0:
                            avg_pause = total_pause_time / max(1, len(nonsilent_ranges) - 1)
                            synthesized_audio = synthesized_audio + AudioSegment.silent(duration=int(avg_pause))
                            print(f"Добавлена пауза: {avg_pause:.0f}ms")

                # Приводим длину аудио к длине оригинала: дополняем тишиной или подрезаем
                if original_ms and original_ms > 0:
                    cur_ms = len(synthesized_audio)
                    if cur_ms < original_ms:
                        pad = original_ms - cur_ms
                        synthesized_audio = synthesized_audio + AudioSegment.silent(duration=pad)
                        print(f"Добита длина до оригинала тишиной: +{pad}ms")
                    elif cur_ms > original_ms + 50:  # небольшой допуск
                        synthesized_audio = synthesized_audio[:original_ms]
                        print(f"Обрезано до длины оригинала: {original_ms}ms")
            
            # Сохраняем результат
            synthesized_audio.export(output_path, format="wav")
            
            print(f"Просодия применена: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Ошибка при применении просодии: {e}")
            # В случае ошибки просто копируем исходный файл
            import shutil
            shutil.copy2(synthesized_audio_path, output_path)
            return output_path

    def _probe_video_duration_ms(self, video_path: str):
        """Возвращает длительность видео в миллисекундах через ffprobe, либо None при ошибке."""
        try:
            import subprocess, json
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "json",
                video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode != 0:
                return None
            data = json.loads(result.stdout.strip() or "{}")
            dur = data.get("format", {}).get("duration")
            if dur is None:
                return None
            seconds = float(dur)
            return int(round(seconds * 1000))
        except Exception:
            return None
    
    def combine_audio_with_video(self, video_path, audio_path, output_path):
        """
        Объединяет аудио с видео (алиас для sync_audio_with_video)
        
        Args:
            video_path (str): Путь к видео файлу
            audio_path (str): Путь к аудио файлу
            output_path (str): Путь для сохранения результата
        
        Returns:
            bool: True если успешно, False если ошибка
        """
        result = self.sync_audio_with_video(video_path, audio_path, output_path)
        return result is not None

# Пример использования
if __name__ == "__main__":
    processor = VideoProcessor()
    
    # Пример извлечения аудио
    # audio_path = processor.extract_audio_from_video("input_video.mp4")
    
    # Пример анализа просодии
    # prosody = processor.analyze_audio_prosody(audio_path)

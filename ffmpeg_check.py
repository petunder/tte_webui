import subprocess
import sys
import os

def check_ffmpeg():
    try:
        # Проверка наличия FFmpeg
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print("FFmpeg уже установлен.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("FFmpeg не найден.")
        return False

def install_ffmpeg():
    # Установка FFmpeg в зависимости от операционной системы
    def install_ffmpeg():
        if os.name == 'nt':  # Windows
            print("Установка FFmpeg на Windows...")
            try:
                subprocess.run(["choco", "install", "ffmpeg", "-y"], check=True)
            except subprocess.CalledProcessError:
                print("Ошибка при установке FFmpeg с помощью Chocolatey.")
        elif os.name == 'posix':  # Unix-подобные системы (Linux, macOS)
            try:
                print("Установка FFmpeg на Unix-подобной системе...")
                subprocess.run(["sudo", "apt-get", "install", "ffmpeg", "-y"], check=True)
            except subprocess.CalledProcessError:
                print("Ошибка при установке FFmpeg с помощью apt-get. Пожалуйста, установите FFmpeg вручную.")
        else:
            print("Неизвестная операционная система. Установка невозможна.")


if not check_ffmpeg():
    install_ffmpeg()

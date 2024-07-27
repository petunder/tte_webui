# modules/audio_processor.py
from classes.audio import Audio

def process_audio(file, change_sample_rate, new_sample_rate, to_mono, apply_filter, filter_choice,
                  remove_silence, silence_duration, silence_threshold, audio_processing, lambd, tau, solver, nfe, output_format):
    audio = Audio(file)
    log = []

    if change_sample_rate:
        audio.change_sample_rate(new_sample_rate)
        log.append(f"Changed sample rate to {new_sample_rate}")

    if to_mono:
        audio.stereo_to_mono()
        log.append("Converted to mono")

    if apply_filter:
        audio.apply_filter(filter_choice)
        log.append(f"Applied filter: {filter_choice}")

    if audio_processing == "Denoise":
        audio.denoise_audio(lambd, tau, solver, nfe, lambda msg: log.append(msg))
        log.append("Applied denoising")
    elif audio_processing == "Enhance":
        audio.enhance_audio(lambd, tau, solver, nfe, lambda msg: log.append(msg))
        log.append("Applied audio enhancement")

    if remove_silence:
        audio.remove_silence(silence_duration, silence_threshold, lambda msg: log.append(msg))
        log.append("Removed silence")

    # Получаем файл в нужном формате
    output_file = audio.get_file_path(output_format)
    log.append(f"Converted to {output_format} format")

    return output_file, "\n".join(log)

import os
import time
import threading
import alsaaudio
import logging
from v4l2py.device import Device, VideoCapture


def set_driver_fps(fps, fps_type="camera"):
    import fcntl
    import struct
    SET_FPS_CMD = 0x40045501
    SET_CAMERA_FPS_CMD = 0x40044301
    device_path = '/dev/video4'
    cmd = SET_CAMERA_FPS_CMD if fps_type == "camera" else SET_FPS_CMD
    fd = os.open(device_path, os.O_RDWR)
    try:
        fcntl.ioctl(fd, cmd, struct.pack("I", fps))
        print(f"Successfully set {fps_type} FPS to {fps}")
    except IOError as e:
        print(f"Failed to set {fps_type} FPS: {e}")
    finally:
        os.close(fd)


def audio_stream(start_barrier, start_event, base_fps=20, sample_rate=16000, chunk_size=800):
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
    except PermissionError:
        pass

    pcm = alsaaudio.PCM(alsaaudio.PCM_CAPTURE, alsaaudio.PCM_NORMAL)
    pcm.setchannels(1)
    pcm.setrate(sample_rate)
    pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
    pcm.setperiodsize(chunk_size)
    pcm.set_tstamp_mode(alsaaudio.PCM_TSTAMP_ENABLE)
    pcm.set_tstamp_type(alsaaudio.PCM_TSTAMP_TYPE_GETTIMEOFDAY)

    print("[Audio] Warming up sensor...")
    for _ in range(5):
        length, data = pcm.read()
        time.sleep(1.0 / base_fps)

    start_barrier.wait()
    start_event.wait()
    print("[Audio] Start collecting...")

    while True:
        length, data = pcm.read()
        if length <= 0:
            continue
        # print(f"[Audio] Collected chunk at {time.time():.3f}")
        time.sleep(1.0 / base_fps)  # 控制采样节奏


def video_stream(start_barrier, start_event, width=640, height=480, base_fps=30):
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
    except PermissionError:
        pass

    with Device.from_id(4) as stream:
        video_capture = VideoCapture(stream, size=4)
        video_capture.set_format(width, height, 'YUYV')
        video_capture.set_fps(base_fps)
        set_driver_fps(base_fps, "camera")
        set_driver_fps(base_fps, "user")

        video_capture.open()

        print("[Video] Warming up sensor...")
        cnt = 0
        for frame in video_capture:
            cnt += 1
            if frame is None:
                raise RuntimeError(f"Video warmup failed at frame {cnt}")
            time.sleep(1.0 / base_fps)
            if cnt == 5:
                break

        start_barrier.wait()
        start_event.wait()
        print("[Video] Start collecting...")

        for frame in video_capture:
            if frame is None:
                continue
            # print(f"[Video] Collected frame at {time.time():.3f}")
            time.sleep(1.0 / base_fps)  # 控制采样节奏


def main():
    start_barrier = threading.Barrier(2)
    start_event = threading.Event()

    audio_thread = threading.Thread(target=audio_stream, args=(start_barrier, start_event))
    video_thread = threading.Thread(target=video_stream, args=(start_barrier, start_event))

    audio_thread.start()
    video_thread.start()

    time.sleep(2)  # 等主程序准备
    start_event.set()  # 统一放行！

    audio_thread.join()
    video_thread.join()


if __name__ == "__main__":
    main()

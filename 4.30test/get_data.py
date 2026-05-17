import time
import struct
import numpy as np
import cv2
import os
from multiprocessing import shared_memory, resource_tracker

# ==============================================================================
# 配置信息
# ==============================================================================
SHM_NAME = "shm_ar_video"
SHM_HEADER_SIZE = 16

# 自动生成带有时间戳的数据集文件夹，防止覆盖之前采集的数据
SESSION_ID = int(time.time())
SAVE_DIR = f"dataset/frames_{SESSION_ID}" 

# 创建保存目录
os.makedirs(SAVE_DIR, exist_ok=True)

def remove_shm_from_resource_tracker():
    try: 
        resource_tracker.unregister('/' + SHM_NAME, 'shared_memory')
    except Exception: 
        pass

def main():
    print("======================================================")
    print(" 📸 极简原始流采集工具 (无损图片序列专用)")
    print("======================================================")
    print(f"📁 数据集将保存在目录: {SAVE_DIR}")
    print("--> 📡 等待接入共享内存...")
    
    shm = None
    while True:
        try:
            shm = shared_memory.SharedMemory(name=SHM_NAME)
            remove_shm_from_resource_tracker()
            print("✅ 成功接入共享内存！开始疯狂抓拍...")
            break
        except FileNotFoundError:
            time.sleep(1.0)

    last_fid = 0
    frame_count = 0

    try:
        while True:
            # 1. 读取共享内存头部信息 (获取帧ID、宽、高)
            header = bytes(shm.buf[:SHM_HEADER_SIZE])
            fid, w, h = struct.unpack('QII', header)
            
            # 2. 帧防抖：如果帧ID没有变化，说明上一帧还没更新，稍作等待
            if fid == last_fid:
                time.sleep(0.002) 
                continue
            last_fid = fid

            # 3. 读取原始画面数据
            img_view = np.ndarray((h, w, 3), dtype=np.uint8, buffer=shm.buf[SHM_HEADER_SIZE : SHM_HEADER_SIZE+w*h*3])
            frame = img_view.copy()
            
            # 4. 画面还原 (与推流端保持一致，倒转并修正颜色通道)
            frame = cv2.flip(frame, 0)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # 5. 无损保存为 PNG 格式
            # 命名格式为 frame_000000.png, frame_000001.png，方便按顺序查看
            img_path = os.path.join(SAVE_DIR, f"frame_{frame_count:06d}.png")
            cv2.imwrite(img_path, frame)

            frame_count += 1
            
            # 每采集 30 张打印一次进度，避免控制台刷新太快卡顿
            if frame_count % 30 == 0:
                print(f"✅ 已保存 {frame_count} 张无损图片...", end='\r')

    except KeyboardInterrupt:
        print("\n\n🛑 收到中断信号 (Ctrl+C)，停止采集...")
    finally:
        if shm: 
            try: 
                shm.close()
            except: 
                pass
        print(f"💾 数据集采集完成！共计保存了 {frame_count} 张无损图像。")
        print(f"📂 图像位置: {os.path.abspath(SAVE_DIR)}")

if __name__ == "__main__":
    main()
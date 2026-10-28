# -*- coding: utf-8 -*-
"""换脸（锁脸）脚本 —— 跑在云端 4090 上

用途：把「主角的脸」换成指定的一张脸，用来保证成片里头必须是本人。
为什么放云端：换脸要 8G 以上显存，本机 4G 跑不动。

用法：
    python /root/face_swap.py 脸照片.jpg 目标视频.mp4 输出.mp4

支持：目标可以是视频（.mp4/.mov）也可以是图片（.jpg/.png）
"""
import os
import sys
import glob
import cv2
import numpy as np

# ---------- 找模型 ----------
def find_buffalo():
    """LivePortrait 下载时自带了一份 insightface 的 buffalo_l，先找它"""
    pats = [
        '/root/LivePortrait/pretrained_weights/**/buffalo_l',
        '/root/models/**/buffalo_l',
        '/root/**/buffalo_l',
    ]
    for p in pats:
        hits = glob.glob(p, recursive=True)
        if hits:
            return hits[0]
    return None


def main():
    if len(sys.argv) < 3:
        print('用法: python face_swap.py 脸照片 目标文件 [输出文件]')
        return 1

    src_face = sys.argv[1]
    target = sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else '/root/swap_out.mp4'

    import insightface
    from insightface.app import FaceAnalysis
    from insightface.model_zoo import get_model

    print('insightface', insightface.__version__)

    buf = find_buffalo()
    print('buffalo_l 位置:', buf)
    if buf:
        root = os.path.dirname(os.path.dirname(buf))  # .../insightface
        app = FaceAnalysis(name='buffalo_l', root=root,
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    else:
        # 没有本地模型包时，让它自己去下载（云上能连 hf-mirror 才行）
        app = FaceAnalysis(name='buffalo_l',
                           providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # ---------- 源脸 ----------
    img = cv2.imread(src_face)
    if img is None:
        print('读不到脸照片:', src_face)
        return 2
    faces = app.get(img)
    print('源图人脸数:', len(faces))
    if not faces:
        print('✘ 源图里没检测到人脸，换一张正脸照')
        return 3
    src_f = faces[0]

    # ---------- 换脸模型 ----------
    model_path = '/root/models/inswapper_128.onnx'
    if not os.path.exists(model_path):
        cands = glob.glob('/root/**/inswapper_128*.onnx', recursive=True)
        model_path = cands[0] if cands else model_path
    print('换脸模型:', model_path, '| 存在:', os.path.exists(model_path))
    if not os.path.exists(model_path):
        print('✘ 换脸模型没下载，先下 inswapper_128.onnx')
        return 4
    swapper = get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # ---------- 目标是图片 ----------
    if target.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        dst = cv2.imread(target)
        tfaces = app.get(dst)
        print('目标图片人脸数:', len(tfaces))
        if not tfaces:
            print('✘ 目标图里没人脸')
            return 5
        for f in tfaces:
            dst = swapper.get(dst, f, src_f, paste_back=True)
        cv2.imwrite(out_path, dst)
        print('✔ 出图:', out_path)
        return 0

    # ---------- 目标是视频 ----------
    cap = cv2.VideoCapture(target)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print('视频 %dx%d fps=%.1f 总帧=%d' % (w, h, fps, total))

    tmp = out_path + '.noaudio.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    n = 0
    hit = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tfaces = app.get(frame)
        if tfaces:
            hit += 1
            for f in tfaces:
                frame = swapper.get(frame, f, src_f, paste_back=True)
        vw.write(frame)
        if n % 20 == 0:
            print('  已处理 %d/%d 帧（检测到脸 %d 帧）' % (n, total, hit), flush=True)
    cap.release()
    vw.release()
    print('完成 %d 帧，有脸 %d 帧' % (n, hit))

    # 合回原声（有 ffmpeg 就合，没有就用无声版）
    try:
        import subprocess
        r = subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                            '-i', tmp, '-i', target, '-map', '0:v:0', '-map', '1:a:0?',
                            '-c:v', 'libx264', '-crf', '18', '-c:a', 'aac', '-shortest', out_path],
                           capture_output=True, timeout=600)
        if r.returncode == 0 and os.path.exists(out_path):
            os.remove(tmp)
            print('✔ 出片（带原声）:', out_path)
        else:
            os.rename(tmp, out_path)
            print('✔ 出片（无声，ffmpeg 没合成）:', out_path)
    except Exception as e:
        os.rename(tmp, out_path)
        print('✔ 出片（无声）:', out_path, '|', str(e)[:80])
    return 0


if __name__ == '__main__':
    sys.exit(main())

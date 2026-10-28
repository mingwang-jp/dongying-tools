# -*- coding: utf-8 -*-
"""换脸（锁脸）脚本 v2 —— 跑在云端 4090 上

v2 修的问题（2026-10-28 晚）：
  原来直接用 FaceAnalysis 拿 source_face.normed_embedding，只要「认人模型」
  (w600k_r50.onnx) 没被 insightface 加载上，这个字段就是 None，然后崩在
  inswapper.py:54 `source_face.normed_embedding.reshape`。
  v2 做三件事：
    1. 主动挑「带 w600k_r50.onnx 的 buffalo_l」，不再拿到半残的包
    2. 打印到底加载了哪几个模型，一眼看出缺什么
    3. 如果 recognition 没加载 / embedding 还是 None，直接手动跑 ArcFace 兜底

用法：
    python face_swap.py 脸照片.jpg 目标视频.mp4 输出.mp4
"""
import os
import sys
import glob
import cv2
import numpy as np


def find_buffalo():
    """找 buffalo_l 目录；优先返回带 w600k_r50.onnx 的那个完整包"""
    hits = []
    for p in ('/root/models/**/buffalo_l', '/root/insightface/**/buffalo_l',
              '/root/LivePortrait/**/buffalo_l', '/root/**/buffalo_l'):
        hits += glob.glob(p, recursive=True)
    hits = list(dict.fromkeys(hits))
    if not hits:
        return None, []
    full = [h for h in hits if os.path.exists(os.path.join(h, 'w600k_r50.onnx'))]
    return (full[0] if full else hits[0]), hits


def make_app(buf):
    from insightface.app import FaceAnalysis
    root = os.path.dirname(os.path.dirname(buf))          # .../insightface
    last = None
    for prov in (['CUDAExecutionProvider', 'CPUExecutionProvider'], ['CPUExecutionProvider']):
        try:
            app = FaceAnalysis(name='buffalo_l', root=root,
                               allowed_modules=['detection', 'recognition'],
                               providers=prov)
            app.prepare(ctx_id=0 if 'CUDAExecutionProvider' in prov else -1, det_size=(640, 640))
            return app, root, prov[0]
        except Exception as e:
            last = e
            print('  建 FaceAnalysis 失败(%s)：%s' % (prov[0], str(e)[:120]))
    raise RuntimeError('FaceAnalysis 起不来：%s' % last)


def ensure_embedding(root, img, face):
    """保证 face 上有 normed_embedding；没有就用 ArcFace 手动算"""
    emb = getattr(face, 'normed_embedding', None)
    if emb is not None:
        return emb
    print('  ! 这个脸没有 embedding，手动补算…')
    from insightface.model_zoo import get_model
    arc = None
    for cand in (os.path.join(root, 'models', 'buffalo_l', 'w600k_r50.onnx'),
                 '/root/w600k_r50.onnx'):
        if os.path.exists(cand):
            try:
                arc = get_model(cand, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                arc.prepare(ctx_id=0)
                print('  用 ArcFace:', cand)
                break
            except Exception as e:
                print('  载入 ArcFace 失败:', str(e)[:100])
                arc = None
    if arc is None:
        return None
    try:
        arc.get(img, face)          # insightface 会写入 face.normed_embedding
    except Exception as e:
        print('  ArcFace 计算失败:', str(e)[:120])
        return None
    return getattr(face, 'normed_embedding', None)


def main():
    if len(sys.argv) < 3:
        print('用法: python face_swap.py 脸照片 目标文件 [输出文件]')
        return 1
    src_face, target = sys.argv[1], sys.argv[2]
    out_path = sys.argv[3] if len(sys.argv) > 3 else '/root/swap_out.mp4'

    buf, all_hits = find_buffalo()
    print('buffalo_l 候选:', all_hits)
    print('选中:', buf)
    if not buf:
        print('X 找不到 buffalo_l 目录')
        return 2
    print('含 w600k_r50:', os.path.exists(os.path.join(buf, 'w600k_r50.onnx')))
    print('目录内文件:', sorted(os.listdir(buf))[:10])

    app, root, prov = make_app(buf)
    print('provider:', prov)
    loaded = sorted((app.models or {}).keys())
    print('已加载模型:', loaded)
    if 'recognition' not in loaded:
        print('! recognition 没加载上 —— 会走手动 ArcFace 兜底')

    from insightface.model_zoo import get_model

    # ---------- 源脸 ----------
    img = cv2.imread(src_face)
    if img is None:
        print('读不到脸照片:', src_face)
        return 3
    faces = app.get(img)
    print('源图人脸数:', len(faces))
    if not faces:
        print('X 源图没检测到人脸，换一张正脸照')
        return 4
    src_f = faces[0]
    if ensure_embedding(root, img, src_f) is None:
        print('X 源脸拿不到特征向量（embedding），无法换脸')
        print('   请确认 w600k_r50.onnx 在', os.path.join(root, 'models', 'buffalo_l'))
        return 5
    print('源脸特征 OK，维度:', len(src_f.normed_embedding))

    # ---------- 换脸模型 ----------
    model_path = '/root/models/inswapper_128.onnx'
    if not os.path.exists(model_path):
        cands = glob.glob('/root/**/inswapper_128*.onnx', recursive=True)
        model_path = cands[0] if cands else model_path
    print('换脸模型:', model_path, '| 存在:', os.path.exists(model_path))
    if not os.path.exists(model_path):
        print('X 缺 inswapper_128.onnx')
        return 6
    swapper = get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    # ---------- 目标：图片 ----------
    if target.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        dst = cv2.imread(target)
        tf = app.get(dst)
        print('目标图片人脸数:', len(tf))
        if not tf:
            print('X 目标图没人脸')
            return 7
        for f in tf:
            dst = swapper.get(dst, f, src_f, paste_back=True)
        cv2.imwrite(out_path, dst)
        print('OK 出图:', out_path)
        return 0

    # ---------- 目标：视频 ----------
    cap = cv2.VideoCapture(target)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print('视频 %dx%d fps=%.1f 总帧=%d' % (w, h, fps, total))

    tmp = out_path + '.noaudio.mp4'
    vw = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    n = hit = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
        tf = app.get(frame)
        if tf:
            hit += 1
            for f in tf:
                frame = swapper.get(frame, f, src_f, paste_back=True)
        vw.write(frame)
        if n % 20 == 0:
            print('  已处理 %d/%d 帧（有脸 %d）' % (n, total, hit), flush=True)
    cap.release()
    vw.release()
    print('完成 %d 帧，有脸 %d 帧' % (n, hit))

    try:
        import subprocess
        r = subprocess.run(['ffmpeg', '-y', '-hide_banner', '-loglevel', 'error',
                            '-i', tmp, '-i', target, '-map', '0:v:0', '-map', '1:a:0?',
                            '-c:v', 'libx264', '-crf', '18', '-c:a', 'aac', '-shortest', out_path],
                           capture_output=True, timeout=900)
        if r.returncode == 0 and os.path.exists(out_path):
            os.remove(tmp)
            print('OK 出片（带原声）:', out_path)
        else:
            os.rename(tmp, out_path)
            print('OK 出片（无声）:', out_path)
    except Exception as e:
        os.rename(tmp, out_path)
        print('OK 出片（无声）:', out_path, '|', str(e)[:80])
    return 0


if __name__ == '__main__':
    sys.exit(main())

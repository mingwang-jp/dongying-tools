# -*- coding: utf-8 -*-
"""换脸（锁脸）+ 人脸增强  v3 —— 跑在云端 4090 上

v3 新增（2026-10-28 深夜，老板反馈"太假/像蒙了层布/像马赛克"）：
  换脸模型 inswapper_128 出来的人脸只有 128x128 像素，贴回画面必然糊。
  业界标准做法是换完脸再跑一道 GFPGAN 人脸增强（FaceFusion 默认就带）。
  v3 在换脸之后对每张脸做人脸增强，把五官细节补回来。

  增强模型用 ONNX 版（不用装 basicsr/gfpgan 那一堆老库，直接 onnxruntime 跑）：
    /root/models/gfpgan14-fp16.onnx  （170MB，fp16，快）
    或 /root/models/GFPGANv1.4.onnx   （340MB，fp32）

v2 修的问题：
  insightface 的 normed_embedding 可能是 None（recognition 模型没加载上），
  现在会自动挑完整 buffalo_l，并用 ArcFace 手动兜底。

用法：
    python face_swap.py 脸照片.jpg 目标视频.mp4 输出.mp4
    python face_swap.py 脸照片.jpg 目标视频.mp4 输出.mp4 --no-enhance   # 关增强，对照用
"""
import os
import sys
import glob
import cv2
import numpy as np


# ---------------- 找模型 ----------------
def find_buffalo():
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
    root = os.path.dirname(os.path.dirname(buf))
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
    if getattr(face, 'normed_embedding', None) is not None:
        return face.normed_embedding
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
        arc.get(img, face)
    except Exception as e:
        print('  ArcFace 计算失败:', str(e)[:120])
        return None
    return getattr(face, 'normed_embedding', None)


# ---------------- 人脸增强（GFPGAN ONNX）----------------
class Enhancer:
    """GFPGAN v1.4 的 ONNX 版：输入 512x512 人脸，输出修好的 512x512"""

    def __init__(self):
        import onnxruntime as ort
        cands = ['/root/models/gfpgan14-fp16.onnx', '/root/models/GFPGANv1.4.onnx']
        cands += glob.glob('/root/**/gfpgan*.onnx', recursive=True)
        path = next((c for c in cands if os.path.exists(c)), None)
        if not path:
            raise RuntimeError('找不到 GFPGAN onnx（放 /root/models/ 下）')
        self.path = path
        prov = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        self.sess = ort.InferenceSession(path, providers=prov)
        inp = self.sess.get_inputs()[0]
        self.in_name = inp.name
        self.in_dtype = np.float16 if 'float16' in str(inp.type) else np.float32
        print('增强模型:', path)
        print('  输入:', inp.name, inp.shape, inp.type)
        print('  输出:', self.sess.get_outputs()[0].name, self.sess.get_outputs()[0].shape)

    def run(self, frame, bbox):
        """把 frame 里 bbox 这块脸增强后贴回去（羽化融合，不留硬边）"""
        H, W = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return frame
        m = int(max(w, h) * 0.35)                    # 多带一点额头和下巴
        cx1, cy1 = max(0, x1 - m), max(0, y1 - m)
        cx2, cy2 = min(W, x2 + m), min(H, y2 + m)
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return frame

        big = cv2.resize(crop, (512, 512), interpolation=cv2.INTER_LINEAR)
        x = big[:, :, ::-1].astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = x.transpose(2, 0, 1)[None].astype(self.in_dtype)

        out = self.sess.run(None, {self.in_name: x})[0][0].astype(np.float32)
        out = out.transpose(1, 2, 0)
        out = (out * 0.5 + 0.5) * 255.0
        out = np.clip(out, 0, 255)[:, :, ::-1]
        out = cv2.resize(out, (cx2 - cx1, cy2 - cy1), interpolation=cv2.INTER_LANCZOS4)

        mask = np.zeros(out.shape[:2], np.uint8)
        cv2.ellipse(mask, ((out.shape[1] - 1) // 2, (out.shape[0] - 1) // 2),
                    (int(out.shape[1] * 0.5), int(out.shape[0] * 0.5)), 0, 0, 360, 255, -1)
        mask = cv2.GaussianBlur(mask, (21, 21), 0)[:, :, None].astype(np.float32) / 255.0
        frame[cy1:cy2, cx1:cx2] = (out * mask + crop.astype(np.float32) * (1 - mask)).astype(np.uint8)
        return frame


# ---------------- 主流程 ----------------
def main():
    if len(sys.argv) < 3:
        print('用法: python face_swap.py 脸照片 目标文件 [输出文件] [--no-enhance]')
        return 1
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    use_enhance = '--no-enhance' not in sys.argv
    src_face, target = args[0], args[1]
    out_path = args[2] if len(args) > 2 else '/root/swap_out.mp4'

    buf, all_hits = find_buffalo()
    print('buffalo_l 候选:', all_hits)
    print('选中:', buf)
    if not buf:
        print('X 找不到 buffalo_l')
        return 2
    print('含 w600k_r50:', os.path.exists(os.path.join(buf, 'w600k_r50.onnx')))

    app, root, prov = make_app(buf)
    print('provider:', prov, '| 已加载模型:', sorted((app.models or {}).keys()))

    from insightface.model_zoo import get_model

    img = cv2.imread(src_face)
    if img is None:
        print('读不到脸照片:', src_face)
        return 3
    faces = app.get(img)
    print('源图人脸数:', len(faces))
    if not faces:
        print('X 源图没检测到人脸')
        return 4
    src_f = faces[0]
    if ensure_embedding(root, img, src_f) is None:
        print('X 源脸拿不到特征向量')
        return 5
    print('源脸特征 OK，维度:', len(src_f.normed_embedding))

    model_path = '/root/models/inswapper_128.onnx'
    if not os.path.exists(model_path):
        c = glob.glob('/root/**/inswapper_128*.onnx', recursive=True)
        model_path = c[0] if c else model_path
    print('换脸模型:', model_path, '| 存在:', os.path.exists(model_path))
    if not os.path.exists(model_path):
        print('X 缺 inswapper_128.onnx')
        return 6
    swapper = get_model(model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

    enh = None
    if use_enhance:
        try:
            enh = Enhancer()
        except Exception as e:
            print('! 增强不可用，只换脸不增强：', str(e)[:160])
    else:
        print('按参数要求：不启用增强')

    # -------- 目标：图片 --------
    if target.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        dst = cv2.imread(target)
        tf = app.get(dst)
        print('目标图片人脸数:', len(tf))
        if not tf:
            print('X 目标图没人脸')
            return 7
        boxes = [f.bbox.copy() for f in tf]
        for f in tf:
            dst = swapper.get(dst, f, src_f, paste_back=True)
        if enh:
            for b in boxes:
                dst = enh.run(dst, b)
        cv2.imwrite(out_path, dst)
        print('OK 出图:', out_path)
        return 0

    # -------- 目标：视频 --------
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
            boxes = [f.bbox.copy() for f in tf]
            for f in tf:
                frame = swapper.get(frame, f, src_f, paste_back=True)
            if enh:
                for b in boxes:
                    frame = enh.run(frame, b)
        vw.write(frame)
        if n % 20 == 0:
            print('  已处理 %d/%d 帧（有脸 %d）' % (n, total, hit), flush=True)
    cap.release()
    vw.release()
    print('完成 %d 帧，有脸 %d 帧' % (n, hit))

    # 换脸出来是 mp4v 编码，浏览器/播放器很多不认，这里直接用系统 ffmpeg 转 H.264 并合原声
    ff = '/usr/bin/ffmpeg' if os.path.exists('/usr/bin/ffmpeg') else 'ffmpeg'
    try:
        import subprocess
        r = subprocess.run([ff, '-y', '-hide_banner', '-loglevel', 'error',
                            '-i', tmp, '-i', target, '-map', '0:v:0', '-map', '1:a:0?',
                            '-c:v', 'libx264', '-crf', '18', '-c:a', 'aac', '-shortest', out_path],
                           capture_output=True, timeout=1800)
        if r.returncode == 0 and os.path.exists(out_path):
            os.remove(tmp)
            print('OK 出片（H.264 + 原声）:', out_path)
        else:
            print('! ffmpeg 失败:', r.stderr.decode('utf-8', 'ignore')[-300:])
            os.rename(tmp, out_path)
            print('OK 出片（mp4v 无转码）:', out_path)
    except Exception as e:
        os.rename(tmp, out_path)
        print('OK 出片（无转码）:', out_path, '|', str(e)[:80])
    return 0


if __name__ == '__main__':
    sys.exit(main())

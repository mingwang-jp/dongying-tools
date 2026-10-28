#!/bin/bash
# 并行换脸：把视频切成 N 段，同时跑，再拼回一条
#
# 用法：bash par_swap.sh 脸.jpg 视频.mp4 输出.mp4 [进程数，默认3]
# 例：  bash par_swap.sh _send_face.jpg _send_clip2.mp4 out8.mp4 3
#
# 什么时候用：视频 ≥ 30 秒。短片（< 15 秒）别用，启动开销吃掉收益。
# 原理：ffmpeg 无损切段 → N 个 face_swap.py 同时跑 → concat 拼回。
set -e

FACE="$1"
SRC="$2"
OUT="$3"
N="${4:-3}"

if [ -z "$FACE" ] || [ -z "$SRC" ] || [ -z "$OUT" ]; then
  echo "用法: bash par_swap.sh 脸.jpg 视频.mp4 输出.mp4 [进程数]"
  exit 1
fi

FF=/usr/bin/ffmpeg
FP=/usr/bin/ffprobe
W=$(mktemp -d /root/parswap.XXXXXX)
echo "临时目录: $W"

DUR=$("$FP" -v error -show_entries format=duration -of csv=p=0 "$SRC")
SEG=$(python3 -c "print(round(max(float('$DUR')/$N, 1.0), 3))")
echo "总时长 ${DUR}s，切成 $N 段，每段约 ${SEG}s"

echo "--- 切分 ---"
"$FF" -y -hide_banner -loglevel error -i "$SRC" -c copy -f segment \
      -segment_time "$SEG" -reset_timestamps 1 "$W/p%02d.mp4"
ls -1 "$W"/p*.mp4 | wc -l | xargs echo "切出段数:"

echo "--- 并行换脸（$N 个进程）---"
T0=$(date +%s)
i=0
for f in "$W"/p*.mp4; do
  i=$((i+1))
  idx=$(printf %02d "$i")
  ( python3 face_swap.py "$FACE" "$f" "$W/o$idx.mp4" > "$W/log$idx.txt" 2>&1 ; \
    echo "  段$idx 完成" ) &
done
wait
T1=$(date +%s)
echo "并行处理结束，用时 $((T1-T0)) 秒"

echo "--- 检查每段 ---"
for f in "$W"/o*.mp4; do
  [ -f "$f" ] && echo "  $(basename $f) $(du -h "$f" | cut -f1)" || echo "  缺 $(basename $f)"
done

echo "--- 拼接 ---"
: > "$W/list.txt"
for f in "$W"/o*.mp4; do echo "file '$f'" >> "$W/list.txt"; done
"$FF" -y -hide_banner -loglevel error -f concat -safe 0 -i "$W/list.txt" -c copy "$OUT" || \
"$FF" -y -hide_banner -loglevel error -f concat -safe 0 -i "$W/list.txt" \
      -c:v libx264 -crf 20 -c:a aac "$OUT"

T2=$(date +%s)
echo "=================================="
echo "完成: $OUT  ($(du -h "$OUT" | cut -f1))"
echo "总耗时 $((T2-T0)) 秒（其中并行处理 $((T1-T0)) 秒）"
echo "分段临时文件: $W"
echo "=================================="

import cv2
import numpy as np
import argparse
import os

def load_ts(tsfile):
    ts = []
    with open(tsfile, 'r') as f:
        for line in f:
            s = line.strip()
            if s: ts.append(float(s))
    return np.array(ts)

def detect_fps(path, fallback=30):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if (fps is None) or (fps < 1):
        print(f'[WARNING] 读取FPS失败，假定为{fallback}')
        return fallback
    return fps

def align_and_clip(
        videoA, tsA, videoB, tsB,
        maxdt=0.05, outA='synced_A.mp4', outB='synced_B.mp4',
        afps=None, bfps=None, outcsv=None):
    # 对齐
    idx_B = np.searchsorted(tsB, tsA)
    pairs = []
    for i, (ta, idx) in enumerate(zip(tsA, idx_B)):
        candidates = []
        for j in [idx-1, idx]:
            if 0 <= j < len(tsB):
                dt = abs(tsB[j] - ta)
                candidates.append((j, dt))
        if not candidates: continue
        best_j, best_dt = min(candidates, key=lambda x:x[1])
        if best_dt <= maxdt:
            pairs.append((i, ta, best_j, tsB[best_j], tsB[best_j]-ta))
    print(f'[INFO] 可对齐帧数: {len(pairs)}（允许阈值: {maxdt:.3f}s）')

    a_idx = [i for i,_,_,_,_ in pairs]
    b_idx = [j for _,_,j,_,_ in pairs]

    # 读取所需帧
    def load_frames(vfile, idxs):
        cap = cv2.VideoCapture(vfile)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        shape = None
        out = []
        for idx in idxs:
            if not (0 <= idx < total):
                out.append(np.zeros((int(cap.get(4)), int(cap.get(3)),3), np.uint8))
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                # 黑帧占位
                if shape is not None:
                    frame = np.zeros(shape, np.uint8)
                else:
                    frame = np.zeros((int(cap.get(4)), int(cap.get(3)),3), np.uint8)
            if shape is None and frame is not None:
                shape = frame.shape
            out.append(frame)
        cap.release()
        return out

    afps = afps or detect_fps(videoA)
    bfps = bfps or detect_fps(videoB)
    aframes = load_frames(videoA, a_idx)
    bframes = load_frames(videoB, b_idx)

    def write_video(frames, outpath, fps):
        if not frames: return
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))
        for frm in frames:
            writer.write(frm)
        writer.release()
        print(f'[INFO] 写入新视频: {outpath} 帧数={len(frames)}')

    write_video(aframes, outA, afps)
    write_video(bframes, outB, bfps)

    # 可选储存csv
    if outcsv:
        with open(outcsv, 'w') as f:
            f.write('A_frame,A_time,B_frame,B_time,B_minus_A(sec)\n')
            for row in pairs:
                f.write('%d,%.6f,%d,%.6f,%.6f\n' % row)
        print(f'[INFO] 对应配对表输出于: {outcsv}')

def auto_align(video_root, name, maxdt=0.05):
    """
    video_root ── 根目录
    name       ── 文件名（例如 out_20250805_190244.mp4）
    """
    aligned_dir = os.path.join(video_root, 'aligned')
    force_dir = os.path.join(video_root, 'force')
    img_dir = os.path.join(video_root, 'img')

    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)

    # 输入文件
    force_mp4 = os.path.join(force_dir, name)
    force_ts  = force_mp4 + '.frame_ts.txt'
    img_mp4   = os.path.join(img_dir, name)
    img_ts    = img_mp4 + '.frame_ts.txt'

    # 校验文件存在
    for f in [force_mp4, force_ts, img_mp4, img_ts]:
        if not os.path.isfile(f):
            raise FileNotFoundError(f'未找到文件: {f}')

    # 输出新名字：force_+日期名_aligned.mp4, img_+日期名_aligned.mp4
    base = os.path.splitext(name)[0]
    force_out = os.path.join(aligned_dir, f'force_{base.split("_",1)[-1]}_aligned.mp4')
    img_out   = os.path.join(aligned_dir, f'img_{base.split("_",1)[-1]}_aligned.mp4')
    csv_out   = os.path.join(aligned_dir, f'aligned_pair_{base.split("_",1)[-1]}.csv')

    print(f'[INFO] 对齐以下文件:\n  FORCE: {force_mp4}\n  IMG:   {img_mp4}')

    tsA = load_ts(force_ts)
    tsB = load_ts(img_ts)

    align_and_clip(
        force_mp4, tsA,
        img_mp4,   tsB,
        maxdt=maxdt,
        outA=force_out, outB=img_out,
        afps=None, bfps=None,
        outcsv=csv_out
    )

def main():
    parser = argparse.ArgumentParser(description='自动对齐并输出到 aligned 文件夹')
    parser.add_argument('--video_root', required=True, help='根目录')
    parser.add_argument('--name', required=True, help='如 out_20250805_190244.mp4，不带路径')
    parser.add_argument('--maxdt', type=float, default=0.05, help='最大同步阈值（秒）')
    args = parser.parse_args()
    auto_align(args.video_root, args.name, maxdt=args.maxdt)

if __name__ == '__main__':
    main()

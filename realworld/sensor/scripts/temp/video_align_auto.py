import cv2
import numpy as np
import argparse
import os

def load_ts(tsfile):
    """Load frame timestamp array from file."""
    ts = []
    with open(tsfile, 'r') as f:
        for line in f:
            s = line.strip()
            if s: ts.append(float(s))
    return np.array(ts)

def detect_fps(path, fallback=30):
    """Read FPS from video file or use fallback if unknown."""
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if (fps is None) or (fps < 1):
        print(f'[WARNING] Failed to read FPS, fallback to {fallback}')
        return fallback
    return fps

def write_aligned_video(vfile, idxs, outpath, fps):
    """Stream aligned frames by index from a video file and write to output, saving RAM."""
    cap = cv2.VideoCapture(vfile)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(outpath, fourcc, fps, (w, h))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    shape = (h, w, 3)
    for idx in idxs:
        if not (0 <= idx < total):
            frm = np.zeros(shape, np.uint8)
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frm = cap.read()
            if not ret or frm is None:
                frm = np.zeros(shape, np.uint8)
        writer.write(frm)
    writer.release()
    cap.release()
    print(f'[INFO] Output video: {outpath}, frames={len(idxs)}')

def align_and_clip(
        videoA, tsA, videoB, tsB,
        maxdt=0.05, outA='synced_A.mp4', outB='synced_B.mp4',
        afps=None, bfps=None, outcsv=None):
    """Find aligned frame pairs within maxdt, stream and save aligned videos and (optionally) correspondence CSV."""
    # Alignment: for every tsA find closest tsB within threshold
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
    print(f'[INFO] Number of aligned frames: {len(pairs)} (threshold: {maxdt:.3f}s)')

    a_idx = [i for i,_,_,_,_ in pairs]
    b_idx = [j for _,_,j,_,_ in pairs]

    afps = afps or detect_fps(videoA)
    bfps = bfps or detect_fps(videoB)

    # Write aligned videos by streaming (no full frame list in RAM)
    write_aligned_video(videoA, a_idx, outA, afps)
    write_aligned_video(videoB, b_idx, outB, bfps)

    if outcsv:
        with open(outcsv, 'w') as f:
            f.write('A_frame,A_time,B_frame,B_time,B_minus_A(sec)\n')
            for row in pairs:
                f.write('%d,%.6f,%d,%.6f,%.6f\n' % row)
        print(f'[INFO] Alignment pairs CSV saved: {outcsv}')

def auto_align_all(root, maxdt=0.05):
    """
    Recursively search for same-named mp4 files in /force and /img under root,
    align each pair, and save outputs in /aligned.
    """
    aligned_dir = os.path.join(root, 'aligned')
    force_dir = os.path.join(root, 'force')
    img_dir = os.path.join(root, 'img')
    if not os.path.exists(aligned_dir):
        os.makedirs(aligned_dir)

    def list_files(d):
        """List all .mp4 files in given directory."""
        return set([f for f in os.listdir(d) if f.lower().endswith('.mp4')])

    # Only align videos with the same file name in both folders
    force_files = list_files(force_dir)
    img_files = list_files(img_dir)
    pair_names = sorted(force_files & img_files)

    if not pair_names:
        print(f'[ERROR] No paired .mp4 files found in both force/ and img/!')
        return

    for name in pair_names:
        force_mp4 = os.path.join(force_dir, name)
        force_ts  = force_mp4 + '.frame_ts.txt'
        img_mp4   = os.path.join(img_dir, name)
        img_ts    = img_mp4 + '.frame_ts.txt'
        # Check files exist
        for f in [force_mp4, force_ts, img_mp4, img_ts]:
            if not os.path.isfile(f):
                print(f'[WARNING] Missing related file, skipping: {f}')
                break
        else:  # Only process if all files are present
            base = os.path.splitext(name)[0]
            force_out = os.path.join(aligned_dir, f'force_{base}_aligned.mp4')
            img_out   = os.path.join(aligned_dir, f'img_{base}_aligned.mp4')
            csv_out   = os.path.join(aligned_dir, f'aligned_pair_{base}.csv')
            print(f'[INFO] Aligning: {name}')
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
    parser = argparse.ArgumentParser(
        description='Batch align videos with same names in /force and /img and output to /aligned'
    )
    parser.add_argument('--root', required=True, help='Root directory path')
    parser.add_argument('--maxdt', type=float, default=0.05, help='Max alignment error in seconds')
    args = parser.parse_args()
    auto_align_all(args.root, maxdt=args.maxdt)

if __name__ == '__main__':
    main()
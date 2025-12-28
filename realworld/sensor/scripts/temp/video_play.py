import cv2



path1 = 'video/aligned/force_20250805_192335_aligned.mp4'
path2 = 'video/aligned/img_20250805_192335_aligned.mp4'

# 目标窗口尺寸
WIDTH, HEIGHT = 1280, 720

cap1 = cv2.VideoCapture(path1)
cap2 = cv2.VideoCapture(path2)

while True:
    ret1, frame1 = cap1.read()
    ret2, frame2 = cap2.read()

    if not ret1 or not ret2:
        break

    # 计算各自要缩放到的尺寸
    target_w, target_h = WIDTH // 2, HEIGHT

    frame1_resized = cv2.resize(frame1, (target_w, target_h))
    frame2_resized = cv2.resize(frame2, (target_w, target_h))

    combined = cv2.hconcat([frame1_resized, frame2_resized])

    cv2.imshow('Aligned Videos (1920x1080)', combined)
    # 可选：设置窗口为指定大小（非全屏，如果是全屏则用 cv2.WND_PROP_FULLSCREEN）
    cv2.namedWindow('Aligned Videos (1920x1080)', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Aligned Videos (1920x1080)', WIDTH, HEIGHT)

    if cv2.waitKey(30) & 0xFF == 27:  # 按Esc退出
        break

cap1.release()
cap2.release()
cv2.destroyAllWindows()
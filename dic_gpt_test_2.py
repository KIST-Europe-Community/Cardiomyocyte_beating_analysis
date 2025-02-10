import cv2
import numpy as np
import os

def compute_divergence(vx, vy):
    """
    vx, vy: Optical Flow의 x, y 성분 (shape=(H,W))
    divergence = ∂vx/∂x + ∂vy/∂y 를 근사적으로 계산합니다.
    """
    dvx_dy, dvx_dx = np.gradient(vx)
    dvy_dy, dvy_dx = np.gradient(vy)
    return dvx_dx + dvy_dy

def strain_to_bgr_bluetored(strain, vmin=-1.0, vmax=1.0):
    """
    strain(부호 있는 2D 배열)을 파랑(음수)↔흰색(0)↔빨강(양수)으로 변환.
    vmin..vmax 범위로 정규화하고, 0.5를 경계로 음수 영역(파랑→흰색),
    양수 영역(흰색→빨강)으로 선형 보간합니다.
    """
    norm = (strain - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)  # [0..1]로 제한
    
    bgr = np.zeros((strain.shape[0], strain.shape[1], 3), dtype=np.float32)
    
    blue  = np.array([255,   0,   0], dtype=np.float32)  # BGR
    white = np.array([255, 255, 255], dtype=np.float32)
    red   = np.array([  0,   0, 255], dtype=np.float32)
    
    neg_mask = (norm < 0.5)
    ratio_neg = norm[neg_mask] / 0.5  # 0..0.5 → 0..1
    for c in range(3):
        bgr[..., c][neg_mask] = (1 - ratio_neg) * blue[c] + ratio_neg * white[c]
    
    pos_mask = (norm >= 0.5)
    ratio_pos = (norm[pos_mask] - 0.5) / 0.5  # 0.5..1 → 0..1
    for c in range(3):
        bgr[..., c][pos_mask] = (1 - ratio_pos) * white[c] + ratio_pos * red[c]
    
    return bgr.astype(np.uint8)

def generate_color_legend_bluetored(height, vmin, vmax, legend_width=50, font_scale=0.5):
    """
    세로(height) 크기에 맞춰, vmin~vmax 구간의 색상 그라디언트 범례를 생성.
    (음수=파랑, 0=흰색, 양수=빨강)
    """
    legend = np.zeros((height, legend_width, 3), dtype=np.uint8)
    
    for row in range(height):
        # row=0 -> vmax, row=height-1 -> vmin
        norm = 1.0 - (row / (height - 1 + 1e-8))  # [0..1]
        val  = vmin + norm * (vmax - vmin)       # [vmin..vmax]
        
        color_1x1 = strain_to_bgr_bluetored(
            np.array([[val]], dtype=np.float32), vmin, vmax
        )
        legend[row, :, :] = color_1x1[0, 0, :]
    
    legend_out = legend.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 상단 레이블 (vmax)
    cv2.putText(
        legend_out,
        f"{vmax:.2f}",
        (2, 15),
        font,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    # 중간(대략 0) 표기 (선택)
    mid_y = height // 2
    cv2.putText(
        legend_out,
        "0",
        (2, mid_y),
        font,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    # 하단 레이블 (vmin)
    cv2.putText(
        legend_out,
        f"{vmin:.2f}",
        (2, height - 10),
        font,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return legend_out

def visualize_strain_by_divergence_farneback_dynamic_range(
    video_path,
    output_path="strain_dynamic_range.avi",
    decay_factor=0.95,    # 시간 스무딩 강도
    blur_ksize=15,        # 공간적 Blur 커널 크기
    flow_threshold=0.0,   # 소값 제거 임계치
    slow_factor=1.0,
    legend_width=50
):
    """
    Farnebäck Optical Flow → divergence(수축/이완) 계산.
    - 시간적 스무딩(지수평활)
    - 매 프레임마다 [1%, 99%] Percentile로 vmin, vmax 추정 → 컬러맵 자동 스케일링
    - 음수 → 파랑, 양수 → 빨강 (0 → 흰색)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {video_path}")
        return
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] FPS={orig_fps}, Size=({width}x{height})")
    
    out_fps = orig_fps / slow_factor
    out_width = width * 2 + legend_width  # (왼)원본 + (중간)컬러맵 + (오른쪽)범례
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read the first frame.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 시간 스무딩(accumulation)용
    acc_strain = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Farnebäck Optical Flow
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )
        vx = flow[..., 0]
        vy = flow[..., 1]
        
        # 발산(divergence) 계산
        divergence = compute_divergence(vx, vy)
        
        # 공간적 Blur
        if blur_ksize > 1:
            divergence = cv2.GaussianBlur(divergence, (blur_ksize, blur_ksize), 0)
        
        # 작은 값 제거 (옵션)
        if flow_threshold > 0:
            divergence[np.abs(divergence) < flow_threshold] = 0
        
        # 시간 스무딩 (지수평활)
        if acc_strain is None:
            acc_strain = divergence.astype(np.float32)
        else:
            acc_strain = (
                decay_factor * acc_strain
                + (1 - decay_factor) * divergence
            ).astype(np.float32)
        
        # === (중요) 동적 범위(Percentile) 계산 ===
        low_pct = np.percentile(acc_strain, 1)   # 하위 1% 지점
        high_pct = np.percentile(acc_strain, 99) # 상위 99% 지점
        
        # 혹시 분포가 너무 좁다면 방어 처리
        if np.isclose(low_pct, high_pct):
            low_pct, high_pct = -0.001, 0.001
        
        # 컬러맵 생성 (Blue↔White↔Red)
        color_map = strain_to_bgr_bluetored(
            acc_strain,
            vmin=low_pct,
            vmax=high_pct
        )
        
        # 범례
        legend_img = generate_color_legend_bluetored(
            height=height,
            vmin=low_pct,
            vmax=high_pct,
            legend_width=legend_width,
            font_scale=0.5
        )
        
        # 합치기
        combined_frame = np.full((height, out_width, 3), 255, dtype=np.uint8)
        # (왼) 원본 영상
        combined_frame[:, :width] = frame
        # (중간) 시각화
        combined_frame[:, width:width*2] = color_map
        # (오른쪽) 범례
        combined_frame[:, width*2:] = legend_img
        
        out.write(combined_frame)
        
        prev_gray = current_gray
    
    cap.release()
    out.release()
    print(f"[DONE] '{output_path}'에 결과가 저장되었습니다.")

if __name__ == "__main__":
    # 예시 사용
    video_file = "/data/cardio_data/Movie-1.mp4"
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = "/home/beating_contest"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f"{video_name}_dynamic_range.avi")
    
    visualize_strain_by_divergence_farneback_dynamic_range(
        video_path=video_file,
        output_path=output_path,
        decay_factor=0.8,  # 시간 스무딩
        blur_ksize=15,
        flow_threshold=0.0, # 노이즈 억제를 위해서 적절히 조정 가능
        slow_factor=1.0,
        legend_width=50
    )

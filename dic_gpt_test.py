import cv2
import numpy as np
import os
import sys

def compute_divergence(vx, vy):
    """
    vx, vy: Optical Flow의 x, y 성분 (shape=(H,W))
    divergence = ∂vx/∂x + ∂vy/∂y 를 근사적으로 계산합니다.
    
    np.gradient( )는 (d/dy, d/dx) 순서로 반환하므로,
    vx에 대해 gradient를 구하면 [dvx_dy, dvx_dx] 형태가 됩니다.
    
    따라서 divergence = dvx_dx + dvy_dy.
    """
    dvx_dy, dvx_dx = np.gradient(vx)
    dvy_dy, dvy_dx = np.gradient(vy)
    return dvx_dx + dvy_dy

def strain_to_bgr_bluetored(strain, vmin=-1.0, vmax=1.0):
    """
    strain(부호 있는 2D 배열)을 파랑(음수)↔흰색(0)↔빨강(양수)으로 변환.
    vmin~vmax 범위로 정규화 후, 0.5 지점을 '흰색'으로 놓고,
    그 이하(음수)는 파랑→흰색, 그 이상(양수)는 흰색→빨강으로 선형 보간.
    """
    # 1) [vmin..vmax] → [0..1]로 정규화
    norm = (strain - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)  # [0..1]
    
    # 결과 BGR
    bgr = np.zeros((strain.shape[0], strain.shape[1], 3), dtype=np.float32)
    
    # 파랑 / 흰색 / 빨강 (OpenCV BGR 순)
    blue  = np.array([255,   0,   0], dtype=np.float32)
    white = np.array([255, 255, 255], dtype=np.float32)
    red   = np.array([  0,   0, 255], dtype=np.float32)
    
    # 2) 0.5를 경계로 음수/양수 분할
    neg_mask = (norm < 0.5)
    pos_mask = (norm >= 0.5)
    
    # a) neg_mask → (파랑~흰색) 보간
    ratio_neg = (norm[neg_mask] / 0.5)  # 0..0.5 → 0..1
    for c in range(3):
        bgr[..., c][neg_mask] = (1 - ratio_neg) * blue[c] + ratio_neg * white[c]
    
    # b) pos_mask → (흰색~빨강) 보간
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
        # 위에서 아래로 갈수록 값이 작아진다고 가정
        norm = 1.0 - (row / (height - 1 + 1e-8))  # 0..1
        val = vmin + norm * (vmax - vmin)        # vmin..vmax
        
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
    # 중앙 (대략 0)
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

def visualize_strain_by_divergence_farneback(
    video_path,
    output_path="strain_bluetored.avi",
    decay_factor=0.95,
    blur_ksize=15,
    flow_threshold=0.0,
    flow_magnitude_threshold=0.5,
    vmin=-0.3,
    vmax=0.3,
    slow_factor=1.0,
    legend_width=50
):
    """
    Farnebäck Optical Flow → (vx, vy) → divergence 계산 → 
    '수축(음수)=파랑, 이완(양수)=빨강' 시각화 후 영상으로 저장.
    
    - (시간 스무딩): acc_strain = alpha*acc_strain + (1-alpha)*current_div
    - (공간 스무딩): blur_ksize로 GaussianBlur
    - (flow_threshold): |divergence|가 threshold 이하인 곳은 0 처리(선택)
    - (vmin, vmax): 발산(수축/이완) 값을 어느 범위로 색맵 표시할지
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오를 열 수 없습니다: {video_path}")
        return
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] FPS={orig_fps}, Size=({width}x{height})")
    
    out_fps = orig_fps / slow_factor
    out_width = width * 2 + legend_width  # (왼)원본 + (중간)시각화 + (오른쪽)범례
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 시간 스무딩(accumulation)용 변수
    acc_strain = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1) Farnebäck Optical Flow 계산
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray,
            None,
            0.5,     # pyr_scale
            5,       # levels
            21,      # winsize
            3,       # iterations
            7,       # poly_n
            1.5,     # poly_sigma
            0        # flags
        )
        vx = flow[..., 0]
        vy = flow[..., 1]
        
        # 추가: flow magnitude가 작은 부분 마스킹
        flow_magnitude = np.sqrt(vx**2 + vy**2)
        flow_mask = flow_magnitude < flow_magnitude_threshold
        vx[flow_mask] = 0
        vy[flow_mask] = 0
        
        # 2) 발산(divergence) 계산 => 간단히 '수축/이완' 지표로 활용
        divergence = compute_divergence(vx, vy)
        
        # 3) 공간적 스무딩 (옵션)
        if blur_ksize > 1:
            divergence = cv2.GaussianBlur(divergence, (blur_ksize, blur_ksize), 0)
        
        # 4) 너무 작은 값은 0으로 (노이즈 억제)
        if flow_threshold > 0:
            divergence[np.abs(divergence) < flow_threshold] = 0
        
        # 5) 시간 스무딩 (지수평활)
        if acc_strain is None:
            acc_strain = divergence.astype(np.float32)
        else:
            acc_strain = (
                decay_factor * acc_strain
                + (1 - decay_factor) * divergence
            ).astype(np.float32)
        
        # 6) 컬러맵: 파랑↔흰색↔빨강
        color_map = strain_to_bgr_bluetored(
            acc_strain,
            vmin=vmin,
            vmax=vmax
        )
        
        # 7) 범례(legend) 생성
        legend_img = generate_color_legend_bluetored(
            height=height,
            vmin=vmin,
            vmax=vmax,
            legend_width=legend_width,
            font_scale=0.5
        )
        
        # 8) 원본 + 컬러맵 + 범례 합치기
        combined_frame = np.full((height, out_width, 3), 255, dtype=np.uint8)
        # 왼쪽: 원본 영상
        combined_frame[:, :width] = frame
        # 중간: 발산(수축/이완) 시각화
        combined_frame[:, width:width*2] = color_map
        # 오른쪽: 범례
        combined_frame[:, width*2:] = legend_img
        
        out.write(combined_frame)
        
        prev_gray = current_gray
    
    cap.release()
    out.release()
    print(f"[DONE] 결과가 '{output_path}'에 저장되었습니다.")

if __name__ == "__main__":
    # ======== 예시 사용법 ========
    # (1) 입력 비디오 경로
    video_file = "/data/videos/Isoprenaline/Normal_1.avi"
    # (2) 출력 비디오 경로
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    output_dir = "/home/calcium_imaging/temp"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(
        output_dir,
        f"{video_name}_divergence_bluetored.avi"
    )
    
    # 함수 호출
    visualize_strain_by_divergence_farneback(
        video_path=video_file,
        output_path=output_path,
        decay_factor=0.95,    # 0.9 → 0.95로 증가 (더 강한 시간적 스무딩)
        blur_ksize=21,        # 15 → 21로 증가
        flow_threshold=0.001,  # 0.0 → 0.001로 증가
        flow_magnitude_threshold=0.5,  # 추가
        vmin=-0.01,
        vmax=0.01,
        slow_factor=1.0,
        legend_width=50
    )

# if __name__ == "__main__":
#     video_file = sys.argv[1]
#     print(video_file)

#     video_name = os.path.splitext(os.path.basename(video_file))[0]
#     output_dir = "/home/beating_contest/vector_relxation_contraction"
#     os.makedirs(output_dir, exist_ok=True)
    
#     output_path = os.path.join(output_dir, f"{video_name}_divergence_bluetored.avi")

#     # 함수 호출
#     visualize_strain_by_divergence_farneback(
#         video_path=video_file,
#         output_path=output_path,
#         decay_factor=0.95,    # 0.9 → 0.95로 증가 (더 강한 시간적 스무딩)
#         blur_ksize=21,        # 15 → 21로 증가
#         flow_threshold=0.001,  # 0.0 → 0.001로 증가
#         flow_magnitude_threshold=0.5,  # 추가
#         vmin=-0.01,
#         vmax=0.01,
#         slow_factor=1.0,
#         legend_width=50
#     )

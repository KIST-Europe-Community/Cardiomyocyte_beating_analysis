import cv2
import numpy as np
import sys
import os

def div_to_bgr(div, vmin=-0.02, vmax=0.02):
    """
    divergence(div)를 [vmin, vmax] 구간으로 정규화한 뒤,
    수축(음수) → 빨강, 0 → 흰색, 이완(양수) → 파랑 으로 
    부드럽게 연결되는 컬러 맵(BGR)을 생성합니다.
    """
    # 1) [vmin, vmax] → [0, 1] 로 정규화
    norm = (div - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)  # [0..1]
    
    # 2) div=0 일 때의 norm 값
    mid = -vmin / (vmax - vmin)

    h, w = div.shape
    bgr = np.zeros((h, w, 3), dtype=np.float32)
    
    # 음수(수축) 영역 마스크 (norm < mid)
    mask_neg = (norm < mid)
    # 양수(이완) 영역 마스크 (norm > mid)
    mask_pos = (norm > mid)
    
    # div=0 근접 (norm == mid) → 흰색
    white = np.array([255, 255, 255], dtype=np.float32)
    # 빨간색 (BGR): [0, 0, 255]
    red   = np.array([0, 0, 255], dtype=np.float32)
    # 파란색 (BGR): [255, 0, 0]
    blue  = np.array([255, 0, 0], dtype=np.float32)
    
    # (A) 음수(수축) : 빨강 ~ 흰색 선형 보간
    #     norm=0 → div=vmin → pure red
    #     norm=mid → div=0 → pure white
    alpha_neg = np.zeros_like(norm, dtype=np.float32)
    alpha_neg[mask_neg] = norm[mask_neg] / mid  # [0..1]
    bgr[mask_neg] = (
        (1 - alpha_neg[mask_neg])[:, None]*red +
        alpha_neg[mask_neg][:, None]*white
    )
    
    # (B) 양수(이완) : 흰색 ~ 파랑 선형 보간
    #     norm=mid → div=0 → pure white
    #     norm=1 → div=vmax → pure blue
    alpha_pos = np.zeros_like(norm, dtype=np.float32)
    denom = (1 - mid) if (1 - mid) != 0 else 1e-6
    alpha_pos[mask_pos] = (norm[mask_pos] - mid) / denom
    bgr[mask_pos] = (
        (1 - alpha_pos[mask_pos])[:, None]*white +
        alpha_pos[mask_pos][:, None]*blue
    )
    
    # (C) div=0 근접 → 흰색
    zero_mask = np.isclose(norm, mid, atol=1e-7)
    bgr[zero_mask] = white
    
    return bgr.astype(np.uint8)


def visualize_strain_field(
    video_path,
    output_path="strain_output.avi",
    flow_threshold=1.0,
    decay_factor=0.8,
    slow_factor=2.0,
    vmin=-0.02,
    vmax=0.02,
    smoothing_factor=0.7,
    blur_ksize=15
):
    """
    1) Farnebäck Optical Flow로 (vx, vy) 계산
    2) divergence(= ∂vx/∂x + ∂vy/∂y) 구한 뒤
       - Gaussian Blur로 공간적 스무딩 (blur_ksize)
       - 너무 작은 Flow(mag < flow_threshold)는 무시(div=0 처리)
    3) 'div_smooth'에 지수적/시간적 스무딩 적용하여 갑작스런 요동 완화
       div_smooth = decay_factor * div_smooth + (1 - decay_factor) * div
    4) 컬러맵 변환(div_to_bgr): 음수=빨강, 0=흰색, 양수=파랑
    5) 결과 영상(배경 흰색)만 저장(원본 오버레이 없음)
    6) slow_factor로 FPS 낮춰(더 느리게 재생)
    """
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Original FPS={orig_fps}, Size=({width}x{height})")
    
    # out_fps를 낮춰 재생속도 늦추기
    out_fps = orig_fps / slow_factor
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 시간적 스무딩을 위한 div_smooth (float32) 준비
    div_smooth = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optical Flow (Farnebäck)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.2,
            0
        )
        
        vx = flow[..., 0]
        vy = flow[..., 1]
        mag = np.sqrt(vx**2 + vy**2)
        
        # divergence = ∂vx/∂x + ∂vy/∂y
        dvx_dy, dvx_dx = np.gradient(vx)
        dvy_dy, dvy_dx = np.gradient(vy)
        div = dvx_dx + dvy_dy
        
        # 공간적 스무딩(Blur) 커널 크게 해서 노이즈 완화
        div = cv2.GaussianBlur(div, (blur_ksize, blur_ksize), 0)
        
        # 너무 작은 흐름(Flow)은 무시
        div[mag < flow_threshold] = 0
        
        # 시간적 스무딩 (지수적/exp filter)
        if div_smooth is None:
            div_smooth = div.copy().astype(np.float32)
        else:
            div_smooth = (
                smoothing_factor * div_smooth +
                (1 - smoothing_factor) * div
            )
        
        # 컬러맵 변환
        # div_smooth는 float32, 시각화 위해 다시 np.float64 / np.float32
        div_color = div_to_bgr(div_smooth, vmin=vmin, vmax=vmax)
        
        # 결과 영상 저장 (배경=흰색, 음수=빨강, 양수=파랑)
        out.write(div_color)
        
        prev_gray = current_gray
    
    cap.release()
    out.release()
    print(f"[DONE] '{output_path}'에 흰 배경의 스트레인 필드 영상이 저장되었습니다.")


if __name__ == "__main__":
    visualize_strain_field(
        video_path="/home/calcium_imaging/Normal_1_calcium.mp4",
        output_path="/home/calcium_imaging/result/Movie_1_calcium_strain_field.mp4",
        flow_threshold=1.0,
        decay_factor=0.9
    )

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python strain_field.py <video_path>")
#         sys.exit(1)
    
#     video_file = sys.argv[1]  # 명령줄에서 비디오 파일 경로 받기
    
#     # 입력 파일명에서 확장자를 제외한 이름 추출
#     video_name = os.path.splitext(os.path.basename(video_file))[0]
    
#     # 결과 저장 경로 설정
#     output_dir = "/home/beating_contest/result_strain_video"
#     os.makedirs(output_dir, exist_ok=True)
#     output_path = os.path.join(output_dir, f"{video_name}_strain.avi")
    
#     visualize_strain_field(
#         video_path=video_file,
#         output_path=output_path,
#         flow_threshold=1.0,
#         blur_ksize=15,
#         smoothing_factor=0.7
#     )
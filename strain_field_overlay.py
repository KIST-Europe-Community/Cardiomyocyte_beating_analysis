import cv2
import numpy as np
import sys
import os

def div_to_bgr(div, vmin=-0.02, vmax=0.02):
    """
    divergence(div)를 [vmin, vmax] 구간으로 정규화한 뒤,
    수축(음수)=빨강, 0=흰색, 양수=파랑 으로 부드럽게 연결되는 컬러 맵(BGR) 생성
    """
    norm = (div - vmin) / (vmax - vmin)
    norm = np.clip(norm, 0, 1)  # [0..1]

    mid = -vmin / (vmax - vmin)  # div=0 일 때의 norm 값

    h, w = div.shape
    bgr = np.zeros((h, w, 3), dtype=np.float32)
    
    # 마스크
    mask_neg = (norm < mid)  # 수축
    mask_pos = (norm > mid)  # 이완

    white = np.array([255, 255, 255], dtype=np.float32)
    red   = np.array([0,   0, 255],  dtype=np.float32)
    blue  = np.array([255, 0,   0],  dtype=np.float32)
    
    # (A) 음수(수축): 빨강 ~ 흰색
    alpha_neg = np.zeros_like(norm, dtype=np.float32)
    alpha_neg[mask_neg] = norm[mask_neg] / mid
    bgr[mask_neg] = ((1 - alpha_neg[mask_neg])[:, None]*red +
                      alpha_neg[mask_neg][:, None]*white)
    
    # (B) 양수(이완): 흰색 ~ 파랑
    alpha_pos = np.zeros_like(norm, dtype=np.float32)
    denom = (1 - mid) if (1 - mid) != 0 else 1e-6
    alpha_pos[mask_pos] = (norm[mask_pos] - mid) / denom
    bgr[mask_pos] = ((1 - alpha_pos[mask_pos])[:, None]*white +
                      alpha_pos[mask_pos][:, None]*blue)

    # (C) div=0 근접 → 흰색
    zero_mask = np.isclose(norm, mid, atol=1e-7)
    bgr[zero_mask] = white
    
    return bgr.astype(np.uint8)

def visualize_strain_overlay(
    video_path,
    output_path,
    flow_threshold=1.0,
    blur_ksize=15,
    smoothing_factor=0.7,
    slow_factor=2.0,
    vmin=-0.02,
    vmax=0.02,
    overlay_alpha=0.3  # 스트레인 컬러맵을 얼마나 진하게 덮을지
):
    """
    - Farnebäck Optical Flow → (vx, vy)
    - divergence = ∂vx/∂x + ∂vy/∂y 계산 후
      - GaussianBlur(blur_ksize)로 공간 스무딩
      - mag < flow_threshold → div=0
      - 지수적 시간 스무딩(smoothing_factor)으로 노이즈 완화
    - div_to_bgr 로 (수축=빨강, 0=흰색, 이완=파랑) 컬러맵
    - 알파 블렌딩으로 원본 프레임 위에 오버레이
    - slow_factor로 FPS 낮춰, 재생속도 늦춤
    """

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return

    # 원본 영상 정보
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Original FPS={orig_fps}, Size=({width}x{height})")

    # 재생속도 늦추기
    out_fps = orig_fps / slow_factor

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("[ERROR] 첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # 시간적 스무딩용
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

        # divergence
        dvy_dy, dvy_dx = np.gradient(vy)
        div = dvx_dx + dvy_dy

        # 공간 스무딩
        div = cv2.GaussianBlur(div, (blur_ksize, blur_ksize), 0)

        # 너무 작은 flow는 div=0 처리
        div[mag < flow_threshold] = 0

        # 시간적 스무딩
        if div_smooth is None:
            div_smooth = div.astype(np.float32)
        else:
            div_smooth = (
                smoothing_factor * div_smooth +
                (1 - smoothing_factor) * div
            )
        
        # 컬러맵 변환 (배경=흰색)
        div_color = div_to_bgr(div_smooth, vmin=vmin, vmax=vmax)

        # 오버레이: frame(원본) + div_color(스트레인 컬러)
        # overlay_alpha : 스트레인 컬러맵의 가중치
        # (1 - overlay_alpha): 원본 영상의 가중치
        overlaid = cv2.addWeighted(
            frame, 
            1 - overlay_alpha, 
            div_color, 
            overlay_alpha, 
            0
        )

        # 최종 프레임 기록
        out.write(overlaid)
        prev_gray = current_gray.copy()

    cap.release()
    out.release()
    print(f"[DONE] '{output_path}'에 원본+스트레인 오버레이 영상이 저장되었습니다.")

if __name__ == "__main__":
    video_path="/home/calcium_imaging/Normal_1_calcium.mp4"
    output_path = "/home/calcium_imaging/result/Movie_1_calcium_strain_field_overlay.mp4"
    visualize_strain_overlay(video_path, output_path)

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python strain_field_overlay.py <video_path>")
#         sys.exit(1)
    
#     video_file = sys.argv[1]  # 명령줄에서 비디오 파일 경로 받기
    
#     # 결과 저장 디렉토리 설정
#     result_dir = "/home/beating_contest/result_strain_video_overlay"
#     os.makedirs(result_dir, exist_ok=True)
    
#     # 비디오 파일 이름 추출
#     video_name = os.path.splitext(os.path.basename(video_file))[0]
#     output_path = os.path.join(result_dir, f"{video_name}_strain_overlay.avi")
    
#     visualize_strain_overlay(video_file, output_path)

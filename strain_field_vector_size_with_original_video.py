import cv2
import numpy as np
import os

def mag_to_bgr_white2red(mag, vmin=0.0, vmax=5.0):
    """
    Optical Flow의 벡터 크기(magnitude)를 [vmin, vmax] 구간으로 정규화한 뒤,
    0일 때 = 흰색(255,255,255), vmax일 때 = 빨강(0,0,255) 으로 선형 보간하여
    배경이 흰색이고, 값이 커질수록 진한 빨강으로 보이는 단색 컬러 맵(BGR)을 생성합니다.
    """
    # 1) [vmin, vmax] → [0, 1]로 정규화
    norm = (mag - vmin) / (vmax - vmin + 1e-8)
    norm = np.clip(norm, 0, 1)  # [0..1]
    
    # 2) 0 → white (BGR: [255,255,255]), 1 → red (BGR: [0,0,255]) 선형 보간
    bgr = np.zeros((mag.shape[0], mag.shape[1], 3), dtype=np.float32)
    
    white = np.array([255, 255, 255], dtype=np.float32)  # BGR
    red   = np.array([  0,   0, 255], dtype=np.float32)  # BGR
    
    for c in range(3):
        bgr[..., c] = (1 - norm) * white[c] + norm * red[c]
    
    return bgr.astype(np.uint8)

def generate_color_legend_white2red(height, vmin, vmax, legend_width=50, font_scale=0.5):
    """
    세로(height) 크기에 맞춰, 0 ~ vmax 구간에 대응하는
    흰색→빨강 그라디언트 범례 이미지를 생성합니다. (BGR)
    """
    legend = np.zeros((height, legend_width, 3), dtype=np.uint8)
    
    for row in range(height):
        # row=0 → vmax, row=height-1 → vmin 에 대응
        norm = 1.0 - (row / (height - 1 + 1e-8))
        
        val_b = int((1 - norm)*255 + norm*  0)  # B
        val_g = int((1 - norm)*255 + norm*  0)  # G
        val_r = int((1 - norm)*255 + norm*255)  # R
        
        legend[row, :, 0] = val_b
        legend[row, :, 1] = val_g
        legend[row, :, 2] = val_r
    
    legend_out = legend.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 범례 상단 (vmax)
    cv2.putText(
        legend_out,
        f"{vmax:.1f}",
        (2, 15),
        font,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    # 범례 하단 (vmin)
    cv2.putText(
        legend_out,
        f"{vmin:.1f}",
        (2, height - 10),
        font,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return legend_out

def visualize_strain_field_accumulation(
    video_path,
    output_path="strain_accum.avi",
    flow_threshold=1.0,
    decay_factor=0.95,
    slow_factor=1.0,
    blur_ksize=15,
    legend_width=50,
    percentile_val=95
):
    """
    Optical Flow 크기를 누적(accumulation)하는 방식으로 시각화.
    
    (주요 특징)
    - 'acc_map'을 유지하며 매 프레임마다:
        acc_map = max(decay_factor * acc_map, current_mag_smooth)
      형태로 업데이트 → 큰 값은 오래 유지하되, 시간이 흐르면 서서히 감소
    - [vmin=0, vmax=동적(percentile_val%) 설정] 으로 컬러맵을 흰색→빨강으로 표현
    - 원본 영상 + 시각화 컬러맵 + 범례를 나란히 붙여서 저장
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Original FPS={orig_fps}, Size=({width}x{height})")
    
    out_fps = orig_fps / slow_factor
    out_width = width * 2 + legend_width  # (왼)원본 + (중간)컬러맵 + (오른쪽)범례
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 누적 맵(acc_map) 초기화
    acc_map = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Optical Flow 계산
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray,
            None,
            0.5, 3, 15, 3, 5, 1.2,
            0
        )
        
        vx = flow[..., 0]
        vy = flow[..., 1]
        mag = np.sqrt(vx**2 + vy**2)
        
        # 1) 공간적 스무딩 + threshold
        mag = cv2.GaussianBlur(mag, (blur_ksize, blur_ksize), 0)
        mag[mag < flow_threshold] = 0
        
        # 2) 누적(accumulation) 맵 갱신
        if acc_map is None:
            # 첫 프레임에서는 그대로 초기화
            acc_map = mag.astype(np.float32)
        else:
            # 지수적 감소 + 최대값
            # -> 이전 값이 서서히 감소하되, 새 mag가 더 크면 갱신
            acc_map = np.maximum(decay_factor * acc_map, mag)
        
        # 3) 동적 vmax: 상위 percentile_val%에 해당하는 값
        dyn_vmax = np.percentile(acc_map, percentile_val)
        dyn_vmax = max(dyn_vmax, 1e-6)  # 혹시 0이 될까봐 방지
        
        # 4) 컬러맵 생성
        mag_color = mag_to_bgr_white2red(acc_map, vmin=0.0, vmax=dyn_vmax)
        
        # 5) 범례 생성
        legend_img = generate_color_legend_white2red(
            height=height,
            vmin=0.0,
            vmax=dyn_vmax,
            legend_width=legend_width,
            font_scale=0.5
        )
        
        # 6) 결과 프레임 합치기
        combined_frame = np.full((height, out_width, 3), 255, dtype=np.uint8)
        # (왼) 원본 영상
        combined_frame[:, :width] = frame
        # (중간) 누적 맵 컬러 시각화
        combined_frame[:, width:width*2] = mag_color
        # (오른쪽) 범례
        combined_frame[:, width*2:] = legend_img
        
        out.write(combined_frame)
        
        # 다음 iteration을 위해 업데이트
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
    
    output_path = os.path.join(output_dir, f"{video_name}_accum_result.avi")
    
    visualize_strain_field_accumulation(
        video_path=video_file,
        output_path=output_path,
        flow_threshold=1.0,
        decay_factor=0.9,    # 큰 값은 오래 유지
        blur_ksize=15,
        percentile_val=95
    )

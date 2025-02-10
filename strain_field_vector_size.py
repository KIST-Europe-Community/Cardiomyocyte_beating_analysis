import cv2
import numpy as np
import sys
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
    h, w = mag.shape
    bgr = np.zeros((h, w, 3), dtype=np.float32)
    
    white = np.array([255, 255, 255], dtype=np.float32)  # BGR
    red   = np.array([  0,   0, 255], dtype=np.float32)  # BGR
    
    # 채널별로 선형 보간
    for c in range(3):
        bgr[..., c] = (1 - norm) * white[c] + norm * red[c]
    
    return bgr.astype(np.uint8)


def generate_color_legend_white2red(height, vmin, vmax, legend_width=50, font_scale=0.5):
    """
    세로(height) 크기에 맞춰, 0 ~ vmax 구간에 대응하는
    흰색→빨강 그라디언트 범례 이미지를 생성합니다. (BGR)
    
    legend_width: 범례 바의 폭
    font_scale  : 범례에 표시할 글자 크기
    """
    # 범례 바 자체(위에서 아래로 그라디언트)
    legend = np.zeros((height, legend_width, 3), dtype=np.uint8)
    
    for row in range(height):
        # row=0 → normalized=1 (가장 위가 vmax) 
        # row=height-1 → normalized=0 (가장 아래가 vmin)
        norm = 1.0 - (row / (height - 1 + 1e-8))  # [0..1]
        
        # 흰색(255,255,255) → 빨강(0,0,255) 보간
        r = int(255 * norm)   # R 채널(흰색 → 빨강 = 255→255, 실제로는 G,B 채널 255→0 보간 필요)
        g = int(255 * (1 - norm))  # 흰색(255) → 빨강(0)
        b = int(255 * (1 - norm))  # 흰색(255) → 빨강(0)
        
        # 위 로직은 채널별로 따로 보간한 것이므로, 완전히 linear한 "white→red"를 쓰려면:
        #   color = white*(1-norm) + red*norm
        #   즉 color = [ (1-norm)*255 + norm*0, (1-norm)*255 + norm*0, (1-norm)*255 + norm*255 ]
        #   아래는 그 방식을 직접 적용
        val_b = int((1 - norm)*255 + norm*  0)  # B
        val_g = int((1 - norm)*255 + norm*  0)  # G
        val_r = int((1 - norm)*255 + norm*255)  # R
        
        legend[row, :, 0] = val_b
        legend[row, :, 1] = val_g
        legend[row, :, 2] = val_r
    
    # 글자(최솟값/최댓값) 표시하기
    legend_out = legend.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # 최댓값(vmax)을 범례의 맨 위에
    cv2.putText(
        legend_out,
        f"{vmax:.1f}",
        (2, 15),  # x=2, y=15(px) 정도 (위쪽)
        font,
        font_scale,
        (0, 0, 0),  # 검정 글씨
        thickness=1,
        lineType=cv2.LINE_AA
    )
    # 최솟값(vmin)을 범례의 맨 아래에
    cv2.putText(
        legend_out,
        f"{vmin:.1f}",
        (2, height - 10),  # 아래쪽
        font,
        font_scale,
        (0, 0, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    return legend_out


def visualize_strain_field_by_magnitude(
    video_path,
    output_path="strain_output.avi",
    flow_threshold=0.5,
    slow_factor=2.0,
    vmin=0.0,
    vmax=5.0,
    smoothing_factor=0.3,
    blur_ksize=7,
    legend_width=50
):
    """
    Optical Flow를 이용해 (vx, vy) 계산 후, 벡터 크기(mag = sqrt(vx^2 + vy^2))를
    흰색→빨강 그라디언트로 시각화(=배경 흰색, 값 클수록 빨강)하고,
    오른쪽에 범례(legend)를 표시하여 저장하는 함수.

    1) Farnebäck Optical Flow로 (vx, vy) 계산
    2) mag = sqrt(vx^2 + vy^2)
       - Gaussian Blur로 공간적 스무딩 (blur_ksize)
       - 너무 작은 mag(< flow_threshold)은 0으로 무시
    3) mag_smooth에 지수적/시간적 스무딩 적용
       mag_smooth = smoothing_factor * mag_smooth + (1 - smoothing_factor) * mag
    4) 컬러맵 변환(mag_to_bgr_white2red) : 0=흰색, vmax=빨강
    5) 결과 영상에 범례 이미지를 오른쪽에 붙여서 저장
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
    
    # 출력 프레임 폭(원본 + 범례 폭)
    out_width = width + legend_width
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (out_width, height))
    
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 시간적 스무딩을 위한 mag_smooth (float32) 준비
    mag_smooth = None
    
    # 범례 이미지는 고정이므로, 한 번만 생성
    legend_img = generate_color_legend_white2red(
        height=height, vmin=vmin, vmax=vmax,
        legend_width=legend_width, font_scale=0.5
    )
    
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
        
        # 벡터 크기
        mag = np.sqrt(vx**2 + vy**2)
        
        # 공간적 스무딩(Blur) 적용
        mag = cv2.GaussianBlur(mag, (blur_ksize, blur_ksize), 0)
        
        # 너무 작은 흐름은 무시
        mag[mag < flow_threshold] = 0
        
        # 시간적 스무딩 (지수적 필터)
        if mag_smooth is None:
            mag_smooth = mag.copy().astype(np.float32)
        else:
            mag_smooth = (
                smoothing_factor * mag_smooth +
                (1 - smoothing_factor) * mag
            )
        
        # 컬러맵 변환 (흰색→빨강)
        mag_color = mag_to_bgr_white2red(mag_smooth, vmin=vmin, vmax=vmax)
        
        # 범례 영상 우측에 붙이기
        combined_frame = np.zeros((height, out_width, 3), dtype=np.uint8)
        # 배경을 흰색으로 채움
        combined_frame[:] = (255, 255, 255)
        
        # 왼쪽 영역(원본 width 폭)에 mag_color 넣기
        combined_frame[:, :width] = mag_color
        # 오른쪽 영역에 legend 붙이기
        combined_frame[:, width:] = legend_img
        
        # 결과 영상 저장
        out.write(combined_frame)
        
        prev_gray = current_gray
    
    cap.release()
    out.release()
    print(f"[DONE] '{output_path}' 에 (흰색→빨강) + 범례가 포함된 시각화 영상이 저장되었습니다.")


if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     print("사용법: python strain_field_mag_white_bg.py <video_path>")
    #     sys.exit(1)
    
    # video_file = sys.argv[1]  # 명령줄에서 비디오 파일 경로 받기
    video_file = "/data/cardio_data/Movie-1.mp4"
    
    # 입력 파일명에서 확장자를 제외한 이름 추출
    video_name = os.path.splitext(os.path.basename(video_file))[0]
    
    # 결과 저장 경로 설정
    output_dir = "/home/beating_contest/result_strain_video_vector_size"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{video_name}_strain_vector_size.avi")
    
    visualize_strain_field_by_magnitude(
        video_path=video_file,
        output_path=output_path,
        flow_threshold=0.5,
        blur_ksize=7,
        smoothing_factor=0.3,
        vmin=0.0,
        vmax=2.0,
        legend_width=50
    )

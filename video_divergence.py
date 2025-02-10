import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def visualize_contraction_expansion(video_path, output_path="output.avi"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 영상 저장용 FourCC와 VideoWriter 설정
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 예: 'XVID', 'mp4v', etc.
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 첫 프레임
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    while True:
        ret, frame = cap.read()
        if not ret:
            # 더 이상 프레임이 없으면 종료
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gunnar Farnebäck Optical Flow 계산
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            current_gray,
            None,
            0.5,
            3,
            15,
            3,
            5,
            1.1,
            0
        )
        
        # flow[..., 0] -> vx, flow[..., 1] -> vy
        vx = flow[..., 0]
        vy = flow[..., 1]
        
        # -----------------------------
        #   발산(divergence) 계산
        # -----------------------------
        # np.gradient는 (dy, dx) 순서로 반환되므로 주의!
        dvx_dy, dvx_dx = np.gradient(vx)
        dvy_dy, dvy_dx = np.gradient(vy)
        
        # div = ∂vx/∂x + ∂vy/∂y
        #      = dvx_dx + dvy_dy
        div = dvx_dx + dvy_dy
        
        # 노이즈 감소를 위한 블 강한 블러링
        div = cv2.GaussianBlur(div, (9, 9), 0)
        
        # -----------------------------
        #   수축/이완 마스크 생성
        # -----------------------------
        # 수축: div < 0
        # 이완: div > 0
        contraction_mask = np.zeros_like(frame, dtype=np.uint8)  # 빨간색
        expansion_mask   = np.zeros_like(frame, dtype=np.uint8)  # 파란색
        
        # 이진화 적용
        _, contraction_binary = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY_INV)
        _, expansion_binary = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY)
        
        # 모폴로지 연산을 통한 노이즈 제거
        kernel = np.ones((5, 5), np.uint8)
        contraction_binary = cv2.morphologyEx(contraction_binary, cv2.MORPH_OPEN, kernel)
        expansion_binary = cv2.morphologyEx(expansion_binary, cv2.MORPH_OPEN, kernel)
        
        contraction_mask[contraction_binary > 0] = (0, 0, 255)
        expansion_mask[expansion_binary > 0] = (255, 0, 0)
        
        # 두 마스크를 합침
        mask = contraction_mask + expansion_mask
        
        # -----------------------------
        #   원본 영상 + 마스크 합성
        # -----------------------------
        alpha = 0.3  # 투명도 조절
        overlaid = cv2.addWeighted(frame, 1 - alpha, mask, alpha, 0)

        # 결과 영상 저장
        out.write(overlaid)
        
        # 다음 루프를 위해 현재 프레임을 prev_gray로 갱신
        prev_gray = current_gray.copy()
    
    cap.release()
    out.release()
    print(f"결과 영상이 {output_path}에 저장되었습니다.")

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python measure_displacement.py <video_path>")
#         sys.exit(1)
    
#     video_file = sys.argv[1]
    
#     # 예: "output.avi"로 저장하고 싶다면
#     visualize_contraction_expansion(video_file, output_path="output.avi")

if __name__ == "__main__":
    # 직접 파일 경로 지정
    video_file = "/data/cardio_data/Movie-1.mp4"  # 여기에 비디오 파일 경로를 입력하세요
    
    # 예: "output.avi"로 저장하고 싶다면
    visualize_contraction_expansion(video_file, output_path="movie1_div_mask_new.avi")

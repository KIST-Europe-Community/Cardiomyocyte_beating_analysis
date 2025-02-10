import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def visualize_contraction_expansion(video_path, output_path="output.avi", flow_threshold=1.0, decay_factor=0.9):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # 마스크 누적용 초기화 (BGR 채널)
    # 값이 0~255 사이를 넘어가지 않도록 uint8 로 설정
    mask_acc = np.zeros((height, width, 3), dtype=np.uint8)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
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
        
        vx = flow[..., 0]
        vy = flow[..., 1]
        mag = np.sqrt(vx**2 + vy**2)

        dvx_dy, dvx_dx = np.gradient(vx)
        dvy_dy, dvy_dx = np.gradient(vy)
        div = dvx_dx + dvy_dy

        div = cv2.GaussianBlur(div, (9, 9), 0)
        
        contraction_mask = np.zeros_like(frame, dtype=np.uint8)  # 빨간색
        expansion_mask   = np.zeros_like(frame, dtype=np.uint8)  # 파란색

        _, contraction_binary = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY_INV)
        _, expansion_binary   = cv2.threshold(div, 0, 255, cv2.THRESH_BINARY)

        contraction_binary = np.where(
            (contraction_binary > 0) & (mag > flow_threshold),
            255,
            0
        ).astype(np.uint8)

        expansion_binary = np.where(
            (expansion_binary > 0) & (mag > flow_threshold),
            255,
            0
        ).astype(np.uint8)
        
        kernel = np.ones((5, 5), np.uint8)
        contraction_binary = cv2.morphologyEx(contraction_binary, cv2.MORPH_OPEN, kernel)
        expansion_binary   = cv2.morphologyEx(expansion_binary, cv2.MORPH_OPEN, kernel)
        
        contraction_mask[contraction_binary > 0] = (0, 0, 255)
        expansion_mask[expansion_binary > 0]     = (255, 0, 0)
        
        mask = contraction_mask + expansion_mask

        # --- 누적 마스크에 현재 프레임의 마스크를 추가 ---
        # 먼저 decay_factor만큼 곱해 오래된 흔적을 서서히 제거 (감쇄)
        mask_acc = (mask_acc * decay_factor).astype(np.uint8)
        
        # 새로 발생한 마스킹을 mask_acc에 더해준다.
        # np.clip은 255를 넘지 않게 해줌
        mask_acc = np.clip(mask_acc + mask, 0, 255).astype(np.uint8)

        # 최종 오버레이는 "현재 누적된 마스크"를 사용
        alpha = 0.3
        overlaid = cv2.addWeighted(frame, 1 - alpha, mask_acc, alpha, 0)

        out.write(overlaid)
        prev_gray = current_gray.copy()
    
    cap.release()
    out.release()
    print(f"결과 영상이 {output_path} 에 저장되었습니다.")


if __name__ == "__main__":
    # 직접 파일 경로 지정
    video_file = "/data/cardio_data/Movie-1.mp4"  # 여기에 비디오 파일 경로를 입력하세요

    # flow_threshold 인자를 통해 원하는 임계값을 설정할 수 있습니다.
    visualize_contraction_expansion(video_file, output_path="movie1_div_mask_accumulation.avi", flow_threshold=1.0)

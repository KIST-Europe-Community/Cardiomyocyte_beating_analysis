import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Matplotlib 백엔드를 'Agg'로 설정 (GUI 방지)
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

def measure_displacement(video_path):
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return
    
    # FPS (초당 프레임 수) 획득
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 첫 프레임 읽어오기
    ret, prev_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return
    
    # 흑백 변환
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    frame_count = 0
    time_list = []
    displacement_list = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            # 더 이상 프레임이 없으면 종료
            break
        
        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Gunnar Farnebäck Optical Flow 계산
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,         # 이전 프레임 (흑백)
            current_gray,      # 현재 프레임 (흑백)
            None,              # 결과를 저장할 공간 (None이면 자동 생성)
            0.5,               # 피라미드 스케일
            3,                 # 피라미드 레벨 수
            15,                # 윈도우 크기(커널 크기)
            3,                 # 반복 횟수
            5,                 # 확장 상수(확실성)
            1.1,               # 가우시안 분포 시그마
            0                  # 기타 플래그
        )
        
        # flow는 (height, width, 2) 크기의 numpy 배열
        # flow[..., 0]은 x방향 흐름, flow[..., 1]은 y방향 흐름
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # 전체 픽셀에 대한 optical flow 크기(magnitude)의 평균값
        avg_displacement = np.mean(mag)
        
        # 시간과 변위 저장
        current_time = frame_count / fps
        time_list.append(current_time)
        displacement_list.append(avg_displacement)
        
        prev_gray = current_gray.copy()
        frame_count += 1
    
    cap.release()
    
    # 결과 저장 디렉토리 설정
    result_dir = "/home/calcium_imaging/result"
    os.makedirs(result_dir, exist_ok=True)
    
    # 비디오 파일 이름 추출
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # CSV 파일로 데이터 저장
    df = pd.DataFrame({
        'Time(s)': time_list,
        'Displacement': displacement_list
    })
    df.to_csv(os.path.join(result_dir, f"{video_name}_displacement.csv"), index=False)
    
    # Plot 저장 (display 없이)
    plt.figure(figsize=(10, 5))
    plt.plot(time_list, displacement_list, marker='o')
    plt.title('Displacement over Time')
    plt.text(
        np.mean(time_list), 
        max(displacement_list) * 0.9, 
        f"File name: {video_name}",
        ha='center'
    )
    plt.xlabel('Time (seconds)')
    plt.ylabel('Average Displacement')
    plt.grid(True)
    plt.savefig(os.path.join(result_dir, f"{video_name}_plot.png"))
    plt.close()  # Figure 닫기

# if __name__ == "__main__":
#     if len(sys.argv) < 2:
#         print("사용법: python measure_displacement.py <video_path>")
#         sys.exit(1)
    
#     video_file = sys.argv[1]  # 명령줄에서 비디오 파일 경로 받기
#     measure_displacement(video_file)

if __name__ == "__main__":
    measure_displacement("/home/calcium_imaging/500nM_isoprenaline_1.avi")

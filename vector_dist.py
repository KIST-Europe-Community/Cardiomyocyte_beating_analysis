import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def compute_flow_histogram(video_path, flow_threshold=1.0, blur_ksize=15, bins=50, frame_skip=1):
    """
    Optical Flow의 magnitude 히스토그램을 실시간으로 계산합니다.
    모든 프레임의 magnitude 값을 분석하여 범위를 결정합니다.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_path}")
        return None

    # 최대/최소값 추적을 위한 변수 초기화
    max_mag = float('-inf')
    min_mag = float('inf')
    frame_count = 0  # frame_count 초기화 추가
    
    # 첫 프레임 읽기
    ret, first_frame = cap.read()
    if not ret:
        print("첫 프레임을 읽을 수 없습니다.")
        return None
    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    
    # 첫 번째 패스: 최대/최소값 찾기
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mag = cv2.GaussianBlur(mag, (blur_ksize, blur_ksize), 0)
        mag[mag < flow_threshold] = 0

        current_max = np.max(mag)
        current_min = np.min(mag[mag > 0]) if np.any(mag > 0) else 0

        max_mag = max(max_mag, current_max)
        min_mag = min(min_mag, current_min) if current_min > 0 else min_mag

        prev_gray = current_gray


    # 비디오 포인터를 처음으로 되돌림
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    prev_gray = cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2GRAY)
    
    # 히스토그램 범위 설정
    bin_edges = np.linspace(min_mag, max_mag, bins + 1)
    histogram = np.zeros(bins, dtype=np.float32)
    frame_count = 0

    # 두 번째 패스: 히스토그램 계산
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if frame_count % frame_skip != 0:
            continue

        current_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, current_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )

        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        mag = cv2.GaussianBlur(mag, (blur_ksize, blur_ksize), 0)
        mag[mag < flow_threshold] = 0

        hist, _ = np.histogram(mag, bins=bin_edges)
        histogram += hist

        prev_gray = current_gray

    cap.release()
    return histogram, bin_edges[:-1], min_mag, max_mag


def plot_histogram(histogram, bin_edges, output_fig="flow_distribution.png"):
    """
    히스토그램 데이터를 이용해 플롯을 생성하고 저장합니다.
    """
    plt.figure(figsize=(6, 4), dpi=100)
    plt.bar(bin_edges, histogram, width=(bin_edges[1] - bin_edges[0]), color='r', alpha=0.7, edgecolor='black')
    plt.title("Distribution of Flow Magnitudes")
    plt.xlabel("Magnitude")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(output_fig)
    print(f"[DONE] 히스토그램 이미지가 저장되었습니다: {output_fig}")
    plt.close()


def main(video_path):
    histogram, bin_edges, min_mag, max_mag = compute_flow_histogram(
        video_path, 
        flow_threshold=1.0, 
        blur_ksize=15, 
        bins=100, 
        frame_skip=1
    )
    if histogram is None:
        print("[WARN] 유효한 Optical Flow 데이터가 없습니다.")
        return

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    print(f"[INFO] Flow magnitude 범위: {min_mag:.2f} ~ {max_mag:.2f}")

    # 히스토그램 저장
    plot_histogram(histogram, bin_edges, output_fig=f"{video_name}_flow_distribution.png")


if __name__ == "__main__":
    video_file = "/data/cardio_data/Movie-1.mp4"
    main(video_file)

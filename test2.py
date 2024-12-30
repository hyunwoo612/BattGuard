import sys
import os
import cv2
import numpy as np
from PyQt5 import uic
from PyQt5.QtWidgets import *
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QPixmap
from ultralytics import YOLO
form_class = uic.loadUiType("/home/hyunwoo/바탕화면/BattGuard.ui")[0]


class ImageProcessorThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(int, int)
    image_signal = pyqtSignal(str)

    def __init__(self, model, folder_path, threshold):
        super().__init__()
        self.model = model
        self.folder_path = folder_path
        self.threshold = threshold
        self.good_count = 0
        self.defect_count = 0

    def run(self):
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        total_files = len(image_files)

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(self.folder_path, image_file)
            self.image_signal.emit(image_path)  # 방출

            try:
                results = self.model(image_path)  # 모델 예측

                # 결과 확인
                if results and len(results) > 0:
                    predictions = results[0]

                    # 클래스 확률 가져오기
                    probs = predictions.probs  # 각 클래스의 확률
                    if probs is not None:
                        good_prob = probs.data[1].item()  # '양품' 확률 (클래스 ID 1)

                        if good_prob >= self.threshold:
                            self.good_count += 1
                        else:
                            self.defect_count += 1
            except Exception as e:
                print(f"{image_file} 처리 중 오류 발생: {e}")

            self.progress.emit(int((idx + 1) / total_files * 100))

        # 최종 결과 방출
        self.result.emit(self.good_count, self.defect_count)


class PrecisionProcessorThread(QThread):
    progress = pyqtSignal(int)
    result = pyqtSignal(int, int)  # 처리된 결과(양품, 불량품 개수)를 방출
    image_signal = pyqtSignal(str, dict)  # 이미지 경로와 클래스 개수 딕셔너리 방출

    def __init__(self, model, folder_path, output_folder):
        super().__init__()
        self.model = model
        self.folder_path = folder_path
        self.output_folder = output_folder
        self.good_count = 0  # 양품 개수
        self.defect_count = 0  # 불량품 개수

    def run(self):
        image_files = [f for f in os.listdir(self.folder_path) if f.endswith((".jpg", ".png", ".jpeg"))]
        total_files = len(image_files)

        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

        for idx, image_file in enumerate(image_files):
            image_path = os.path.join(self.folder_path, image_file)

            try:
                # 모델 예측
                results = self.model(image_path)

                if results and len(results) > 0:
                    predictions = results[0]

                    if hasattr(predictions, 'masks') and predictions.masks is not None:
                        # 클래스별 개수 계산
                        class_counts = {}
                        for cls_id in predictions.boxes.cls.cpu().numpy():
                            class_name = self.model.names[int(cls_id)]
                            class_counts[class_name] = class_counts.get(class_name, 0) + 1

                        # 마스크 데이터 가져오기
                        masks = predictions.masks.data.cpu().numpy()
                        image = cv2.imread(image_path)
                        original_height, original_width = image.shape[:2]

                        # 오버레이 생성
                        overlay = image.copy()
                        for mask in masks:
                            mask_resized = cv2.resize((mask > 0.5).astype(np.uint8), (original_width, original_height))
                            color = (0, 255, 0)  # 초록색
                            contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(overlay, contours, -1, color, thickness=cv2.FILLED)

                        blended = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)
                        output_path = os.path.join(self.output_folder, f"processed_{image_file}")
                        cv2.imwrite(output_path, blended)

                        # 이미지 경로와 클래스 개수 방출
                        self.image_signal.emit(output_path, class_counts)

                        # 양품과 불량품 개수 계산 (예: '양품' 클래스가 있을 경우 양품 개수 증가)
                        if '양품' in class_counts:
                            self.good_count += class_counts['양품']
                        else:
                            self.defect_count += sum(class_counts.values())

            except Exception as e:
                print(f"{image_file} 처리 중 오류 발생: {e}")

            # 진행률 업데이트
            self.progress.emit(int((idx + 1) / total_files * 100))

        # 최종 결과 방출
        self.result.emit(self.good_count, self.defect_count)


class WindowClass(QMainWindow, form_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("BattGuard v1")

        # UI 연결
        self.actionInput_folder.triggered.connect(self.select_folder)
        self.fastsearch.clicked.connect(self.on_start_clicked)
        self.slowsearch.clicked.connect(self.on_precision_clicked)  # 정밀검사 버튼 연결
        self.PrevButton.clicked.connect(self.show_previous_image)  # PrevButton 연결
        self.NextButton.clicked.connect(self.show_next_image)      # NextButton 연결

        self.selected_folder = None
        self.output_folder = "/home/hyunwoo/processed_images"
        self.processed_images = []  # 처리된 이미지 리스트
        self.current_index = 0  # 현재 이미지 인덱스
        self.threshold = 0.55

        # 모델 로드
        try:
            self.fast_model = YOLO("/home/hyunwoo/model/battguardtest/class1.pt")  # 일반 검사 모델
            self.precision_model = YOLO("/home/hyunwoo/model/battguardtest/seg.pt")  # 정밀검사 모델
        except Exception as e:
            print(f"모델 로드 중 오류 발생: {e}")
            self.fast_model = None
            self.precision_model = None

        self.progressBar = QProgressBar(self)
        self.statusBar().addWidget(self.progressBar)
        self.progressBar.setValue(0)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(100, 100, 500, 500)

        self.result_label = QLabel(self)
        self.result_label.setGeometry(870, 100, 300, 200)
        self.result_label.setStyleSheet("font-size: 16px; color: black; text-align: center;")
        self.result_label.setText("결과가 여기에 표시됩니다.")

        self.is_fast_processing = False  # 빠른 검사 중인지 여부를 추적하는 변수

    def select_folder(self):
        folder_path = QFileDialog.getExistingDirectory(self, "폴더 선택")
        if folder_path:
            self.selected_folder = folder_path
            QMessageBox.information(self, "폴더 선택 완료", f"선택된 폴더: {self.selected_folder}")

    def on_start_clicked(self):
        if not self.selected_folder:
            QMessageBox.warning(self, "경고", "먼저 폴더를 선택하세요.")
            return
        if not self.fast_model:
            QMessageBox.warning(self, "경고", "모델이 초기화되지 않았습니다.")
            return

        # 빠른 검사 중이면 이전/다음 버튼 비활성화
        self.is_fast_processing = True
        self.PrevButton.setEnabled(False)
        self.NextButton.setEnabled(False)

        self.start_processing(self.selected_folder, self.fast_model, "fast")

    def on_precision_clicked(self):
        if not self.selected_folder:
            QMessageBox.warning(self, "경고", "먼저 폴더를 선택하세요.")
            return
        if not self.precision_model:
            QMessageBox.warning(self, "경고", "정밀검사 모델이 초기화되지 않았습니다.")
            return

        # 정밀검사 시 이전/다음 버튼 활성화
        self.is_fast_processing = False
        self.PrevButton.setEnabled(True)
        self.NextButton.setEnabled(True)

        self.start_processing(self.selected_folder, self.precision_model, "precision")

    def start_processing(self, folder_path, model, mode):
        if mode == "fast":
            self.thread = ImageProcessorThread(model, folder_path, self.threshold)
            self.thread.result.connect(self.show_fast_results)
        elif mode == "precision":
            self.thread = PrecisionProcessorThread(model, folder_path, self.output_folder)
            self.thread.image_signal.connect(self.add_processed_image)  # 실시간 이미지 업데이트
            self.thread.result.connect(self.show_precision_results)
        self.thread.progress.connect(self.update_progress)
        self.thread.start()

    def add_processed_image(self, image_path, class_counts):
        # 처리된 이미지를 리스트에 추가
        self.processed_images.append((image_path, class_counts))

        if len(self.processed_images) == 1:  # 첫 번째 이미지 표시
            self.display_image_and_result(0)

    def display_image_and_result(self, index):
        if not self.processed_images:
            return

        self.current_index = max(0, min(index, len(self.processed_images) - 1))
        image_path, class_counts = self.processed_images[self.current_index]

        # 이미지 업데이트
        pixmap = QPixmap(image_path)
        pixmap = pixmap.scaled(self.image_label.width(), self.image_label.height(), aspectRatioMode=True)
        self.image_label.setPixmap(pixmap)

        # 클래스별 결과 업데이트
        class_counts_text = ", ".join([f"{cls}: {count}" for cls, count in class_counts.items()])
        self.result_label.setText(f"갯수: {class_counts_text}")

    def show_previous_image(self):
        if self.is_fast_processing:
            return  # 빠른 검사 중에는 이전 이미지 버튼이 작동하지 않음
        self.display_image_and_result(self.current_index - 1)

    def show_next_image(self):
        if self.is_fast_processing:
            return  # 빠른 검사 중에는 다음 이미지 버튼이 작동하지 않음
        self.display_image_and_result(self.current_index + 1)

    def show_fast_results(self, good_count, defect_count):
        # 빠른 검사 후 이전/다음 버튼 활성화
        self.is_fast_processing = False
        self.PrevButton.setEnabled(True)
        self.NextButton.setEnabled(True)

        self.result_label.setText(f"양품: {good_count}\n불량: {defect_count}")

    def show_precision_results(self, good_count, defect_count):
        self.result_label.setText(f"정밀검사 결과:\n양품: {good_count}\n불량: {defect_count}")

    def update_progress(self, value):
        self.progressBar.setValue(value)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WindowClass()
    window.show()
    sys.exit(app.exec_())

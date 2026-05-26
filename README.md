# Drone vs Airplane vs Bird Classification (SVM vs ResNet-18)

본 프로젝트는 비행체(Drone, Airplane)와 조류(Bird)를 분류하기 위해 머신러닝 기법인 HOG+SVM과 딥러닝 기반의 ResNet-18 모델의 성능을 비교 분석하는 코드입니다.

## 프로젝트 개요
* **목표:** 하늘에 떠 있는 객체(드론, 비행기, 새)를 정확하게 탐지하고 분류.
* **데이터셋:** Skyborne Objects, Maryam Drone Detection, Seraphim Drone Dataset을 병합하여 사용.
* **클래스:** `Drone`, `Airplane`, `Bird` (클래스당 900장 평형화, 총 2,700장)
* **결과 요약:**
  * **HOG + SVM (Baseline):** 정확도 약 52.41%
  * **ResNet-18 (제안 모델):** 정확도 약 94.44%

##  기술 스택
* **언어:** Python 3
* **주요 라이브러리:** PyTorch, Torchvision, Scikit-learn, OpenCV, Scikit-image
* **환경:** Google Colab (T4 GPU)

##  프로젝트 구조 및 파이프라인
1. **데이터 수집:** Kaggle API 및 Hugging Face Hub를 통한 자동 다운로드
2. **전처리:** 데이터 통합, 결측치 제거 및 클래스 평형화 (Balancing)
3. **데이터 분할:** Train/Val(80%) 및 Test(20%) 계층적 분할 (Stratified Split)
4. **베이스라인 모델 (SVM):**
   * Grayscale 변환 및 사이즈 정규화 (128x128)
   * HOG(Histogram of Oriented Gradients) 특징 벡터 추출
   * GridSearchCV를 활용한 5-Fold 교차 검증 및 하이퍼파라미터 튜닝
5. **제안 모델 (ResNet-18):**
   * ImageNet 사전 학습 가중치 사용 (Transfer Learning)
   * 데이터 증강 (Random Horizontal Flip) 적용
   * Adam 옵티마이저 및 조기 종료(Early Stopping) 적용

## 실행 및 재현 방법 (How to Reproduce)

### 1. 저장소 클론 및 환경 설정
```bash
git clone [https://github.com/사용자계정/저장소이름.git](https://github.com/사용자계정/저장소이름.git)
cd 저장소이름
pip install -r requirements.txt
```

### 2. kaggle id 및 kaggle API key 설정
```bash
kaggle_credentials = {
    "username": "kaggle 닉네임",
    "key": "kaggle API 키"
}
```
* 위 코드 내 닉네임 및 API 키 변경 필요


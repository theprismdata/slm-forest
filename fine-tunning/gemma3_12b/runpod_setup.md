# RunPod Gemma3 12B 파인튜닝 가이드

## 🚀 RunPod 설정

### 1. 권장 GPU 스펙
- **A100 40GB**: 최적 ($2.89/hr)
- **A100 80GB**: 최고 성능 ($3.18/hr)
- **RTX 4090 24GB**: 최소 사양 ($0.79/hr)

### 2. RunPod 인스턴스 생성
```bash
# 템플릿 선택: PyTorch 2.0+ with CUDA 11.8+
# 컨테이너 디스크: 최소 100GB
# 볼륨 스토리지: 50GB (모델 저장용)
```

### 3. 초기 설정
```bash
# 1. 작업 디렉토리 생성
mkdir -p /workspace
cd /workspace

# 2. 코드 업로드 (GitHub 또는 직접 업로드)
git clone [your-repo-url]
cd fine-tunning/gemma3_12b

# 3. 데이터셋 업로드
mkdir -p /workspace/fine-tunning-ds
# 데이터셋 파일을 업로드하세요
```

## 💡 비용 절약 전략

### 1. 스팟 인스턴스 사용
- 비용 50-70% 절약
- 중단 위험 있음 (체크포인트 자주 저장)

### 2. 훈련 시간 최적화
- `max_steps: 1000` 설정으로 제한
- `save_steps: 250` 자주 저장
- `max_train_samples: 5000` 데이터 제한

### 3. 모델 크기 선택
```yaml
# 메모리 부족시 더 작은 모델 사용
model_name: "google/gemma-2-9b-it"  # 9B 모델
# 또는
model_name: "google/gemma-2-2b-it"  # 2B 모델
```

## 📊 예상 비용 (A100 40GB 기준)

| 설정 | 시간 | 비용 |
|------|------|------|
| 1000 steps | ~2시간 | ~$6 |
| 3 epochs | ~4시간 | ~$12 |
| Full dataset | ~8시간 | ~$24 |

## 🔧 RunPod 실행 명령어

### 1. 환경 설정
```bash
# 의존성 설치
pip install -r ../requirements-finetunning.txt

# 환경 변수 설정
export TRANSFORMERS_CACHE=/workspace/cache
export HF_DATASETS_CACHE=/workspace/cache
export WANDB_DISABLED=true
```

### 2. 훈련 실행
```bash
# 백그라운드 실행 (연결 끊어져도 계속 실행)
nohup python finetuning-gemma3.py > training.log 2>&1 &

# 실시간 로그 확인
tail -f training.log
```

### 3. 모델 다운로드
```bash
# 훈련 완료 후 모델 압축
tar -czf gemma3-finetuned.tar.gz gemma3-12b-tax-law-finetuned/

# 로컬로 다운로드 (RunPod 웹 인터페이스 사용)
```

## ⚠️ 주의사항

1. **정기적인 체크포인트 확인**
   - 250 스텝마다 저장되는지 확인
   - 스팟 인스턴스 중단에 대비

2. **메모리 모니터링**
   ```bash
   # GPU 메모리 사용량 확인
   nvidia-smi
   
   # 시스템 리소스 확인
   htop
   ```

3. **비용 관리**
   - 훈련 완료 후 즉시 인스턴스 종료
   - 불필요한 파일 삭제로 스토리지 비용 절약

## 🎯 성능 최적화

### 1. 배치 크기 조정
```yaml
# GPU 메모리에 따라 조정
per_device_train_batch_size: 1  # A100 40GB
gradient_accumulation_steps: 32  # 효과적인 배치 크기 32
```

### 2. 학습률 스케줄링
```yaml
learning_rate: 5e-6  # 큰 모델에 맞게 낮은 학습률
warmup_steps: 100
```

### 3. LoRA 설정
```yaml
r: 64  # 더 큰 rank로 성능 향상
lora_alpha: 128
lora_dropout: 0.05
```

이 설정으로 RunPod에서 효율적으로 Gemma3 파인튜닝을 진행할 수 있습니다! 
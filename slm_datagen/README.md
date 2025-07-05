
# Phi-2 Tax Law Q&A Fine-tuning Project
이 프로젝트는 Microsoft Phi-2 모델을 세법 Q&A 데이터셋으로 파인튜닝하여 실용적인 세법 질의응답 시스템을 구축하는 프로젝트입니다.


## 🎯 프로젝트 개요

- **모델**: Microsoft Phi-2 (2.7B 파라미터)
- **데이터셋**: 실용적인 세법 Q&A 데이터 (학생용/교사용 답변 포함)
- **훈련 환경**: RunPod (GPU 클라우드)
- **추론 환경**: M4 MacBook Pro (Metal Performance Shaders)

## 📁 프로젝트 구조

```
tax-law-gen/
├── fine-tunning-ds/
│   └── distillation_legal_qa_dataset.json    # 파인튜닝용 데이터셋
├── train_config.yaml                         # 훈련 설정 파일
├── train.py                                  # 훈련 스크립트
├── inference.py                              # 추론 스크립트
├── runpod_train.sh                          # RunPod 훈련 실행 스크립트
├── setup_m4_mac.sh                          # M4 Mac 설정 스크립트
├── requirements-train.txt                    # 훈련용 의존성
├── requirements-inference.txt                # 추론용 의존성
└── README.md                                 # 프로젝트 문서
```

## 🚀 빠른 시작

### 1. RunPod에서 훈련하기

#### RunPod 설정
1. RunPod에서 GPU 인스턴스 생성 (RTX 4090, A100 등 권장)
2. 프로젝트 파일들을 RunPod에 업로드
3. 터미널에서 다음 명령어 실행:

```bash
# 실행 권한 부여
chmod +x runpod_train.sh

# 훈련 시작
./runpod_train.sh
```

#### 훈련 설정 조정
`train_config.yaml` 파일에서 다음 설정을 조정할 수 있습니다:

```yaml
training_args:
  num_train_epochs: 3                    # 훈련 에포크
  per_device_train_batch_size: 4         # 배치 크기
  learning_rate: 2e-5                    # 학습률
  gradient_accumulation_steps: 4         # 그래디언트 누적

lora_config:
  r: 16                                  # LoRA 랭크
  lora_alpha: 32                         # LoRA 알파
  lora_dropout: 0.1                      # LoRA 드롭아웃
```

### 2. M4 MacBook Pro에서 추론하기

#### 환경 설정
```bash
# 실행 권한 부여
chmod +x setup_m4_mac.sh

# 환경 설정 및 의존성 설치
./setup_m4_mac.sh
```

#### 추론 실행

**대화형 모드:**
```bash
python inference.py --interactive
```

**단일 질문:**
```bash
python inference.py --question "부가가치세 신고는 어떻게 하나요?"
```

**배치 처리:**
```bash
# 질문 파일 생성
echo "연말정산 시 의료비 공제는 어떻게 받나요?" > questions.txt
echo "개인사업자 부가가치세 신고 절차는?" >> questions.txt

# 배치 추론 실행
python inference.py --questions_file questions.txt --output_file answers.json
```

## 📊 데이터셋 정보

현재 데이터셋은 다음과 같은 특징을 가집니다:

- **총 샘플 수**: 약 1,000개의 Q&A 쌍
- **답변 유형**: 학생용 (간단한 설명) + 교사용 (상세한 법적 근거)
- **세법 분야**: 부가가치세, 소득세, 법인세, 증여세, 부동산세, 투자세 등
- **질문 유형**: 실용적인 시나리오 기반 질문

### 데이터셋 예시

```json
{
  "question": "부모님께 20억원 증여받았는데 증여세는 얼마나 되나요?",
  "answer": "20억원 증여에 대한 증여세는 다음과 같이 계산됩니다:\n\n【증여세 계산】\n- 증여재산가액: 20억원\n- 공제: 없음 (부모님 증여는 공제 대상 아님)\n- 과세가액: 20억원\n\n【세율 적용】\n- 1억원 이하: 10%\n- 1억원 초과 5억원 이하: 20%\n- 5억원 초과 10억원 이하: 30%\n\n【세액 계산】\n- 1억원 × 10% = 1,000만원\n- 4억원 × 20% = 8,000만원\n- 총 증여세: 9,000만원",
  "type": "student",
  "source": "practical"
}
```

## 🔧 기술적 세부사항

### 모델 아키텍처
- **베이스 모델**: Microsoft Phi-2 (2.7B 파라미터)
- **파인튜닝 방법**: LoRA (Low-Rank Adaptation)
- **최적화**: FP16 훈련, 그래디언트 누적

### 훈련 설정
- **최대 시퀀스 길이**: 2,048 토큰
- **프롬프트 템플릿**: 
  ```
  ### 질문: {question}
  
  ### 답변: {answer}
  ```
- **검증 분할**: 10%
- **조기 종료**: 3 에포크 연속 개선 없을 시

### 추론 최적화
- **M4 Mac 최적화**: Metal Performance Shaders (MPS) 활용
- **배치 처리**: 다중 질문 처리 지원
- **메모리 효율성**: LoRA 어댑터만 로드

## 💰 비용 추정

### RunPod 훈련 비용
- **RTX 4090 (24GB)**: 약 $0.6/시간
- **A100 (40GB)**: 약 $1.2/시간
- **예상 훈련 시간**: 2-4시간
- **총 비용**: $1.2 - $4.8

### M4 MacBook Pro 추론
- **메모리 사용량**: 약 8-12GB
- **추론 속도**: 초당 10-20 토큰
- **비용**: 무료 (로컬 실행)

## 🛠️ 문제 해결

### 일반적인 문제들

**1. CUDA 메모리 부족**
```bash
# train_config.yaml에서 배치 크기 줄이기
per_device_train_batch_size: 2
gradient_accumulation_steps: 8
```

**2. MPS 관련 오류**
```bash
# CPU로 강제 실행
python inference.py --device cpu
```

**3. 모델 로딩 실패**
```bash
# 캐시 정리
rm -rf ~/.cache/huggingface/
```

### 로그 확인
```bash
# 훈련 로그 확인
tail -f phi2-tax-law-finetuned/logs/trainer_state.json

# 추론 로그 확인
python inference.py --interactive 2>&1 | tee inference.log
```

## 📈 성능 모니터링

### 훈련 중 모니터링
- **WandB**: 실시간 훈련 메트릭 추적
- **TensorBoard**: 로컬 로그 확인
- **콘솔 출력**: 진행 상황 및 손실값

### 추론 성능
- **응답 시간**: 평균 2-5초
- **토큰 생성 속도**: 초당 10-20 토큰
- **메모리 사용량**: 8-12GB

## 🔄 워크플로우

1. **데이터 준비**: 실용적인 세법 Q&A 데이터셋 생성
2. **RunPod 훈련**: GPU 환경에서 LoRA 파인튜닝
3. **모델 다운로드**: 훈련된 모델을 로컬로 다운로드
4. **M4 Mac 설정**: 추론 환경 구성
5. **추론 실행**: 실시간 세법 Q&A 서비스

## 📝 라이선스

이 프로젝트는 교육 및 연구 목적으로 제작되었습니다. 상업적 사용 시 관련 라이선스를 확인하시기 바랍니다.

## 🤝 기여하기

버그 리포트, 기능 요청, 풀 리퀘스트를 환영합니다!

---

**참고**: 이 프로젝트는 세법 전문가의 검토를 거치지 않았으므로, 실제 세무 상담에는 전문가의 도움을 받으시기 바랍니다.

# NER Pipeline


## 환경구성    
```cmd
conda create -n NER_Pipeline python=3.8
conda activate NER_Pipeline
pip install -r requirements.txt
```
---

## 프로젝트 docs
```
1. 데이터
2. 학습 
3. 추론 only
4. 배포  
```

- 모든 configuration은 `config.yaml` 파일에서 관리됩니다.  

### 1. 데이터  

- 사용 데이터셋: 3가지  
    
    - `NIKL`: 국립국어원 데이터셋 
    - `KMOU`: 한국해양대학교 데이터셋  
    - `KLUE`: Korean Language Understanding Evaluation 내 NER 데이터셋  
    
    - cf) `네이버x창원대` 데이터셋: 데이터의 품질이 매우 낮은 관계로 학습데이터에서 배제 
    
- 데이터 위치 및 구성     

    ```bash
    ./
    └─ train 
       └─ data
          ├─ completed
          │  ├─ train_data.parquet
          │  └─ valid_data.parquet
          ├─ preprocessed
          │  ├─ klue.pkl
          │  ├─ kmou.pkl
          │  └─ nikl.pkl
          └─ raw
              ├─ KMOU
              │  ├─ ner.dev
              │  ├─ ner.test
              │  └─ ner.train
              └─ NIKL_NE_2022_v1
                 ├─ MXNE2202211218.json
                 ├─ NXNE2202211218.json
                 └─ SXNE2202211218.json
    ```   

    - `raw` : 다운로드 받은 raw 파일 
        - KMOU : [데이터 다운 링크](https://corpus.korean.go.kr/request/requestList.do)의 `개체명 분석 말뭉치 2022`
        - NIKL : [데이터 다운 링크](https://github.com/kmounlp/NER/tree/master/2016klp) 
        - cf) KLUE NER: HuggingFace API를 사용하여 자동 다운로드     
    - `preprocessed` : 각 파일에 대한 전처리 결과 (중간 저장용)
    - `completed` : 전처리된 데이터를 통합한 최종 파일 (학습 및 검증에 사용) 

- 학습용 데이터 `./train/data/completed/train_data.parquet`  
  검증용 데이터 `./valid/data/completed/valid_data.parquet`
    
- 학습 데이터 생성 방법 
    1. `config.yaml` 의 stage 값을 1로 설정  
        ```yaml
        stage: 1        
        ```   
    2. 파일 실행  
        ```cmd
        python main.py
        ```

### 2. 학습   

1. `config.yaml`의 stage 값을 2로 변경  
    ```yaml
    stage: 2        
    ```  
    
2. 파일 실행  
    ```bash
    python main.py
    ```

- 모든 hyperparameter는 `config.yaml`의 `stage2_cfg`에서 관리합니다. 
- 학습한 모델의 weight는 `./pretrained_ner` 아래에 저장되며, 모델의 학습 양상은 wandb에 기록됩니다.   


### 3. 추론 only
   
1. `config.yaml`의 stage 값을 3으로 변경  
    ```yaml
    stage: 3        
    ```

2. 목적에 따라 추론 방식을 변경  
    ```yaml
    unit: inference # inference / batch
    ```  

    - batch: 배치 단위의 추론 (for MLE, DS)  
    - inference: 인스턴스 단위의 추론 (for BE)


### 4. 배포  

- 구성

    ![Demo figure](/figure/figure1.png)

1. `config.yaml`의 stage 값을 4로 변경  
    ```yaml
    stage: 4       
    ```
   
2. `config.yaml`내 `stage4_cfg`의 `ml_server_exist`에 `true`를 입력    
    ```yaml  
    stage4_cfg:
        ml_server_exist: true
    ```

3. Pytorch 모델 Serialize 및 `.mar` 파일 생성  
    ```bash
    bash script/model-archiver.sh
    ```

    - 생성된 serialized 모델은 `./pretrained_ner_script`에 저장됩니다.

4. 이미지 빌드 및 컨테이너 생성
    ```bash
    docker-compose up
    ```

5. 데모페이지 사용      
    - 예시    
    ![Demo figure](/figure/figure2.png)

---

위 과정을 모두 완료한 후에는, 서버의 작동 여부를 터미널에서 테스트 해볼 수 있습니다.  
아래 커맨드로부터 응답이 온다면, 서버 환경 구축에 성공한 것입니다.  
```
curl -X POST -H 'Content-Type: text/plain' http://127.0.0.1:8080/predictions/ner -d '삼성전자'
```
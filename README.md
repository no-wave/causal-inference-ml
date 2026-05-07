# 인과추론 1: Causal Inference & ML Python 쿡북
#### 상관을 넘어 처치 효과로, 식별과 추정으로 풀어내는 인과 추론 핵심 가이드이드

<img src="https://beat-by-wire.gitbook.io/beat-by-wire/~gitbook/image?url=https%3A%2F%2F3055094660-files.gitbook.io%2F%7E%2Ffiles%2Fv0%2Fb%2Fgitbook-x-prod.appspot.com%2Fo%2Fspaces%252FYzxz4QeW9UTrhrpWwKiQ%252Fuploads%252FGDN4JF0oKuHHWgBT2uzt%252Fcausal-inference1.png%3Falt%3Dmedia%26token%3D551e3680-a807-4207-aba1-24fc44963a10&width=300&dpr=3&quality=100&sign=d2d8ce34&sv=2"  width="500" height="707"/>



## 책 소개

머신러닝을 어느 정도 다뤄 본 사람들이 공통적으로 마주치는 순간이 있다. 잘 훈련된 모델이 96%의 정확도로 고객 이탈을 예측하거나, 정교하게 튜닝된 그래디언트 부스팅이 매출을 정확히 추정하는 모습을 보고도, 의사결정자가 던지는 한 마디 앞에서 말문이 막히는 순간이다. '그래서, 이 마케팅 캠페인을 늘리면 매출이 얼마나 오를까?' 모델은 이탈 가능성과 매출 규모를 잘 맞히지만, 정작 '개입했을 때 무엇이 달라지는지'를 묻는 이 질문에는 답하지 못한다.

문제는 그 다음이다. 회귀계수를 인과 효과로 해석하려는 유혹, 상관관계가 높으니 인과관계도 있을 것이라는 직관, 무작위 실험은 비현실적이라는 체념. 인과추론을 본격적으로 공부하려 하면 잠재적 결과 프레임워크, 식별 가능성, 무시가능성, 양성성, DAG와 d-분리, 도구변수의 외생성, 이중차분의 평행 추세 같은 가정들이 쏟아진다. 책마다 다른 기호와 다른 방법론이 등장하고, 통계학자의 언어와 경제학자의 언어와 컴퓨터 과학자의 언어가 미묘하게 어긋난다. 많은 실무자가 바로 그 지점에서 멈춘다. 이 책은 그 지점을 통과하기 위해 썼다.

이 책은 머신러닝과 인과추론이 결합하여 '왜'와 '얼마나'를 데이터로부터 답하는 원리와 방법을 이론과 실전 모두에서 체계적으로 다룬다. 단순히 방법론을 나열하는 것을 넘어, 각각의 접근법이 왜 등장했고, 어떤 가정 위에서 작동하며, 어떤 상황에 가장 적합한지를 함께 짚는다.

이 책은 Causal Inference 시리즈의 첫 번째 권이다. 시리즈의 1권으로서 이 책이 맡은 역할은 이론적 기반과 실용 도구의 견고한 결합이다. Neyman과 Rubin이 1920년대에 세운 잠재적 결과의 언어부터, 2024년의 인과 딥러닝 논문까지 — 한 세기에 걸친 지적 축적을 한 권에 압축해 단단한 토대로 만든다. 이 책의 자매편이자 시리즈의 두 번째 권인 『인과추론 2: LLM Causal Agent Python 쿡북』이 이 토대 위에서 자연스럽게 이어진다. 1권이 '무엇을 어떻게 추정할 것인가'를 가르쳤다면, 2권은 '그것을 어떻게 LLM 에이전트로 자동화할 것인가'를 가르친다.

지금 우리는 인과 AI(Causal AI)의 시대에 들어서 있다. 단순히 '예측이 잘 맞는가'만 묻던 시대는 저물고 있다. 의료에서 어떤 환자에게 어떤 치료가 효과적일지, 정책에서 어떤 개입이 실제로 결과를 바꾸는지, 비즈니스에서 어떤 가격 변화가 매출을 끌어올리는지 — 이 모든 질문은 상관이 아닌 인과를 묻는다. 머신러닝의 강력한 예측력을 가지고도 인과를 묻지 못하는 시스템은 점점 더 부족해 보일 것이다.

방법의 이름을 아는 것과, 그 작동 원리를 손으로 직접 이해하고 자신의 데이터에 적용해 본 것은 전혀 다른 역량이다. 이 책을 마칠 때쯤이면, 새로운 인과추론 논문을 읽고 그 가정과 식별 전략을 꿰뚫어 볼 수 있는 눈과, 직접 분석을 설계할 수 있는 토대가 만들어져 있을 것이다


## 목 차

저자 소개

Table of Contents (목차)

들어가며

Chapter01. 인과 관계 기초

1.1 인과관계란 무엇인가?

1.2 인과추론의 두 가지 프레임워크

1.3 교란 (Confounding)

1.4 펄의 인과 계층 (Pearl's Causal Hierarchy)

1.5 구조적 인과 모델의 구현

1.6 상관관계 없는 인과관계

1.7 인과추론 최신 연구 동향

1.8 Causality Basic Python 실습

Chapter02. 회귀 분석과 인과추론

2.1 OLS 회귀 분석의 기초

2.2 역방향 회귀와 인과 방향성

2.3 모든 공변량을 통제해야 하는가?

2.4 회귀와 구조 인과 모델의 연결

2.5 비선형 항 포함하기

2.6 회귀 유형 최신 연구 동향

2.7 회귀 분석 (Regression) Python 실습

Chapter03. DAG 방향성 비순환 그래프와 인과 구조

3.1 그래프 이론 기초

3.2 방향성 비순환 그래프 (DAG)

3.3 인접 행렬

3.4 Python에서 그래프 표현

3.5 연결 구조와 독립성

3.6 기본 인과 구조

3.7 d-Separation과 그래프 독립성

3.8 마르코프 조건과 충실성

3.9 실제 응용 예시

3.10 DAG 그래픽 모델 Python 실습

Chapter04. d-분리와 식별가능 추정량

4.1 DAG와 경로 분석

4.2 d-분리 규칙

4.3 백도어 기준 (Backdoor Criterion)

4.4 전면 기준 (Frontdoor Criterion)

4.5 도구변수 (Instrumental Variables)

4.6 인과 추정량 (Causal Estimands)

4.7 식별가능성 (Identifiability)

4.8 실전 예제: 보조금과 매출

4.9 최신 연구 동향

4.10 d-분리와 식별가능 추정 Python 실습

Chapter05. DoWhy와 EconML 라이브러리

5.1 DoWhy 라이브러리 개요

5.2 DoWhy의 4단계 인과추론 파이프라인

5.3 EconML 라이브러리 개요

5.4 이중 머신러닝 (Double Machine Learning)

5.5 이질적 처치 효과의 다양한 추정 방법

5.6 DoWhy와 EconML의 실무 적용

5.7 고급 주제 및 확장

5.8 DoWhy & EconML Python 구현

Chapter06. 인과추론의 가정과 도전 과제

6.1 인과추론의 핵심 가정

6.2 SUTVA 가정

6.3 무시가능성 (Ignorability)

6.4 양성 가정 (Positivity)

6.5 일관성 (Consistency)

6.6 민감도 분석

6.7 선택 편향

6.8 미측정 교란

6.9 관찰 연구의 실무적 도전 과제

6.10 견고성 검증

6.11 인과 모델: 가정과 과제 Python 실습

Chapter07. 인과 머신러닝 기초

7.1 매칭 (Matching)

7.2 역확률 가중법 (IPW)

7.3 메타-학습자 (Meta-Learners)

7.4 S-Learner (단일 모델 학습자)

7.5 T-Learner (이원 모델 학습자)

7.6 X-Learner (확장형 모델 학습자)

7.7 메타-학습자 성능 비교

7.8 이중 강건 방법 (DR-Learner)

7.9 이중 머신러닝 (DML)

7.10 실제 응용: Hillstrom 데이터

7.11 고급 주제 및 확장

7.12 모범 사례 및 권장사항

7.13 Causal Machine Learning Python 실습

Chapter08. 고급 인과 머신러닝

8.1 고급 인과 머신러닝의 개요

8.2 메타-학습자: 빠른 복습

8.3 이중 기계 학습 (DML)

8.4 인과 숲 (Causal Forest)

8.5 일반화 무작위 숲 (GRF)

8.6 상승 모델링 (Uplift Modeling)

8.7 차원의 저주와 대응 전략

8.8 베이지안 인과 숲 (BCF)

8.9 정책 학습 (Policy Learning)

8.10 실제 응용 사례

8.11 최신 연구 동향

8.12 고급 인과추론 머신러닝 Python 실습

Chapter09. 인과 딥러닝

9.1 인과 딥러닝 개요

9.2 표현 학습과 인과성

9.3 TARNet과 CFR

9.4 메타-학습자 (복습)

9.5 Dragonnet

9.6 CEVAE

9.7 Deep IV

9.8 고급 주제와 최신 연구

9.9 인과 딥러닝 Python 예시

9.10 방법론 비교 및 성능 평가

9.11 인과적 딥러닝 Python 실습

9.12 결론 및 향후 연구 방향

마치며


## E-Book 구매

- Yes24: https://www.yes24.com/product/goods/189183079
- 교보문고: https://ebook-product.kyobobook.co.kr/dig/epd/ebook/E000012953221
- 알라딘: http://aladin.kr/p/6SL1J

## Github 코드: 

https://github.com/no-wave/llm-agent-RL




---
layout: post
title:  "Deep Learning for Text Style Transfer: A Survey"
categories: [ paper_review, AI, NLP ]
image: assets/images/text_image.png
---

# TST, style languages

같은 뜻을 지닌 말도 상황에 따라 다르게 표현됩니다. 예를 들어 비행기에서 승무원이 "please consider taking a seat."라고 하는 formal한 말투와
친구가 "come and seat!"하는 경우가 있을 것입니다. 상황에 따라 다른 말투를 ai가 배우게 하려면 어떠한 방법을 사용해야 할지 알아봅시다.

인공지능 시스템이 언어를 정확하게 이해하고 생성하기 위해서는 단순히 언어를 번역하는 것을 넘어 style(또는 attribute)으로 언어를 모델링하는 것이 필요합니다.

여기서 말한 attribute는 굉장히 광범위할 수 있는데, 실용성, 형식성, 친절함, 단순함, 성격, 감정, 장르 등 다양한 범위들로부터 선택될 수 있습니다. 

**text style transfer (TST)의 목적은 content를 보존하면서 동시에 자동적으로 style attribute를 조절하는 것입니다.**

TST의 활용은 광범위할 수 있는데 챗봇, 말투바꾸기 프로그램, 폭력적인 말 찾아내기 등등이 있을 것입니다.

Source sentence x: “Come and sit!” Source attribute a: Informal
Target sentence x′: “Please consider taking a seat.” Target attribute a′: Formal

위와 같이 x, x', a가 주어졌을 때 TST의 목표는 p(x'|a, x)를 모델링하는 것입니다. 

**TST에서는 style과 content를 구별하여 정의합니다. 그런데 구별 방식에 두가지 정의가 있습니다.**

첫번째 방법은 언어적 정의에 의한 것으로 '비기능적 언어 특성'(가령 예의바름)은 style, 그리고 '의미'는 content로 분류하는 방법

두번째 방법은 data-driven된 방법으로 두개의 corpora가 주어졌을 때 일관적인 것은 content, 비일관적이고 분산적인 것은 'style'로 분류하는 방법
(예를 들어 긍정리뷰셋, 부정리뷰셋)

첫번째 방법인 전통적인 TST 접근 방식은 domain specific한 용어 교체와 템플릿에 의존합니다. 한마디로 인간의 노동력이 많이 필요합니다.
 
최근의 NN(neural network)를 이용한 방식은 1. parallel corpora를 이용한 방법과 2. non-parallel corpora를 이용한 두가지 방법이 존재합니다.
parallel corpora는 content는 같지만 style은 다른 두 문장쌍 데이터를 말하는데, 이 데이터가 있을 경우 sec to sec 모델을 이용해볼 수 있습니다. 
그러나 parallel corpora는 많지 않아 non-parallel data를 이용하는 TST 방법이 다양하게 나오고 있습니다.
non-parallel data를 이용하는 방법론의 계보를 따라가 봅시다.

1. disentanglement, 텍스트를 latent space에서 content와 attribute으로 나누고 생성모델을 적용한 모델(Hu et al. 2017; Shen et al. 2017)
2. prototype editing, 1번과 비슷한 생성모델로, text 생성을 위해 문장 템플릿과 속성 마커를 추출하는 모델 (Li et al. 2018) 
3. pseudo-parallel corpus construction, pseudo-parallel corpus를 만들어 훈련시키는 모델(Zhang et al. 2018d; Jin et al. 2019)
4. 1,2,3에 Transformer-based models 적용한 방식 (Sudhakar, Upadhyay, and Maheswaran 2019; Malmi, Severyn, and Rothe 2020a)

TST의 발전으로 인해 persona-based dialog generation (Niu and Bansal 2018; Huang et al. 2018), stylistic summarization (Jin et al. 2020a), 
stylized language modeling to imitate specific authors (Syed et al. 2020), online text debiasing (Pryzant et al. 2020; Ma et al. 2020),
simile generation (Chakrabarty, Muresan, and Peng 2020)등의 다양한 down stream 영역까지도 발전 중에 있습니다.

### Data-Driven Definition of Style as the Scope of this Survey.

<img width="612" alt="style" src="https://user-images.githubusercontent.com/85322951/190151046-b6bd691b-cff1-47be-859b-4f2646a39ebf.png">

언어적, 규칙기반적인 정의에 의한 style은 미리 정해진 룰에 따라 고정적으로 정해진 반면, data-driven된 style은 고정적인 언어적 style과는 match되지 않더라도
딥러닝 모델에는 적용될 수 있는 style을 포함합니다.





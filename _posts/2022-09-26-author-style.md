---
layout: post
title:  "Steal the author's style"
categories: [ keyword, AI, NLP ]
image: assets/images/authors.png
use_math: true
---
# Adapting Language Models for Non-Parallel Author-Stylized Rewriting

논문링크: https://arxiv.org/pdf/1909.09962.pdf

해당 게시글은 제가 본 논문을 읽고 이해한 바를 정리한 내용입니다.  

틀린 내용이 있거나 보충이 필요한 내용, 또는 질문이 있다면 댓글로 남겨주시면 감사하겠습니다 :)  

글에서 사용된 단어 중 문자 그대로의 의미로 이해하기 보다는 **학술적 정의**로 받아들여야 하는 단어는 번역하지 않고 영어 그대로 쓰려고 노력하였으니 이 점 참고해주시기 바랍니다.

## Abstract

최근 [transformer](https://wikidocs.net/31379) 기반의 다양한 language modeling이 발전하고 있고 그 중 **스타일링된 텍스트**를 만드는 것에 대해 많은 관심이 쏠리고 있습니다.
이러한 배경에서 본 논문에서는 **[language model](https://wikidocs.net/21668)을 활용해 텍스트를 target 작가의 스타일로 다시 쓰는 모델**을 소개합니다.
예를 들면 평범한 일상의 문장을 input으로 넣으면 output으로 셰익스피어의 말투로 바뀐 문장을 내어주는 모델을 만드는 것입니다.
본 모델은 pre-trained language model을 [fine-tuning](https://eehoeskrap.tistory.com/186)해 author-stylized text를 만들어내는 것이 목적인데요,
이를 위해 [Denoising AutoEncoder)](https://deepinsight.tistory.com/126) loss를 사용해 [cascaded](https://daebaq27.tistory.com/79) encoder-decoder 구조에서 학습하였고 데이터는 author-specific corpus로 fine-tuning 하였습니다.  
language modeling을 할 때 언어의 뉘앙스를 그대로 가져가기가 어려울 때가 많은데, 해당 모델에서는 DAE loss를 활용해 parallel data없이도 모델이 뉘앙스를 학습할 수 있었습니다.

* 여기서 parallel data는 뜻은 같지만 스타일이 다른 두 문장의 쌍이라고 보면 되는데, (일반문장, 셰익스피어 어투의 문장)을 예로 들 수 있습니다. 이러한 parallel data를 이용하면 supervised learning이 가능하다는 장점이 있지만 현실적으로 parallel data는 없거나 매우 부족한 상황이기에 non-parallel data를 이용한 방법론이 주목을 받고 있습니다.

모델평가에 대해서는 target 작가에 대해 생성된 텍스트의 stylistic alignment를 어휘, 통사, 표면적 측면에서 정량화하기 위해 언어적 frame work를 활용했습니다. 정량적, 정성적 평가를 통해 해당 접근방식이 SOTA보다 원본을 잘 보존하면서도 target style을 더 잘 align한다는 것을 보여줍니다. 


## Introduction

최근 들어 NLP 분야에서 stylized text generation과 style transfer에 관심이 집중되고 있습니다.
stylized text generation과 style transfer 두 개의 tasks 모두 input text를 target style에 align하여 새로운 text를 만들어내는 것이 목표인데,이에 대한 작업들은 sentiment와 formality, 또는 두개의 결합에 대해 다양한 level 별로 말투를 바꾸는 것이었습니다.

- 예를 들어 형식적이면서 기분좋은 말투: 안녕하십니까 만나게 되어 반갑습니다, 비형식적이며 기분 좋은 말투: ㅎㅇㅎㅇ 반갑쓰

이러한 연구들은 주로 formality와 sentiment에 대해 정반대의 level을 지닌 parallel data를 이용해 이루어졌습니다.
그러나 author의 style(심리언어학적 측면의 개념이 아닌, author의 글에 드러난 언어적 선택으로서의 style)에 맞게 텍스트를 생성하는 것에 대해서는 연구가 부족했습니다.
Jhamtani et al. (2017)는 “Shakespearized text”를 만들려고 하긴 했지만 parallel data를 사용했다는 점에 한계가 있습니다.
parallel data를 이용한다면 author별로 parallel 데이터를 구축해야하는 한계가 있으므로 이에 의존하지 않는 방식이 필요했습니다.
이에 따라 본논문에서는 parallel data에 의존하지 않는 author-stylized rewriting 방법에 대해 propose하였습니다.

<img width="503" alt="스크린샷 2022-09-26 오후 4 23 02" src="https://user-images.githubusercontent.com/85322951/192216941-66b8c760-8f3b-40fd-910f-4c056be2259d.png">{: width="500" height="400"}

이 모델의 특징은 **SOTA language model을 fine tuning해 parallel data없이 target author의 스타일 특성에 맞는 text rewriting**에 성공했다는 점 입니다.
구체적인 학습방법은 다음과 같습니다.

1.  masked language model을 author courpus와 wikipedia data 두가지 데이터를 활용해 pre-train합니다.
2. encoder와 decoder 둘 다에 앞서 pre-train한 언어모델의 가중치로 initialization합니다.
3. 해당 모델을 denoising autoencoder loss로 학습해 target author의 corpus로 fine-tuning합니다.

본 모델은 특정 스타일을 지니거나 아무런 스타일이 없는 text를 input으로 받아 target author's style을 가진 text로 변환합니다.
그러나 author의 스타일은 어휘, 구문, 의미 3가지 언어적 특성에 의해 결정되기에 모델 평가하기는 쉽지 않습니다.
따라서 본 논문은 stylistic alignment의 정도를 수량화하기 위해 새로운 방법을 제안합니다.
뒤에서 더 자세히 알아봅시다.

본 논문의 key contribution은 다음의 3가지 입니다.

- parallel data 없이 SOTA모델을 fine tuning해 author-stylized text를 만드는 방법을 propose하고 evaluate했다.
- 스타일의 어휘적 측면과 통사적 측면의 alignment을 설명하는 평가 프레임워크를 제안한다. 기존 평가 기법과 달리, 해당 평가방법은 언어적으로 쉽게 해석할 수 있다.
-author-stylized text generation에 대해 baseline보다 질적, 양적으로 발전했다.

## Related Work

### Stylized Text Geneartion(STG)
최근의 연구에서 언어심리학적 측면에서 sentiment나 formality의 다양한 level을 구현하는 방식의 STG가 등장했습니다.
접근방식은 supervised인 방식(parallel data에 의존), unsupervised 방식이 있습니다.
unsupervised방식 중 influential한 방법들은  (a) using readily available classificationbased discriminators to guide the process of generation (Fuet al. 2018), (b) using simple linguistic rules to achieve alignment with the target style (Li et al. 2018), or (c) using auxiliary modules (called scorers) that score the generation process on aspects like fluency, formality and semantic relatedness while deciding on the learning scheme
of the encoder-decoder network (Jain et al. 2019)이 있습니다.
그러나 본 논문에서 만들고자 하는 모델은 author의 style을 가진 text generation 문제 이므로 discriminator나 score를 만들기는 어렵습니다. 
또한 author의 style을 가진 text generation을 하기 위해 rule-based generation을 하는 것은 수많은 rule과 작가들의 스타일에 대한 정의를 요구하므로 너무나 복잡합니다.
이러한 이유로 본 논문에서는 non-parallel data를 이용해 SOTA language model을 pre-training하기로 했습니다.
language model을 사용한 가장 큰 이유는 stylistic rewriting이 단순한 text generation에 기반을 두기 때문입니다.
본 논문의 목적과 비슷하게 author의 스타일로 text를 adapt하기 위한 시도가 있어왔습니다.

Tikhonov and Yamshchikov (2018)는 stylistic variables들의 임베딩을 concatenate하고 conditioning을 사용해 
style을 end-to-end로 학습하여 author-stylized poetry를 생성하는 모델을 만들었지만 본논문과 똑같이 rewriting을 시도하지는 않았습니다.
Jhamtani et al. (2017)는 parallel data를 사용해 “Shakespearized” version의 현대영어를 만들어내려고 했습니다.
그러나 본 논문은 parallel data가 아닌 non-parallel data를 활용한다는 점에서 차이가 있습니다.
즉 본논문의 모델은 오직 target author의 corpus만을 필요로 합니다.
뒤에서 나오겠지만 본논문이 제시하는 모델은 non-parallel data를 사용했음에도 content preservation과 style transmission metric에서  Jhamtani et al. (2017)에 비견될만한 성능을 보입니다.

* Language models
Generative pre-training of sentence encoders (Radford et al. 2018; Devlin et al. 2019; Howard and Ruder 2018)는 nlp task에서 큰 발전을 가져왔습니다.
해당 접근방식은 transformer language model을 커다란 unsupervised 코퍼스로 학습시키고 분류문제 또는 추론기반의 NLU 테스크에 fine-tuning해 접목하는 방법입니다. Lample and Conneau(2019)는 이를 토대로 cross-lingual language models을 만들었습니다.
이에 착안하여 본 논문에서는 generative pre-training을 author-stylized rewriting 테스크에 적용합니다.
 GPT-2 (Radfordet al. 2019)에서는 크고 다양한 코퍼스로 pre-trained 되었고 NL generation을 비롯한 다양한 도메인과 데이터셋에서 잘 작동합니다.
 unsupervised pre-training는 이전의 말로 다음말을 예측하는 모델로 다음단어가 나올 확률을 예측합니다.
(일반적으로 causal language modeling (CLM) objective로 알려져 있습니다.
$P(y_t|y_{1:t-1}, x)$

text generation 테스크에서만 특징적으로 input prompt x를 취하여 input의 context를 고수하도록 합니다.
GPT-2에서는 author-specific corpus로 fine-tuned된 경우 target author의 style에 대해 상당히 stylistic alignment가 잘 됩니다. 그러나 stylistic rewriting(우리가 하려는 테스크)와 stylized text generation의 본질적인 차이로 인해 GPT-2는 content를 보존하는데에 좋은 성능을 보이지는 못했습니다.
stylistic rewriting에서는 stylized generation에서 input text에서의 정보(content)를 유지하지만 GPT-2에 의한 스타일 생성은 입력 프롬프트와 관련된 콘텐츠를 생성하므로 미세 조정된 GPT-2는 stylistic rewriting을 처리할 수 없습니다.
 Lample and Conneau (2019)에서는 crosslingual language models를 3개의 다른 language modeling objectives(CLM, MLM, TLM)로 pre-training을 하고 encoder와 decoder를 

* Evaluating Stylized Generation

## Propsed Approach : StyleLM

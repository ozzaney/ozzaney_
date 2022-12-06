---
layout: post
title:  "Steal the author's style"
categories: [ keyword, AI, NLP ]
image: assets/images/authors.png
use_math: true
---
# Adapting Language Models for Non-Parallel Author-Stylized Rewriting

[논문링크](https://arxiv.org/pdf/1909.09962.pdf)

해당 게시글은 제가 본 논문을 읽고 이해한 바를 정리한 내용입니다.  

틀린 내용이 있거나 보충이 필요한 내용, 또는 질문이 있다면 댓글로 남겨주시면 감사하겠습니다 :)  

## Abstract

최근 [transformer](https://wikidocs.net/31379) 기반의 다양한 language modeling이 발전하고 있고 그 중 **스타일링된 텍스트**를 만드는 것에 대해 많은 관심이 쏠리고 있습니다. **스타일링된 텍스트**란 뜻은 동일하지만 스타일을 다르게 만든 텍스트를 이르는 말로, 말투나 부정/긍정, formal/informal 등의 여러가지 범주를 스타일로 정의해볼 수 있습니다.
이러한 배경에서 본 논문에서는 **[language model](https://wikidocs.net/21668)을 활용해 텍스트를 target 작가의 스타일로 다시 쓰는 모델**을 소개합니다. 본 논문에서는 스타일을 **작가의 문체**로 정의했습니다.
모델에 평범한 일상의 문장을 input으로 넣으면 output으로 셰익스피어라는 target 작가의 말투로 바뀐 문장을 output으로 만들어내는 모델을 만든 것입니다.
본 모델은 사전학습 모델을 [fine-tuning](https://eehoeskrap.tistory.com/186)해 작가의 말투로 변형된 텍스트를 만들어내는 것이 목적인데요,
이를 위해 [Denoising AutoEncoder(DAE)](https://deepinsight.tistory.com/126) loss를 사용해 [cascaded](https://daebaq27.tistory.com/79) encoder-decoder 구조에서 학습하였고 데이터는 author-specific corpus로 fine-tuning 하였습니다.  
language modeling을 할 때 언어의 뉘앙스를 그대로 가져가기가 어려울 때가 많은데, 해당 모델에서는 DAE loss를 활용해 parallel data없이도 모델이 뉘앙스를 학습할 수 있었습니다.

* 여기서 parallel data는 뜻은 같지만 스타일이 다른 두 문장의 쌍이라고 보면 되는데, (일반문장, 셰익스피어 어투의 문장)을 예로 들 수 있습니다. 이러한 parallel data를 이용하면 supervised learning이 가능하다는 장점이 있지만 현실적으로 parallel data는 없거나 매우 부족한 상황이기에 non-parallel data를 이용한 방법론이 주목을 받고 있습니다.

모델평가에 대해서는 target 작가에 대해 생성된 텍스트의 stylistic alignment를 어휘, 통사, 표면적 측면에서 정량화하기 위해 언어적 frame work를 활용했습니다. 정량적, 정성적 평가를 통해 해당 접근방식이 SOTA보다 원본을 잘 보존하면서도 target style을 더 잘 align한다는 것을 보여줍니다. 


## Introduction

최근 들어 NLP 분야에서 stylized text generation과 style transfer에 관심이 집중되고 있습니다.
stylized text generation과 style transfer 두 개의 tasks 모두 input text를 target style에 align하여 새로운 text를 만들어내는 것이 목표인데,이에 대한 작업들은 sentiment와 formality, 또는 두개의 결합에 대해 다양한 level 별로 말투를 바꾸는 것이었습니다.

(예시)  
내용은 동일하나 formality만 다른 두 문장
- 형식적이면서 기분좋은 말투: 안녕하십니까 만나게 되어 반갑습니다  
- 비형식적이며 기분 좋은 말투: ㅎㅇㅎㅇ 반갑쓰  

이러한 연구들은 주로 formality와 sentiment에 대해 정반대의 level을 지닌 parallel data를 이용해 이루어졌습니다.
그러나 author의 style(심리언어학적 측면의 개념이 아닌, author의 글에 드러난 언어적 선택으로서의 style)에 맞게 텍스트를 생성하는 것에 대해서는 연구가 부족했습니다.
Jhamtani et al. (2017)는 “Shakespearized text”를 만들려고 하긴 했지만 parallel data를 사용했다는 점에 한계가 있습니다.
parallel data를 이용한다면 author별로 parallel 데이터를 구축해야하는 한계가 있으므로 이에 의존하지 않는 방식이 필요했습니다.
이에 따라 본논문에서는 parallel data에 의존하지 않는 author-stylized rewriting 방법에 대해 propose하였습니다.

<img width="503" alt="스크린샷 2022-09-26 오후 4 23 02" src="https://user-images.githubusercontent.com/85322951/192216941-66b8c760-8f3b-40fd-910f-4c056be2259d.png">{: width="700" height="400"}

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
- author-stylized text generation에 대해 baseline보다 질적, 양적으로 발전했다.

## Related Work

### Stylized Text Geneartion(STG)
최근의 연구에서 언어심리학적 측면에서 sentiment나 formality의 다양한 level을 구현하는 방식의 STG가 등장했습니다.
본 논문에서는 non-parallel data를 이용해 SOTA language model을 pre-training하기로 했습니다.
language model을 사용한 가장 큰 이유는 stylistic rewriting이 단순한 text generation에 기반을 두기 때문입니다.
본 논문의 목적과 비슷하게 author의 스타일로 text를 adapt하기 위한 시도가 있어왔습니다.

Tikhonov and Yamshchikov (2018)는 stylistic variables들의 임베딩을 concatenate하고 conditioning을 사용해 
style을 end-to-end로 학습하여 author-stylized poetry를 생성하는 모델을 만들었지만 본논문과 똑같이 rewriting을 시도하지는 않았습니다.
Jhamtani et al. (2017)는 parallel data를 사용해 “Shakespearized” version의 현대영어를 만들어내려고 했습니다.
그러나 본 논문은 parallel data가 아닌 non-parallel data를 활용한다는 점에서 차이가 있습니다.
즉 본논문의 모델은 오직 target author의 corpus만을 필요로 합니다.
뒤에서 나오겠지만 본논문이 제시하는 모델은 non-parallel data를 사용했음에도 content preservation과 style transmission metric에서  Jhamtani et al. (2017)에 비견될만한 성능을 보입니다.

### Language models

Generative pre-training of sentence encoders (Radford et al. 2018; Devlin et al. 2019; Howard and Ruder 2018)는 nlp task에서 큰 발전을 가져왔습니다.
해당 접근방식은 transformer language model을 커다란 unsupervised 코퍼스로 학습시키고 분류문제 또는 추론기반의 NLU 테스크에 fine-tuning해 접목하는 방법입니다. Lample and Conneau(2019)는 이를 토대로 cross-lingual language models을 만들었습니다.
이에 착안하여 본 논문에서는 generative pre-training을 author-stylized rewriting 테스크에 적용합니다.
 GPT-2 (Radfordet al. 2019)에서는 크고 다양한 코퍼스로 pre-trained 되었고 NL generation을 비롯한 다양한 도메인과 데이터셋에서 잘 작동합니다.
 unsupervised pre-training는 이전의 말로 다음말을 예측하는 모델(causal language modeling, CLM)로 다음단어가 나올 확률을 예측합니다. 다음은 text generation에 대한 CLM의 확률식입니다.  
 
$P(y_t|y_{1:t-1}, x)$  
위와 같이 CLM중 에서도 text generation 테스크에서만 특징적으로 input prompt x를 취하여 input의 context를 반영하도록 합니다.
GPT-2에서는 author-specific corpus로 fine-tuned된 경우 target author의 style에 대해 상당히 stylistic alignment가 잘 됩니다. 그러나 stylistic rewriting(우리가 하려는 테스크)와 stylized text generation의 본질적인 차이로 인해 GPT-2는 content를 보존하는데에 좋은 성능을 보이지는 못했습니다.
stylistic rewriting에서는 stylized generation에서 input text에서의 정보(content)를 유지하지만 GPT-2에 의한 스타일 생성은 입력 프롬프트와 관련된 콘텐츠를 **생성**하므로 미세 조정된 GPT-2는 stylistic rewriting을 처리할 수 없습니다.
즉 GPT2를 사용하면 내용을 유지하는 것이 아니라 관련된 내용을 새롭게 생성하기에, 내용을 온전히 유지하면서 타겟 작가의 스타일을 닮은 글을 생성하는 rewriting task에는 적합하지 않습니다.

해당 모델에서는 MLM에 대해 거대 corpus로 학습시킨 뒤, encoder-decoder setup에서 DAE loss를 사용해 author specific corpus로 fine-tuning했습니다.

### Evaluating Stylized Generation

Fu et al. (2018)에서는 style transfer models의 성능을 1. content preservation, 2. transfer strength이라는 두가지의 축으로 평가했습니다.
content preservation의 성능은 input과 만들어진 text간에 얼마나 유사한지를 나타내는 [BLEU](https://wikidocs.net/31695)로 평가하고, transfer strength는 target style에 대해 만들어진 텍스트가 얼마나 align하는가로 평가합니다. unsupervised learning이기에 generated된 text의 style이 잘 생성되었는지를 판단할 정답 데이터(groud-truth)가 없고, 그래서 transfer strength를 평가하기가 쉽지 않습니다. 
따라서 본 논문은 author-stylized text에서 스타일의 여러 어휘 및 통사적 측면의 alignment를 정량화하는 평가법을 제안합니다.  

## Propsed Approach : StyleLM

이제 본격적으로 모델에 대해 설명해보도록 하겠습니다.
해당 모델은 두가지 중요한 양상을 가집니다.
1. transformer based model을 커다란 데이터셋으로 pre training하고 author specific corpus로 DAE를 사용해 fine-tuning한다.
2. 해당 모델은 self-supervised 방식으로 학습되며 parallel data에 의존하지 않는다.

<img width="927" alt="스크린샷 2022-09-27 오후 1 34 56" src="https://user-images.githubusercontent.com/85322951/192433048-8dbc1cb7-13b3-4c23-bef0-b7d88297c8b2.png">{: width="700" height="400"}

위 그림은 해당 모델이 프레임워크를 표현한 것입니다. 

먼저 방대한 corpus(Author들의 전체 corpus, 위키피디아 corpus)로 transformer based language model을 MLM 목적함수를 가지고 훈련합니다.
MLM은 masked된 단어가 앞뒤 맥락(bidirectional context)을 통해 유추되도록 합니다.
x가 주어진 문장일 떄 $x_{\u}$는 x에서 position u가 masked된 상태를 나타낸다고 합시다.
(u위치의 token이 masked될 경우 해당 token이 [MASK]로 대체됩니다. 이를 통해 masked된 문장의 길이가 전과 달라지지 않도록 합니다.)
MLM 목적함수는 $x_{\u}$를 input으로 받아들였을 때 $x_u$를 예측하도록 language model을 훈련시킵니다. 

<img width="370" alt="스크린샷 2022-09-27 오후 1 57 31" src="https://user-images.githubusercontent.com/85322951/192439148-f6c52ec6-6381-4e14-aed7-d93d0ef1e1d0.png">

이때 X는 전체 training corpus입니다.
MLM을 pre-training을 하기 위해  Devlin et al. (2019)의 방법을 따라, 각각의 input sentence마다 전체 중 15%의 token들을 랜덤하게 mask합니다. 그리고 그 중 80%를 [MASK]로 대체하고 10%는 radom token으로 대체하며, 나머지 10%는 바꾸지 않은 채로 둡니다.

이제 author-stylized rewriting을 가능하게 하기 위해, 앞서 만든 pre-trained laguage model 두 개를 직렬으로 이어붙여 encoder-decoder 구조를 만듭니다. 
즉 encoder와 decoder의 학습가능한 파라미터들이 pre-trained LM으로 초기화되도록 합니다.

**Transformer 기반 언어 모델의 구조는 attention 메커니즘이 Transformer 설계에 내재되어 있기 때문에 인코더의 출력과 디코더의 입력을 명시적으로 align하지 않고도 pre-trained된 LM의 두 인스턴스를 직렬로 연결할 수 있습니다.**

이제 연결된 encoder-decoder를 다음의 DAE-loss를 이용해 fine-tune합니다.

<img width="371" alt="스크린샷 2022-09-27 오후 2 14 47" src="https://user-images.githubusercontent.com/85322951/192439238-644b3aa2-b529-47e3-9245-434a45ea391b.png">

이때 C(x)는 input 문장 x의 noisy한 버전이고 S는 target author corpus의 문장들입니다.
C(x)를 만들기 위해 x안의 모든 각각의 단어들에 대해서 $P_{drop}$으로 버리고, $P_{blank}$으로 [BLANK]로 대체합니다. 이렇게 하는 이유는 모델에게 일부러 알아보기 어려운 문장을 알려줌으로써 noise에 robust한 model을 만들기 위해서입니다. 

![denoising ](https://user-images.githubusercontent.com/85322951/192441299-a8b9ba23-4397-4ca2-955a-751c15fb39e9.png)

pre-trained LM를 각각 두 개로 복사해 각각 enocdoer와 decoder로 삼아 두 개를 연결합니다.
이는 autoencoder 구조로 encoder-decoder 구조의 일종입니다.
이제 해당 구조에 input으로 텍스트의 noisy version을 넣어주면 enocder는 masked words를 유추하여 만들어냅니다.(이것은 원래의 pre-trained model의 의도대로 된 모습입니다.)
이렇게 encoder가 만든 output은 decoder의 input으로 들어가게 되고 decoder에서는 noisy input text의 clear한 버전을 reconstruct하게 됩니다.
즉 모델의 디코더가 인코더를 통과한 noisy text를 rewrite할 때 target author의 스타일을 가진 문장을 생성하게 됩니다.

### Implementation detail

<img width="927" alt="스크린샷 2022-09-27 오후 1 34 56" src="https://user-images.githubusercontent.com/85322951/192433048-8dbc1cb7-13b3-4c23-bef0-b7d88297c8b2.png">{: width="700" height="400"}

본논문에서는 MLM을 pre-training할 때 12-layer의 tranformer encoder(Vaswani et al. 2017)를 사용하였고, 해당 transformer encoder에서 GLEU 활성화함수(Hendrycks and Gimpel 2017)를 이용, 하이퍼파리미터로는 hidden unit는 512, 16 heads, dropout 비율은 0.1을 채택했고 positional embedding을 학습했습니다. 또 Adam optimizer를 사용했고 learning rate은 0.0001이었습니다.
학습시 input으로 256개 토큰의 streams를 이용했고 mini-batch size는 32를 채택했습니다.
학습은 validation set에 대해 LM의 [perplexity](https://wikidocs.net/21697)가 더이상 줄어들지 않을 때까지 학습하였습니다.
target author에 대해 fine-tune할 때 사용하는 encoder와 decoder 모두에서 똑같은 pre-trained MLM transformer를 초기 값으로 사용합니다. 이때 하이퍼파라미터는 pre-training을 할 때와 동일한 값을 사용합니다.(Lample and Conneau (2019)를 참고한 방법론)
$P_{drop}$, $P_{blank}$는  둘 다 0.1로 정했고 모델이 수렴할 때까지 fine-tune을 했습니다.
또한 거대한 corpus를 다루기 위해 combined training dataset에 Byte Pair Encoding [BPE](https://wikidocs.net/22592)를 적용했고 80k BPE codes를 훈련시켰습니다.




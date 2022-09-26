---
layout: post
title:  "Steal the author's style"
categories: [ keyword, AI, NLP ]
image: assets/images/authors.png
use_math: true
---
# Adapting Language Models for Non-Parallel Author-Stylized Rewriting

## Abstract

최근 [transformer](https://wikidocs.net/31379) 기반의 다양한 언어모델링이 발전하고 있고 스타일링된 텍스트를 만드는 것에 대해 많은 관심이 쏠리고 있습니다.
이러한 발전에 힘입어 본 논문에서는 **언어모델의  일반화 능력을 활용해 텍스트를 target 작가의 스타일로 다시 쓰는 모델**을 소개합니다.
예를 들면 평범한 일상의 말을 input으로 넣으면 output으로 셰익스피어의 말투를 내어주는 모델을 만들어 볼 수 있습니다.
본 모델은 pre-trained language model을 fine-tuning해 author-stylized text를 만들어내는 것이 목적인데요,
이를 위해 DAE(denoising autoencoder) loss를 사용해 encoder-decoder 구조에서 학습하였고 데이터는 author-specific corpus로 fine-tuning 하였습니다.  
언어모델링을 할 때 언어의 nuance(뉘앙스)를 그대로 가져가기가 어려울 때가 많은데, 해당 모델에서는 DAE loss를 활용해 parallel data없이도 모델이 nuance를 학습할 수 있었습니다. 그게 어떻게 가능할 수 있었는지는 본론에서 더 자세히 이야기해보도록 하겠습니다.

* 여기서 parallel data는 뜻은 같지만 스타일이 다른 두 문장의 쌍이라고 보면 되는데, (일반문장, 셰익스피어 어투의 문장)을 예로 들 수 있습니다. 이러한 parallel data를 이용하면 supervised learning이 가능하다는 장점이 있지만 현실적으로 parallel data는 없거나 매우 부족한 상황이기에 non-parallel data를 이용한 방법론이 주목을 받고 있습니다.

모델평가에 대해서는 target 작가에 대해 생성된 텍스트의 stylistic alignment를 어휘, 통사, 표면적 측면에서 정량화하기 위해 언어적 frame work를 활용했습니다. 정량적, 정성적 평가를 통해 해당 접근방식이 SOTA보다 원본을 잘 보존하면서도 target style을 더 잘 alignment한다는 것을 보여줍니다. 

## Introduction

NLP 분야의 다양한 재미난 문제들(장르 분류, 작가 profiling, 감성분석, 사회적 관계분류 등)에 대한 sutdy가 계속해서 있어왔습니다.
최근 들어 stylized text generation과 style transfer에 관심이 집중되는데요, 두 개의 tasks 모두 input text를 target style해 align하여 새로운 text를 만들어내는 것이 목표입니다.
지금까지 관련된 대다수의 작업들은 '감정 수준별 text 만들기' ,'formality alginment', '인코더 디코더를 학습시킬 때 보조사를 활용해 유창성(fluency), formality(형식성), semantic relatedness(의미 관련성)의 양상에 대해 점수를 매기는 방식' 등에 대한 것들이었습니다.
그러나 해당 모델은 author의 style을 가진 text generation문제 이므로 discriminator나 score를 만들기는 어렵습니다. 
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




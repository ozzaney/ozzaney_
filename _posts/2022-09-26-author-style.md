---
layout: post
title:  "Adapting Language Models for Non-Parallel Author-Stylized Rewriting"
categories: [ keyword, AI, NLP ]
image: assets/images/authors.png
use_math: true
---

## Abstract

최근 (transformer)[https://wikidocs.net/31379] 기반의 다양한 언어모델링이 발전하고 있고 스타일링된 텍스트를 만드는 것에 대해 많은 관심이 쏠리고 있습니다.
이러한 발전에 힘입어 본 논문에서는 **언어모델의  일반화 능력을 활용해 텍스트를 target 작가의 스타일로 다시 쓰는 모델**을 소개합니다.
예를 들면 평범한 일상의 말을 input으로 넣으면 output으로 셰익스피어의 말투를 내어주는 모델을 만들어 볼 수 있습니다.
본 모델은 pre-trained language model을 fine-tuning해 author-stylized text를 만들어내는 것이 목적인데요,
이를 위해 DAE(denoising autoencoder) loss를 사용해 encoder-decoder 구조에서 학습하였고 데이터는 author-specific corpus로 fine-tuning 하였습니다.  
언어모델링을 할 때 큰 어려움이 언어의 nuance(뉘앙스)를 그대로 가져가기 어렵다는 건데요, 해당 모델에서는 DAE loss를 활용해 parallel data없이도 모델이 nuance를 학습할 수 있었습니다. 그게 어떻게 가능할 수 있었는지는 본론에서 더 자세히 이야기해보도록 하겠습니다.

* 여기서 parallel data는 뜻은 같지만 스타일이 다른 두 문장의 쌍이라고 보면 되는데, (일반문장, 셰익스피어 어투의 문장)을 예로 들 수 있습니다. 이러한 parallel data를 이용하면 supervised learning이 가능하다는 장점이 있지만 현실적으로 parallel data는 없거나 매우 부족한 상황이기에 non-parallel data를 이용한 방법론이 주목을 받고 있습니다.

모델평가에 대해서는 target 작가에 대해 생성된 텍스트의 stylistic alignment를 어휘, 통사, 표면적 측면에서 정량화하기 위해 언어적 frame work를 활용했습니다. 정량적, 정성적 평가를 통해 해당 접근방식이 SOTA보다 원본을 잘 보존하면서도 target style을 더 잘 alignment한다는 것을 보여줍니다. 

## Introduction


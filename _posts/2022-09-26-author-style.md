---
layout: post
title:  "Adapting Language Models for Non-Parallel Author-Stylized Rewriting"
categories: [ keyword, AI, NLP ]
image: assets/images/author.png
use_math: true
---

## Abstract

최근 transformer 기반의 다양한 언어모델링이 발전하고 있고 스타일링된 텍스트를 만드는 것에 대해 많은 관심이 쏠리고 있습니다.
이러한 발전에 힘입어 본 논문에서는 **언어모델의  일반화 능력을 활용해 텍스트를 타겟 작가의 스타일로 다시 쓰는 모델**을 소개합니다.
예를 들면 평범한 일상의 말을 input으로 넣으면 output으로 셰익스피어의 말투를 내어주는 모델을 만들어 볼 수 있습니다.
본 모델은 pre-trained language model이 author-stylized text를 만들어내도록 하기 위해 author-specific corpus를 이용해 
fine-tuning 하였습니다.

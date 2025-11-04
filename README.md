<h1 align="center"> Personalized Language Models via Privacy-Preserving Evolutionary Model Merging </h1>

<div align="center" style="line-height: 1;">
  <a href="https://arxiv.org/abs/2503.18008" target="_blank">
    <img alt="arXiv" src="https://img.shields.io/badge/arXiv-2503.18008-b31b1b?logo=arxiv&logoColor=white" />
  </a>
  <a href="https://github.com/kykim0/PriME" target="_blank"><img alt="GitHub"
    src="https://img.shields.io/badge/GitHub-PriME-000000?logo=github&logoColor=white&color=000000"/></a>
</div>


## Overview

Personalization in language models aims to tailor model behavior to individual users or user groups. Prompt-based methods incorporate user preferences into queries, while training-based methods encode them into model parameters. Model merging has also been explored for personalization under limited data. However, existing methods often fail to directly optimize task-specific utility and lack explicit mechanisms for privacy preservation.

<br>
<div align="center">
<img src="./asset/concept.png" width="75%" height="75%">
</div>
<br>

To address the limitations, we propose Privacy-Preserving Model Merging via Evolutionary Algorithms (PriME), a novel personalization approach that employs gradient-free methods to directly optimize utility while reducing privacy risks. By integrating privacy preservation into the optimization objective, PriME creates personalized modules that effectively capture target user preferences while minimizing privacy risks for data-sharing users. Experiments on the LaMP benchmark show that PriME consistently outperforms a range of baselines, achieving up to a 45% improvement in task performance. Further analysis demonstrates that PriME achieves a superior privacy-utility trade-off compared to a prior state-of-the-art, with enhanced robustness to membership inference attacks and greater utility in capturing user preferences.


## Citation ##

If our work is useful in your research, please consider citing it:
```bibtex
@inproceedings{kim2025personalized,
  title={Personalized Language Models via Privacy-Preserving Evolutionary Model Merging},
  author={Kim, Kyuyoung and Shin, Jinwoo and Kim, Jaehyung},
  booktitle={Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing},
  year={2025}
}

# SentiSynset
This repository is for the paper Identifying Emotional and Polar Concepts via Synset Translation. In *Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (\*SEM 2024)*, pages 142–152, Mexico City, Mexico. Association for Computational Linguistics.

[[Paper](https://aclanthology.org/2024.starsem-1.12/)] [[Poster](https://github.com/UAlberta-NLP/SentiSynset/blob/main/assets/poster.pdf)] [[Slides](https://github.com/UAlberta-NLP/SentiSynset/blob/main/assets/slides.pdf)]

## Directory
+ **emolex** - English and translated multingual NRC Emotion Lexicons (EmoLex)
+ **generated_files** - Intermediary files generated while creating SentiSynset
```
SentiSynset
├─── assets
|    ├─── paper.pdf
|    ├─── poster.pdf
|    └─── slides.pdf
├─── emolex
|    ├─── NRC-Emotion-Lexicon-Wordlevel-v0.92.txt
|    └─── OneFilePerLanguage
|         ├─── Afrikaans-NRC-EmoLex.txt
|         ├─── Albanian-NRC-EmoLex.txt
|         ├─── ...
|         └─── Zulu-NRC-EmoLex.txt
├─── generated_files
|    ├─── all_wn_translations.pkl
|    ├─── lemmatized_multilingual_lexicons.pkl
|    ├─── non_sentiment_normalized_lemmatized_multilingual_lexicons.pkl
|    └─── normalized_lemmatized_multilingual_lexicons.pkl
├───.gitignore
├─── babelnet_conf.yml
├─── create_lexicons.py
├─── load_multingual_translations.py
├─── main.py
├─── README.md
├─── requirements.txt
├─── select_languages.pkl
├─── sentisynset_lexicon.xml
└─── translations.py
```

## Dependencies
+ python == 3.8
+ babelnet == 1.1.0
+ langcodes == 3.4.0
+ nltk == 3.8.1
+ simplemma == 0.9.1
+ spacy == 3.7.5
+ Unidecode == 1.3.8
+ xmltodict == 0.13.0

## Setup
Please ensure required packages are already installed. A virtual environment is recommended.
```bash
$ cd SentiSynset
$ pip install pip --upgrade
$ pip install -r requirements.txt
```

## Run
The script needs to be able to read data from a local copy of the BabelNet indices (you cannot use the BabelNet API in online mode as you will quickly exceed the daily requests limit). Note that the BabelNet API requires Python 3.8.
```bash
$ python3 main.py
```

## Authors
* **Logan Woudstra** - lwoudstr@ualberta.ca
* **Moyo Dawodu** - mdawodu@ualberta.ca
* **Frances Igwe** - figwe@ualberta.ca
* **Senyu Li** - senyu@ualberta.ca
* **Ning Shi** - ning.shi@ualberta.ca
* **Bradley Hauer** - bmhauer@ualberta.ca
* **Grzegorz Kondrak** - gkondrak@ualberta.ca

## BibTex
```bibTex
@inproceedings{woudstra-etal-2024-identifying,
    title = "Identifying Emotional and Polar Concepts via Synset Translation",
    author = "Woudstra, Logan  and
      Dawodu, Moyo  and
      Igwe, Frances  and
      Li, Senyu  and
      Shi, Ning  and
      Hauer, Bradley  and
      Kondrak, Grzegorz",
    editor = "Bollegala, Danushka  and
      Shwartz, Vered",
    booktitle = "Proceedings of the 13th Joint Conference on Lexical and Computational Semantics (*SEM 2024)",
    month = jun,
    year = "2024",
    address = "Mexico City, Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.starsem-1.12",
    pages = "142--152",
    abstract = "Emotion identification and polarity classification seek to determine the sentiment expressed by a writer. Sentiment lexicons that provide classifications at the word level fail to distinguish between different senses of polysemous words. To address this problem, we propose a translation-based method for labeling each individual lexical concept and word sense. Specifically, we translate synsets into 20 different languages and verify the sentiment of these translations in multilingual sentiment lexicons. By applying our method to all WordNet synsets, we produce SentiSynset, a synset-level sentiment resource containing 12,429 emotional synsets and 15,567 polar synsets, which is significantly larger than previous resources. Experimental evaluation shows that our method outperforms prior automated methods that classify word senses, in addition to outperforming ChatGPT. We make the resulting resource publicly available on GitHub.",
}
```
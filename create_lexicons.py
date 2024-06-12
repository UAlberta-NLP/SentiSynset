import nltk
import xmltodict
import langcodes
import re
from nltk.corpus import wordnet
import pickle
from unidecode import unidecode
import simplemma
import spacy

nltk.download('sentiwordnet')
nltk.download('stopwords')
nltk.download('wordnet')

# need to run 'python -m spacy download en_core_web_md' to load
nlp = spacy.load('en_core_web_md')


class CreateLexicon:
    def __init__(self):
        # define emotions/polarities
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        self.sentiments = self.emotions + ['negative', 'positive']
        self.sentiments.sort()

        # load english NRC-lexicon
        self.word_lexicon = {}
        self.word_lexicon_sentiment = {}
        self.load_english_NRC()

        # load multingual lexicons
        self.lemmatized_multilingual_lexicons = {}
        self.normalized_lemmatized_multilingual_lexicons = {}
        self.non_sentiment_normalized_lemmatized_multilingual_lexicons = {}
        self.load_lexcions()

        # get all wn synsets
        self.all_wn_synsets = {synset for synset in wordnet.all_synsets()}

        # load bn translations
        self.bn_all_translations = {}
        self.clean_bn_translations()

        self.plutchik_opposite_emotion = {'anger': 'fear',
                                          'anticipation': 'surprise',
                                          'disgust': 'trust',
                                          'fear': 'anger',
                                          'joy': 'sadness',
                                          'sadness': 'joy',
                                          'surprise': 'anticipation',
                                          'trust': 'disgust',
                                          'positive': 'negative',
                                          'negative': 'positive',
                                          'ambigous': 'ambigous'}

        # init the 2 lexicons
        self.extended_lexicon = {}
        self.extended_sentiment_lexicon = {}

    def split_multiword_phrases(self, all_translations):
        """associates the emotions attached to a multi-word phrase to all words in the phrase"""
        split_translations = {language.lower(): {} for language in all_translations}
        for language, translations in all_translations.items():
            for synset, translated_words in translations.items():
                new_translations = set()
                for word in translated_words:
                    new_translations.add(word.replace('_', ' '))
                split_translations[language.lower()][synset] = new_translations
        return split_translations

    def get_lang_code(self, language):
        """gets the 2-letter language code from a language name"""
        if language == 'Norwegian':
            return 'nb'
        else:
            return langcodes.find(language).language

    def lemmatize_translations(self, all_translations):
        """lemmatizes words"""
        count = 0
        lemmatized_translations = {language.lower(): {} for language in all_translations}
        for language, translations in all_translations.items():
            if language not in ['korean', 'chinese']:  # langauges not support by simplemma
                language_code = self.get_lang_code(language)
                for synset, translated_words in translations.items():
                    new_translations = set()
                    for word in translated_words:
                        if word == '':  # sometiems empty string passed
                            print(f'Synset with empty translation: {synset}, {language}')
                            continue
                        new_translations.add(word)
                        lemmatized_word = simplemma.lemmatize(word, lang=language_code)
                        new_translations.add(lemmatized_word)
                    lemmatized_translations[language][synset] = new_translations
            else:
                lemmatized_translations[language] = all_translations[language]
            count += 1
            print(f'{count}/{len(lemmatized_translations)}')
        return lemmatized_translations

    def clean_bn_translations(self):
        """splits multi-word phrases and applies lemmatization"""
        with open('generated_files/all_wn_translations.pkl', 'rb') as f:
            loaded_bn_all_translations = pickle.load(f)

            bn_all_translations = {}
            for language, translations in loaded_bn_all_translations.items():
                bn_all_translations[language.lower()] = {synset_str: {word.lower() for word in translated_words} for
                                                         synset_str, translated_words in translations.items()}

            bn_all_translations = self.split_multiword_phrases(bn_all_translations)
            bn_all_translations = self.lemmatize_translations(bn_all_translations)
            return bn_all_translations

    def load_english_NRC(self):
        """loads English NRC lexicon (emolex)"""
        nrc_file = 'emolex/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt'
        with open(nrc_file, 'r') as file:
            nrc_lines = file.readlines()

        for line in nrc_lines:
            word, emotion, score = re.split(r'\t+', line.strip())
            if emotion in ['positive', 'negative']:
                continue
            if word not in self.word_lexicon:
                self.word_lexicon[word] = {}
            self.word_lexicon[word][emotion] = int(score)

        for line in nrc_lines:
            word, emotion, score = re.split(r'\t+', line.strip())
            if word not in self.word_lexicon_sentiment:
                self.word_lexicon_sentiment[word] = {}
            self.word_lexicon_sentiment[word][emotion] = int(score)

    def load_lexcions(self):
        """loads multilingual emolex lexicons that have gone through various parsing procedures"""
        with open('generated_files/lemmatized_multilingual_lexicons.pkl', 'wb') as f:
            self.lemmatized_multilingual_lexicons = pickle.load(f)
        with open('generated_files/normalized_lemmatized_multilingual_lexicons.pkl', 'wb') as f:
            self.normalized_lemmatized_multilingual_lexicons = pickle.load(f)
        with open('generated_files/non_sentiment_normalized_lemmatized_multilingual_lexicons.pkl', 'wb') as f:
            self.non_sentiment_normalized_lemmatized_multilingual_lexicons = pickle.load(f)

    def get_shared_emotions(self, synset, translated_word, language, sentiment=False, normalized=False,
                            only_sentiment=False):
        """gets the common emotions of an English synset and multilingual lemma"""
        # load proper lexicon
        if normalized:
            langauge_lexicon = self.normalized_lemmatized_multilingual_lexicons[language]
        else:
            langauge_lexicon = self.lemmatized_multilingual_lexicons[language]

        # get proper lemmas
        lemmas = {lemma.name().lower() for lemma in synset.lemmas()}

        if lemmas & set(self.word_lexicon.keys()):
            lemmas_emotions = set()
            if sentiment:
                for lemma in lemmas:
                    if lemma in self.word_lexicon_sentiment.keys():
                        for emotion, score in self.word_lexicon_sentiment[lemma].items():
                            if score:
                                lemmas_emotions.add(emotion)
                                # lemmas_emotions |= set(sentiment_neigbours[emotion])
            else:
                for lemma in lemmas:
                    if lemma in self.word_lexicon.keys():
                        for emotion, score in self.word_lexicon[lemma].items():
                            if score:
                                lemmas_emotions.add(emotion)
                                # lemmas_emotions |= set(sentiment_neigbours[emotion])

        else:  # if we have no information on any of the english lemmas, we assume them to be emotional
            if sentiment or only_sentiment:
                lemmas_emotions = set(self.sentiments)
            else:
                lemmas_emotions = set(self.emotions)

        translated_word_emotions = {emotion for emotion, score in langauge_lexicon[translated_word].items() if score}
        if not sentiment and not only_sentiment:
            translated_word_emotions -= {'negative', 'positive'}
        if only_sentiment:
            lemmas_emotions -= set(self.emotions)
            translated_word_emotions -= set(self.emotions)

        return lemmas_emotions & translated_word_emotions

    def get_translated_words_and_lexicon(self, language, all_translations, synset, normalized=False, sentiment=False):
        """gets the multilingual lemmas associated with synset in a specific language and the translated version of
        emolex in that language"""
        if language == 'english':
            if normalized:
                translated_words = {unidecode(lemma.name()).strip() for lemma in synset.lemmas()}
                if sentiment:
                    langauge_lexicon = self.normalized_lemmatized_multilingual_lexicons[language]
                else:
                    langauge_lexicon = self.non_sentiment_normalized_lemmatized_multilingual_lexicons[language]

            else:
                translated_words = {lemma.name() for lemma in synset.lemmas()}
                langauge_lexicon = self.lemmatized_multilingual_lexicons[language]

        else:
            if normalized:
                translated_words = {unidecode(word).strip() for word in all_translations[language][synset.name()]}
                langauge_lexicon = self.normalized_lemmatized_multilingual_lexicons[language]
            else:
                translated_words = set(all_translations[language][synset.name()])
                langauge_lexicon = self.lemmatized_multilingual_lexicons[language]

        translated_words = {word.lower() for word in translated_words}

        return translated_words, langauge_lexicon

    def check_if_emotional_in_langauge(self, synset, language, all_translations, normalized=False,
                                       emotion_specific=False, sentiment=False, only_sentiment=False):
        """checks if a synset is associated with emotions/sentiments in a specific language"""
        if only_sentiment:
            sentiment = True

        translated_words, langauge_lexicon = self.get_translated_words_and_lexicon(language, all_translations, synset,
                                                                                   normalized=normalized,
                                                                                   sentiment=sentiment)
        emotional_translations = set(translated_words) & set(langauge_lexicon)

        if emotion_specific:
            for translated_word in emotional_translations:
                shared_emotions = self.get_shared_emotions(synset, translated_word, language, sentiment=sentiment,
                                                           normalized=normalized,
                                                           only_sentiment=only_sentiment)
                if shared_emotions:
                    return True
        else:
            if emotional_translations:
                return True
        return False

    def get_synset_languages_count(self, all_translations, normalized=False, emotion_specific=False, sentiment=False,
                                   only_sentiment=False):
        """determines how many languages each synset is emotion/polar in"""
        translations_synset_languages_count = {wordnet.synset(synset_str): 0 for synset_str in
                                               list(all_translations.values())[0]}
        for i, synset in enumerate(translations_synset_languages_count):
            for language in all_translations:
                is_emotional_in_language = self.check_if_emotional_in_langauge(synset, language, all_translations,
                                                                               normalized=normalized,
                                                                               emotion_specific=emotion_specific,
                                                                               sentiment=sentiment,
                                                                               only_sentiment=only_sentiment)
                if is_emotional_in_language:
                    translations_synset_languages_count[synset] += 1
            max_synset_languages_count = len(all_translations)
            translations_synset_languages_count[synset] = min(translations_synset_languages_count[synset],
                                                              max_synset_languages_count)

        return translations_synset_languages_count

    def get_language_emotions(self, synset, language, all_translations, normalized=False, sentiment=False,
                              only_sentiment=False):
        """returns the emotions that are associated with a synset in a specific language"""
        if only_sentiment:
            sentiment = True
        translated_words, langauge_lexicon = self.get_translated_words_and_lexicon(language, all_translations, synset,
                                                                                   normalized=normalized,
                                                                                   sentiment=sentiment)

        language_emotions = set()
        for word in set(translated_words) & set(langauge_lexicon.keys()):
            language_emotions |= {emotion for emotion, score in langauge_lexicon[word].items() if score}

        if not sentiment:
            language_emotions -= {'positive', 'negative'}
        elif only_sentiment:
            language_emotions -= set(self.emotions)
        return language_emotions

    def break_emotion_tie(self, synset, possible_emotions):
        """decides which single emotion (out of a given list of possible emotions) has the most similar gloss to the
        given synset"""
        emotions_to_synset = {'anger': wordnet.synset('anger.n.01'),
                              'anticipation': wordnet.synset('anticipation.n.02'),
                              'disgust': wordnet.synset('disgust.n.01'),
                              'fear': wordnet.synset('fear.n.01'),
                              'joy': wordnet.synset('joy.n.01'),
                              'sadness': wordnet.synset('sadness.n.01'),
                              'surprise': wordnet.synset('surprise.n.01'),
                              'trust': wordnet.synset('trust.n.03'),
                              'positive': wordnet.synset('positive.a.01'),
                              'negative': wordnet.synset('negative.a.01'),
                              }

        emotion_sim = {}
        synset_sentence = nlp(f"{', '.join([lemma.name() for lemma in synset.lemmas()])} is {synset.definition()}")
        for emotion in possible_emotions:
            emotion_synset = emotions_to_synset[emotion]
            emotion_sentence = nlp(
                f"{', '.join([lemma.name() for lemma in emotion_synset.lemmas()])} is {emotion_synset.definition()}")
            emotion_sim[emotion] = synset_sentence.similarity(emotion_sentence)

        return max(emotion_sim, key=emotion_sim.get)

    def extend_lexicon(self, core_classifications, task='emotion identification'):
        """extends the core of lexicon using WordNet relations"""
        if task not in ['emotion identification', 'sentiment identification', 'emotion detection']:
            raise Exception('Invalid task')

        if 'identification' in task:
            if type(core_classifications) != dict:
                raise Exception('Identification requires dict as an input')
            new_classifications = {}
            for synset, emotion in core_classifications.items():
                related_synsets = set(synset.similar_tos())
                related_synsets |= set(synset.attributes())
                related_synsets |= set(synset.also_sees())
                related_synsets |= {related_form.synset() for lemma in synset.lemmas() for related_form in
                                    lemma.derivationally_related_forms()}
                related_synsets |= {pertainym.synset() for lemma in synset.lemmas() for pertainym in lemma.pertainyms()}

                for related_synset in related_synsets:
                    if related_synset not in core_classifications:
                        if related_synset not in new_classifications:
                            new_classifications[related_synset] = set()
                        new_classifications[related_synset].add(emotion)

            if task == 'emotion identification':
                for synset, emotions in new_classifications.items():
                    if len(emotions) > 1:
                        emotion = self.break_emotion_tie(synset, emotions)
                    else:
                        emotion = list(emotions)[0]
                    new_classifications[synset] = emotion
                extended_lexicon = new_classifications | core_classifications

            elif task == 'sentiment identification':
                for synset, emotions in new_classifications.items():
                    new_classifications[synset] = list(emotions)[0] if len(emotions) == 1 else 'positive_negative'
                extended_lexicon = new_classifications | core_classifications

        elif task == 'emotion detection':
            if type(core_classifications) != set:
                raise Exception('Detection requires set as an input')
            new_classifications = set()
            for synset in core_classifications:
                new_classifications |= set(synset.similar_tos())
                new_classifications |= set(synset.attributes())
                new_classifications |= set(synset.also_sees())
                new_classifications |= {related_form.synset() for lemma in synset.lemmas() for related_form in
                                        lemma.derivationally_related_forms()}
                new_classifications |= {pertainym.synset() for lemma in synset.lemmas() for pertainym in
                                        lemma.pertainyms()}
                new_classifications |= {antonym.synset() for lemma in synset.lemmas() for antonym in lemma.antonyms()}
            extended_lexicon = new_classifications | core_classifications
        return extended_lexicon

    def find_sentiment_classifications(self, all_translations, synsets, normalized=False):
        """determines the sentiment (if any) associated with all sysnets"""
        seed_set_annotations = {}

        for synset in synsets:
            english_word_emotions = self.get_language_emotions(synset, 'english', all_translations,
                                                               normalized=normalized,
                                                               sentiment=True, only_sentiment=True)
            if not english_word_emotions:
                english_word_emotions = {'positive', 'negative'}
            if english_word_emotions == {'positive', 'negative'}:
                english_word_emotions.add('positive_negative')

            emotions_count = {emotion: 0 for emotion in english_word_emotions}
            for language in all_translations:
                language_emotions = self.get_language_emotions(synset, language, all_translations,
                                                               normalized=normalized,
                                                               sentiment=True, only_sentiment=True)
                language_emotions &= english_word_emotions

                if language_emotions == {'positive'}:
                    emotions_count['positive'] += 1
                elif language_emotions == {'negative'}:
                    emotions_count['negative'] += 1
                elif len(language_emotions) > 1:
                    emotions_count['positive_negative'] += 1

            max_count = max(emotions_count.values())
            labels = [emotion for emotion, count in emotions_count.items() if count == max_count]

            seed_set_annotations[synset] = labels[0] if len(labels) == 1 else 'positive_negative'
        return seed_set_annotations

    def find_emotion_classifications(self, all_translations, synsets, normalized=False, sentiment=False):
        """determines the emotion (if any) associated with all sysnets"""
        seed_set_annotations = {}
        count = 0

        for synset in synsets:
            english_word_emotions = self.get_language_emotions(synset, 'english', all_translations,
                                                               normalized=normalized,
                                                               sentiment=sentiment)
            if not english_word_emotions:
                if sentiment:
                    english_word_emotions = set(self.sentiments)
                else:
                    english_word_emotions = set(self.emotions)

            emotions_count = {emotion: 0 for emotion in english_word_emotions}
            for language in all_translations:
                language_emotions = self.get_language_emotions(synset, language, all_translations,
                                                               normalized=normalized,
                                                               sentiment=sentiment)
                language_emotions &= english_word_emotions

                for emotion in language_emotions:
                    emotions_count[emotion] += 1

            max_count = max(emotions_count.values())
            labels = [emotion for emotion, count in emotions_count.items() if count == max_count]
            if len(labels) > 1:
                seed_set_annotations[synset] = self.break_emotion_tie(synset, labels)
            else:
                seed_set_annotations[synset] = labels[0]
        return seed_set_annotations

    def create_sentiment_lexicon(self):
        """creates sentiment lexicon"""
        # determine confidence of each synset
        bn_all_synsets_lang_count_sentiment = self.get_synset_languages_count(self.bn_all_translations,
                                                                              emotion_specific=True,
                                                                              sentiment=False,
                                                                              normalized=True,
                                                                              only_sentiment=True)

        # determines the ratio of sentimental to non-sentimental languages for sysnets
        bn_all_synsets_lang_count_sentiment_frac = {}
        for synset, count in bn_all_synsets_lang_count_sentiment.items():
            translation_count = 0
            for translations in self.bn_all_translations.values():
                if translations[synset.name()]:
                    translation_count += 1
            bn_all_synsets_lang_count_sentiment_frac[synset] = count / max(translation_count, 0.1)

        # saves all synsets that are predicted to be sentimental with high-confidence
        high_confidence_sentiment_predictions = set()
        for synset in self.all_wn_synsets:
            if bn_all_synsets_lang_count_sentiment_frac[synset] >= 0.7:
                high_confidence_sentiment_predictions.add(synset)

        # labels the core sentimental synsets
        core_sentiment_classifications = self.find_sentiment_classifications(self.bn_all_translations,
                                                                             high_confidence_sentiment_predictions,
                                                                             normalized=True)

        # extends the sentimental core
        self.extended_sentiment_lexicon = self.extend_lexicon(core_sentiment_classifications,
                                                              'sentiment identification')

    def create_emotion_lexicon(self):
        """creates emotion lexicon"""
        # determine confidence of each synset
        bn_all_synsets_lang_count = self.get_synset_languages_count(self.bn_all_translations,
                                                                    emotion_specific=True,
                                                                    sentiment=False,
                                                                    normalized=True)

        # determines the ratio of emotional to non-emotional languages for sysnets
        bn_all_synsets_lang_count_frac = {}
        for synset, count in bn_all_synsets_lang_count.items():
            translation_count = 0
            for translations in self.bn_all_translations.values():
                if translations[synset.name()]:
                    translation_count += 1

            bn_all_synsets_lang_count_frac[synset] = count / max(translation_count, 0.1)

        # saves all synsets that are predicted to be emotional with high-confidence
        high_confidence_predictions = set()
        for synset in self.all_wn_synsets:
            if bn_all_synsets_lang_count_frac[synset] >= 0.7:
                high_confidence_predictions.add(synset)

        # labels the core emotional synsets
        core_emotion_classifications = self.find_emotion_classifications(self.bn_all_translations,
                                                                         high_confidence_predictions,
                                                                         normalized=True)

        # extends the emotional core
        self.extended_lexicon = self.extend_lexicon(core_emotion_classifications, task='emotion identification')

    def save_lexicon(self):
        """saves the lexicon to an xml file"""
        lexicon_info = {'emotion': [], 'sentiment': []}

        for task, info in lexicon_info.items():
            if task == 'emotion':
                lexicon = self.extended_lexicon
            else:
                lexicon = self.extended_sentiment_lexicon

            for synset, emotion in lexicon.items():
                if emotion in self.sentiments:
                    offset = synset.offset()
                    pos = synset.pos()

                    lexicon_info[task].append({
                        '@id': f'wn:{offset:0>8}{pos}',
                        f'@{task}': emotion,
                    })

        xml_data = {'SentiSynset': {'PolarityLexicon': {'Synset': lexicon_info['sentiment']},
                                    'EmotionLexicon': {'Synset': lexicon_info['emotion']},
                                    }
                    }

        with open('generated_files/sentisynset_lexicon.xml', 'w') as f:
            f.write(xmltodict.unparse(xml_data, pretty=True))


def main():
    """creates emotion and sentiment lexicon"""
    lex = CreateLexicon()
    lex.create_sentiment_lexicon()
    lex.create_emotion_lexicon()
    lex.save_lexicon()


if __name__ == '__main__':
    main()

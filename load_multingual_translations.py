import re
import langcodes
from unidecode import unidecode
import simplemma
import pickle

class LoadMultiLingualLexicons:
    def __init__(self):
        # load english NRC-lexicon
        self.word_lexicon = {}
        self.word_lexicon_sentiment = {}
        self.load_english_NRC()

        # define emotions/polarities
        self.emotions = ['anger', 'anticipation', 'disgust', 'fear', 'joy', 'sadness', 'surprise', 'trust']
        self.sentiments = self.emotions + ['negative', 'positive']
        self.sentiments.sort()

        with open('select_languages.pkl', 'rb') as f:
            self.select_languages = pickle.load(f)

    def load_english_NRC(self):
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

    def get_emotional_multilingual_lexicon(self, language):
        language = language.lower().capitalize()
        if language == 'English':
            multilingual_lexicon = self.word_lexicon_sentiment
        else:
            NRC_language = 'Chinese-Simplified' if language == 'Chinese' else language

            with open(f'emolex/OneFilePerLanguage/{NRC_language}-NRC-EmoLex.txt', 'r', encoding="utf8") as f:
                lines = f.readlines()
            content_order = lines[0].strip().split('\t')
            lines = [line.strip() for line in lines[1:]]

            multilingual_lexicon = {}
            for line in lines:
                content = {content_order[i]: line.split('\t')[i] for i in range(len(content_order))}
                emotional_scores = {emotion: int(content[emotion]) for emotion in self.sentiments}
                translated_word = content[f'{NRC_language} Word']

                if sum(emotional_scores.values()) > 0:
                    multilingual_lexicon[translated_word] = emotional_scores
        return multilingual_lexicon

    def get_lang_code(self, language):
        if language == 'Norwegian':
            return 'nb'
        else:
            return langcodes.find(language).language

    def load_multilingual_lexicon(self, language, lemmatize=False, normalized=False, sentiment=True,
                                  emotion_specific=False):
        language = language.lower().capitalize()
        lang_code = self.get_lang_code(language)
        full_lexicon = self.get_emotional_multilingual_lexicon(language)

        multilingual_lexicon = {}
        for translated_word, emotional_scores in full_lexicon.items():
            words_to_be_added = [translated_word]  # either just word from NRC, or also includes lemmatizatin

            if lemmatize and language not in ['Chinese', 'Korean']:  # languages not support by simplelemma:
                words_to_be_added.append(simplemma.lemmatize(translated_word, lang=lang_code))

            if normalized:
                words_to_be_added = [unidecode(word).strip() for word in words_to_be_added]

            for word in words_to_be_added:
                word = word.lower().strip()
                if word in multilingual_lexicon:
                    old_emotions = {emotion for emotion, score in multilingual_lexicon[word].items() if score}
                    new_emotions = {emotion for emotion, score in full_lexicon[translated_word].items() if score}
                    joined_emotions = old_emotions | new_emotions
                    multilingual_lexicon[word] = {emotion: 1 if emotion in joined_emotions else 0 for emotion in
                                                  self.sentiments}
                else:
                    multilingual_lexicon[word] = full_lexicon[translated_word]

                if not sentiment:
                    multilingual_lexicon[word].pop('negative')
                    multilingual_lexicon[word].pop('positive')

        # emotion_specific
        if not emotion_specific:
            multilingual_lexicon = set(multilingual_lexicon.keys())

        return multilingual_lexicon

    def load_and_clean_multingual_lexicons(self):
        lemmatized_multilingual_lexicons = {}
        for language in [lang.lower() for lang in self.select_languages.keys()] + ['english']:
            lemmatized_multilingual_lexicons[language] = self.load_multilingual_lexicon(language,
                                                                                                         lemmatize=True,
                                                                                                         sentiment=True,
                                                                                                         emotion_specific=True)

        normalized_lemmatized_multilingual_lexicons = {}
        for language, emotional_words in lemmatized_multilingual_lexicons.items():
            normalized_lemmatized_multilingual_lexicons[language] = {unidecode(word).strip().lower(): scores for
                                                                     word, scores in emotional_words.items()}

        non_sentiment_normalized_lemmatized_multilingual_lexicons = {}
        for language, lexicon in normalized_lemmatized_multilingual_lexicons.items():
            non_sentiment_normalized_lemmatized_multilingual_lexicons[language] = {}
            for word, emotions in lexicon.items():
                non_sentiment_normalized_lemmatized_multilingual_lexicons[language][word] = {emotion: score for
                                                                                             emotion, score in
                                                                                             emotions.items() if
                                                                                             emotion not in [
                                                                                                 'negative',
                                                                                                 'positive']}

        with open('generated_files/lemmatized_multilingual_lexicons.pkl', 'wb') as f:
            pickle.dump(lemmatized_multilingual_lexicons, f)
        with open('generated_files/normalized_lemmatized_multilingual_lexicons.pkl', 'wb') as f:
            pickle.dump(normalized_lemmatized_multilingual_lexicons, f)
        with open('generated_files/non_sentiment_normalized_lemmatized_multilingual_lexicons.pkl', 'wb') as f:
            pickle.dump(non_sentiment_normalized_lemmatized_multilingual_lexicons, f)


def main():
    lex = LoadMultiLingualLexicons()
    lex.load_and_clean_multingual_lexicons()


if __name__ == '__main__':
    main()

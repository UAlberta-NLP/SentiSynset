from translations import GetBabelNetTranslations
from load_multingual_translations import LoadMultiLingualLexicons
from create_lexicons import CreateLexicon


def main():
    """runs all processes necessary to build lexicon"""
    # get bn_translations
    bn_translations = GetBabelNetTranslations()
    bn_translations.get_translations()

    # get multingual emolex lexicons
    lex = LoadMultiLingualLexicons()
    lex.load_and_clean_multingual_lexicons()

    # create sentisynset
    lex = CreateLexicon()
    lex.create_sentiment_lexicon()
    lex.create_emotion_lexicon()
    lex.save_lexicon()


if __name__ == "__main__":
    main()

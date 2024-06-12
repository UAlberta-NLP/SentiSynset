import babelnet as bn
import nltk
import pickle
from babelnet.language import Language
from babelnet.resources import WordNetSynsetID
from babelnet.resources import BabelSynsetID
from nltk.corpus import wordnet


class GetBabelNetTranslations:
    def __init__(self):
        self.save_file = 'generated_files/all_wn_translations.pkl'

    def get_all_wn_offsets(self):
        """gets and formats the offsets for all WordNet synsets"""
        all_offsets = [f"{synset.offset():08}{'a' if synset.pos() == 's' else synset.pos()}" for synset in
                       wordnet.all_synsets()]
        return all_offsets

    def get_translations(self):
        """gets the translations of all WordNet synsets from BabelNet"""
        offsets = self.get_all_wn_offsets()

        with open('select_languages.pkl', 'rb') as f:
            select_languages = pickle.load(f)

        langauge_names = [lang for lang in select_languages.keys()]
        to_languages = [Language.from_iso(iso_code) for iso_code in select_languages.values()]

        synsets_translation = {language: {} for language in langauge_names}

        for offset_count, offset in enumerate(offsets[len(synsets_translation[langauge_names[0]]):]):
            wn_synset = wordnet.synset_from_pos_and_offset(offset[-1], int(offset[:8]))
            bn_id = str(bn.get_synset(WordNetSynsetID(f'wn:{offset}')).id)

            # bn can only get translate synsets to 3 languages at a time
            for i in range(((len(langauge_names) - 1) // 3) + 1):
                three_langs_names = langauge_names[i * 3: (i + 1) * 3]
                three_to_langs = to_languages[i * 3: (i + 1) * 3]

                bn_synset = bn.get_synset(BabelSynsetID(bn_id), to_langs=three_to_langs)

                for j, language in enumerate(three_to_langs):
                    lang_translations = [str(lemma) for lemma in bn_synset.lemmas(language)]
                    synsets_translation[three_langs_names[j]][wn_synset.name()] = lang_translations

            if offset_count % 5000 == 0:
                with open(self.save_file, 'wb') as f:
                    pickle.dump(synsets_translation, f)
                print(f'{len(synsets_translation[langauge_names[0]])}/{len(offsets)}')

        # save once more at the very end
        with open(self.save_file, 'wb') as f:
            pickle.dump(synsets_translation, f)


def main():
    """gets all BabelNet translations"""
    bn_translations = GetBabelNetTranslations()
    bn_translations.get_translations()


if __name__ == '__main__':
    main()

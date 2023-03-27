import re
from operator import itemgetter
from random import randint  # только для тестирования
from time import time  # только для тестирования

import pandas as pd  # только для тестирования
import pymorphy2
from nltk import collocations
from nltk.corpus import stopwords
from razdel import sentenize, tokenize

analyzer = pymorphy2.MorphAnalyzer()


bigram_rules = {
    1: {"pos_1": {"ADJF", "ADJS", "PRTF", "PRTS"}, "pos_2": {"NOUN", "LATN", "UNKN", None}},
    2: {"pos_1": {"NOUN", "LATN", "UNKN", None}, "pos_2": {"NOUN", "LATN", "UNKN", None}},
}

trigram_rules = {
    1: {
        "pos_1": {"NOUN", "LATN", "UNKN", None},
        "pos_2": {"NOUN", "LATN", "UNKN", None},
        "pos_3": {"NOUN", "LATN", "UNKN", None},
    },
    2: {
        "pos_1": {"NOUN", "LATN", "UNKN", None},
        "pos_2": {"ADJF", "ADJS", "PRTF", "PRTS"},
        "pos_3": {"NOUN", "LATN", "UNKN", None},
    },
    3: {
        "pos_1": {"ADJF", "ADJS", "PRTF", "PRTS"},
        "pos_2": {"NOUN", "LATN", "UNKN", None},
        "pos_3": {"NOUN", "LATN", "UNKN", None},
    },
    4: {
        "pos_1": {"ADJF", "ADJS", "PRTF", "PRTS"},
        "pos_2": {"ADJF", "ADJS", "PRTF", "PRTS"},
        "pos_3": {"NOUN", "LATN", "UNKN", None},
    },
}


def rpd_to_wordlist(text, remove_stopwords=False, morph=True):
    """Предобработка списка слов из содержания"""
    text = re.sub(r"[^a-zA-Zа-яА-ЯёЁ\'\-\#\+]", " ", text)
    words = [_.text for _ in tokenize(text.lower())]
    if not words:
        return None
    if morph:
        words = [analyzer.parse(word)[0].normal_form for word in words]
    if remove_stopwords:
        stops = stopwords.words("russian")
        words = [w for w in words if w not in stops]
    return words


def rpd_to_sentences(text, remove_stopwords=False, morph=True):
    """Выделение предложений и обработка слов внутри них.
    Если в предложении получилось 2 и 3 слова – считаем результатом без доп.обработок.
    Предложения из одного слова игнорируем.
    Предложения из 4+ слов разбиваются на токены и обрабатываются дальше"""
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    text = re.sub(r"([0-9]+).([0-9]+)", "", text)
    text = re.sub(r"([0-9]+).(\s+)", "", text)
    text = re.sub(r"[:\.,»«)(\n\t–]", ". ", text)
    text = re.sub(r"([а-яa-zё]+)([А-ЯA-ZЁ]+)", r"\1. \2", text)
    text = re.sub(r"([0-9]+)(\s+)([0-9]+)", "", text)
    text = re.sub(r"\.(\s+)([a-zA-Zа-яА-ЯёЁ0-9\-])", lambda x: x.group(0).upper(), text)
    text = text.replace("\xa0", " ")
    text = text.replace(".", ". ").strip()
    raw_sentences = [_.text for _ in list(sentenize(text))]

    valid_phrases, sentences = [], []
    for sent in raw_sentences:
        if len(sent) > 0:
            tokens = rpd_to_wordlist(sent, remove_stopwords, morph)
            if 2 <= len(tokens) <= 3 and "и" not in tokens:
                key_phrase = re.sub(r"[^a-zA-Zа-яА-ЯёЁ0-9\'\-\#\+]", " ", sent).strip(" .").lower()
                valid_phrases.append((key_phrase, tuple(tokens)))
            elif len(tokens) > 3:
                sentences.append(tokens)

    valid_phrases = [[data, valid_phrases.count(data)] for data in valid_phrases]

    return valid_phrases, sentences


def get_bigrams_trigrams(sentences):
    """Наивный поиск словосочетаний внутри длинного предложения.
    Берутся только биграммы и триграммы"""
    bigram_measures = collocations.BigramAssocMeasures()
    trigram_measures = collocations.TrigramAssocMeasures()
    bigrams_, trigrams_ = [], []
    for sent in sentences:
        finder_bi = collocations.BigramCollocationFinder.from_words(sent)
        finder_tri = collocations.TrigramCollocationFinder.from_words(sent)
        bigrams_.extend([phrase for phrase, _ in finder_bi.score_ngrams(bigram_measures.likelihood_ratio)])
        trigrams_.extend([phrase for phrase, _ in finder_tri.score_ngrams(trigram_measures.likelihood_ratio)])
    return bigrams_, trigrams_


def score_bigrams_trigrams(text, valid_tokens):
    """Подсчитывается частота появления словосочетания внутри содержания.
    Если лемматизированные токены совпадают с токенами уже выбранных результатов, кандидат игнорируется.
    Скоры (частоты) суммируются.
    Случай, когда биграмма входит в триграмму не рассматривается, т.к. часто срезается много полезных слов"""

    bigrams_, trigrams_ = get_bigrams_trigrams(text)
    bigrams_score = [(bigram, bigrams_.count(bigram)) for bigram in bigrams_]
    trigrams_score = [(trigram, trigrams_.count(trigram)) for trigram in trigrams_]

    to_remove_2 = set()
    to_remove_3 = set()

    for i, token in enumerate(valid_tokens):
        for bigram, n in bigrams_score:
            if set(bigram) < set(token[0][1]):
                valid_tokens[i][1] += n
                to_remove_2.add((bigram, n))
        for trigram, n in trigrams_score:
            if set(token[0][1]) == set(trigram):
                valid_tokens[i][1] += n
                to_remove_3.add((trigram, n))

    valid_tokens = {(phrase[0], score) for phrase, score in valid_tokens}

    for token in to_remove_2:
        bigrams_score.remove(token)

    for token in to_remove_3:
        trigrams_score.remove(token)

    return set(bigrams_score), set(trigrams_score), valid_tokens


def get_keyphrases(sent, valid_tokens=None, n_best=None):
    """Получение фразы из набора биграмм и триграмм: склонение по правилам.
    Возможны неточности. Обычно из-за pymorphy.
    n_best: количество фраз, которое нужно получить из содержания. Лучше брать побольше, чтобы оставался выбор"""

    if valid_tokens is None:
        valid_tokens = []
    bigrams_, trigrams_, keyphrases = score_bigrams_trigrams(sent, valid_tokens)

    if bigrams_:
        tagged_phrases = [
            (
                words,
                [
                    analyzer.parse(words[0])[0].tag.POS if words[0] != "данные" else "NOUN",
                    analyzer.parse(words[1])[0].tag.POS if words[1] != "данные" else "NOUN",
                ],
                score,
            )
            for words, score in bigrams_
        ]
        for words, tags, score in tagged_phrases:
            try:
                if tags[0] in bigram_rules[1]["pos_1"] and tags[1] in bigram_rules[1]["pos_2"]:
                    gender = analyzer.parse(words[1])[0].tag.gender
                    attribute = analyzer.parse(words[0])[0].inflect({gender, "nomn"})
                    if attribute:
                        keyphrases.add((" ".join([attribute.word, words[1]]), score))

                elif tags[0] in bigram_rules[2]["pos_1"] and tags[1] in bigram_rules[2]["pos_2"]:
                    attribute = analyzer.parse(words[1])[0].inflect({"gent"})
                    if attribute:
                        keyphrases.add((" ".join([words[0], attribute.word]), score))
                    else:
                        keyphrases.add((" ".join(words), score))
            except ValueError:
                continue

    if trigrams_:
        tagged_phrases = [
            (
                words,
                [
                    analyzer.parse(words[0])[0].tag.POS if words[0] != "данные" else "NOUN",
                    analyzer.parse(words[1])[0].tag.POS if words[1] != "данные" else "NOUN",
                    analyzer.parse(words[2])[0].tag.POS if words[2] != "данные" else "NOUN",
                ],
                score,
            )
            for words, score in trigrams_
        ]
        for words, tags, score in tagged_phrases:
            try:
                if (
                    tags[0] in trigram_rules[1]["pos_1"]
                    and tags[1] in trigram_rules[1]["pos_2"]
                    and tags[2] in trigram_rules[1]["pos_3"]
                ):
                    attribute_noun1 = analyzer.parse(words[1])[0].inflect({"gent"})
                    attribute_noun2 = analyzer.parse(words[2])[0].inflect({"gent"})
                    if not attribute_noun1 and attribute_noun2:
                        keyphrases.add((" ".join([words[0], words[1], attribute_noun2.word]), score))
                    elif not attribute_noun2 and attribute_noun1:
                        keyphrases.add((" ".join([words[0], attribute_noun1.word, words[2]]), score))
                    elif not (attribute_noun1 and attribute_noun2):
                        keyphrases.add((" ".join(words), score))
                    else:
                        keyphrases.add((" ".join([words[0], attribute_noun1.word, attribute_noun2.word]), score))

                elif (
                    tags[0] in trigram_rules[2]["pos_1"]
                    and tags[1] in trigram_rules[2]["pos_2"]
                    and tags[2] in trigram_rules[2]["pos_3"]
                ):
                    gender = analyzer.parse(words[2])[0].tag.gender
                    attribute_adj = analyzer.parse(words[1])[0].inflect({"gent", gender})
                    attribute_noun = analyzer.parse(words[2])[0].inflect({"gent"})
                    if gender and attribute_adj and attribute_noun:
                        keyphrases.add((" ".join([words[0], attribute_adj.word, attribute_noun.word]), score))
                elif (
                    tags[0] in trigram_rules[3]["pos_1"]
                    and tags[1] in trigram_rules[3]["pos_2"]
                    and tags[2] in trigram_rules[3]["pos_3"]
                ):
                    gender = analyzer.parse(words[1])[0].tag.gender
                    attribute_adj = analyzer.parse(words[0])[0].inflect({gender, "nomn"})
                    attribute_noun = analyzer.parse(words[2])[0].inflect({"gent"})
                    if gender and attribute_adj and attribute_noun:
                        keyphrases.add((" ".join([attribute_adj.word, words[1], attribute_noun.word]), score))
                elif (
                    tags[0] in trigram_rules[4]["pos_1"]
                    and tags[1] in trigram_rules[4]["pos_2"]
                    and tags[2] in trigram_rules[4]["pos_3"]
                ):
                    gender = analyzer.parse(words[2])[0].tag.gender
                    attribute_adj1 = analyzer.parse(words[0])[0].inflect({gender, "nomn"})
                    attribute_adj2 = analyzer.parse(words[1])[0].inflect({gender, "nomn"})
                    if gender and attribute_adj1 and attribute_adj2:
                        keyphrases.add((" ".join([attribute_adj1.word, attribute_adj2.word, words[2]]), score))
            except ValueError:
                continue

    res = sorted(keyphrases, reverse=True, key=itemgetter(1))
    if n_best is not None:
        return [phrase for phrase, _ in res][:n_best]
    return [phrase for phrase, _ in res]


def simple_outcomes_extraction(text, remove_stopwords=False, n_best=None):
    """Извлечение ключевых слов из текста."""
    phrases, sents = rpd_to_sentences(text, remove_stopwords)
    return get_keyphrases(sents, phrases, n_best)


if __name__ == "__main__":
    # В среднем обработка одной дисциплины занимает 0.05 секунды

    df = pd.read_excel("rpd_27012022.xlsx")
    N = randint(1, df.shape[0] - 1)

    print(df.loc[N].title)
    rpd = df.loc[N].text
    print(rpd)

    start = time()

    print(simple_outcomes_extraction(rpd, n_best=10))

    print(round(time() - start, 3))

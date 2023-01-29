import utils
import re
import pprint
import spacy

ACTION_VERBS_FILE = './data/action_verbs.txt'

spacy_nlp = spacy.load('en_core_web_sm')
action_verbs = utils.stem(" ".join(open(ACTION_VERBS_FILE).readlines())).split()


# Collect basic statistics about text.
class StatCollector:
    def __init__(self, s):
        self.txt = s

    def num_words(self):
        # tokenize by whitespace
        return len(utils.split_into_words(self.txt, use_nltk=False))

    def avg_sentence_length(self):
        # tokenize by nltk
        sentences = utils.split_into_sentences(self.txt)
        avg_len = sum([len(utils.split_into_words(s, use_nltk=True)) for s in sentences]) / len(sentences)
        return round(avg_len)

    def num_paragraphs(self):
        merge_newlines = re.sub(r'(\n)+', r'\n', self.txt.strip())
        paras = merge_newlines.split(sep='\n')
        return len(paras)

    def get_stat(self):
        return {
            "word count": self.num_words(),
            "avg sentence length": self.avg_sentence_length(),
            "number of paragraphs": self.num_paragraphs(),
        }


# Analyze text for information other than basic statistics.
class TextAnalyzer:
    def __init__(self, s):
        self.txt = s

    # Count frequencies of target words
    def word_count(self, targets):
        # Stem before count, to handle cases and punctuations correctly
        stemmed = utils.stem(self.txt)
        words = stemmed.split(' ')

        res = {}
        for t in targets:
            res[t] = words.count(utils.stem(t))

        return res

    # Count frequencies of target phrases
    def phrase_count(self, targets):
        # Stem before count, to handle cases and punctuations correctly
        stemmed = utils.stem(self.txt)

        res = {}
        for t in targets:
            stemmed_t = utils.stem(t)

            if len(stemmed_t.split()) == 0:
                print(f"Target phrase '{t}' contains 0 words?")

            if len(stemmed_t.split()) == 1 and len(t) < 5:
                print(f"Target phrase '{t} is a single word, and is short. Watch out for false positive!")

            # This is technically not correct. String count method does simple
            # string matching instead of token matching. For example,
            # 'xxxsuch asxxx' would be counted as a match with 'such as'.
            # But this rarely happens if the phrase contains at least two words
            # or when the word is long enough.
            res[t] = stemmed.count(stemmed_t)

        return res

    # Count frequencies of target verbs
    def verb_count(self, targets=action_verbs):
        tagged_tokens = utils.pos_tag(self.txt)

        res = {}
        for t in targets:
            res[utils.stem(t)] = 0

        for (token, tag) in tagged_tokens:
            if tag == "VB" or tag == "VBP":
                verb = utils.stem(token)
                if verb in res:
                    res[verb] += 1

        return res


if __name__ == "__main__":
    pp = pprint.PrettyPrinter(sort_dicts=False, depth=4)

    INPUT = "./data/essay1.txt"
    with open(INPUT, 'r') as file:
        txt = file.read()

    stat = StatCollector(txt).get_stat()
    print(f"basic stat: {stat}")

    text_analyzer = TextAnalyzer(txt)

    target_words = ['we', 'our']
    print(f"word count: {text_analyzer.word_count(target_words)}")

    target_phrases = ['such as', 'for example', 'for instance']
    print(f"phrase count: {text_analyzer.phrase_count(target_phrases)}")

    target_phrases = ['therefore', 'as a result']
    print(f"phrase count: {text_analyzer.phrase_count(target_phrases)}")

    used_action_words_count = {k: v for k, v in text_analyzer.verb_count(action_verbs).items() if v > 0}
    print(f"action verbs: {used_action_words_count}")

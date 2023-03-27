
import logging
import spacy

logger = logging.getLogger()

def get_spacy_pipeline(lang):
    return (
        spacy.load("en_core_web_sm")
        if lang == "en"
        else spacy.load(f"{lang}_core_news_sm")
    )

def chunk_sentences(sentences, max_chars):
    sentences_with_fractional_splits = []
    logger.debug("Fractional splits...")
    for sent in sentences:
        sent_len = len(sent)
        if sent_len > max_chars:
            prev_i = 0
            for i in range(0, sent_len, max_chars - 1):
                sentences_with_fractional_splits.append(sent[prev_i:i])
        else:
            sentences_with_fractional_splits.append(sent)
    sentences = sentences_with_fractional_splits
    chunks = []
    prev_cutoff = 0
    i = 0
    logger.debug("Chunk consolidation...")
    while i < len(sentences):
        cumulative_string = " ".join(sentences[prev_cutoff:i])
        if len(cumulative_string) > max_chars:
            while len(cumulative_string) > max_chars:
                i -= 1
                cumulative_string = " ".join(sentences[prev_cutoff:i])
            chunks.append(" ".join(sentences[prev_cutoff:i]))
            prev_cutoff = i
        i += 1
    return chunks


def chunk_sentences_by_span(text, sentence_spans, max_chars):
    logger.debug("Fractional splits...")
    fractional_spans = []
    for span in sentence_spans:
        sent_len = span[1] - span[0]
        if sent_len > max_chars:
            space_index = text[span[0] : span[1]].find(" ", 0, max_chars)
            if space_index == -1:
                space_index = max_chars - 1
            fractional_spans.extend(
                ((span[0], span[0] + space_index), (span[0] + space_index + 1, span[1]))
            )
        else:
            fractional_spans.append(span)
    chunks = []
    prev_cutoff = 0
    logger.debug("Chunk consolidation...")
    for i in range(len(fractional_spans)):
        start = fractional_spans[prev_cutoff][0]
        end = fractional_spans[i][1]
        length = end - start
        if length > max_chars:
            chunks.append((start, fractional_spans[i - 1][1]))
            prev_cutoff = i
    return chunks


def break_up_sentences(text, sentence_spans, max_chars):
    logger.debug("Breaking up sentences beyond API limit...")
    fractional_spans = []
    for span in sentence_spans:
        sent_len = span[1] - span[0]
        if sent_len > max_chars:
            space_index = text[span[0] : span[1]].find(" ", 0, max_chars)
            if space_index == -1:
                space_index = max_chars - 1
            fractional_spans.extend(
                ((span[0], span[0] + space_index), (span[0] + space_index + 1, span[1]))
            )
        else:
            fractional_spans.append(span)
    return fractional_spans

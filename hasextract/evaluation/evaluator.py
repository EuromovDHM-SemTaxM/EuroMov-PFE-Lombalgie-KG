from abc import ABC, abstractmethod
from itertools import combinations, product
import logging

from jellyfish import levenshtein_distance
from tqdm import tqdm

from hasextract.kext.knowledgeextractor import ExtractedKnowledge

logger = logging.getLogger()


class ExtractedKnowledgeEvaluator(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def __call__(self, extracted: list[ExtractedKnowledge]):
        pass


class CompositeKnowledgeEvaluator(ExtractedKnowledgeEvaluator):
    def __init__(self):
        super().__init__()
        self.registry = []  # type: list[ExtractedKnowledgeEvaluator]

    def add_evaluator(self, extractors=list[ExtractedKnowledgeEvaluator]):
        self.registry.extend(extractors)

    def __call__(self, extracted: list[ExtractedKnowledge]):
        global_evaluation = {}
        for extractor in self.registry:
            logger.debug(f"Invoking {extractor.__class__.__name__}")
            global_evaluation[extractor.__class__.__name__] = extractor(extracted)
        return global_evaluation


class DescriptiveStatisticsEvaluator(ExtractedKnowledgeEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, extracted: dict[str, ExtractedKnowledge]):
        evaluations = {}

        for ke in tqdm(
            extracted.values(), desc="Computing descriptive statistics over extractions"
        ):
            concepts = ke.concepts
            avg_len = 0.0
            num_mentions = 0
            avg_confidence = 0
            num_mappings = 0
            for c in concepts:
                avg_len += len(c.label)

                num_mentions += len(c.instances) if c.instances else 1
                num_mappings += (
                    len(c.mappings)
                    if c.mappings and isinstance(c.mappings, dict)
                    else 0
                )

                if c.confidence_score:
                    avg_confidence += avg_confidence

            avg_len /= len(concepts) * 1.0
            avg_confidence /= len(concepts) * 1.0

            source_concepts = set()
            target_concepts = set()
            relation_types = set()
            for relation in ke.relations:
                source_concepts.add(relation.source.idx)
                target_concepts.add(relation.target.idx)
                relation_types.add(relation.name)

            evaluations[ke.agent] = {
                "num_entities": len(concepts),
                "num_mentions": num_mentions,
                "num_mappings": num_mappings,
                "avg_len": avg_len,
                "num_relations": len(ke.relations),
                "unique_sources": len(source_concepts),
                "unique_targets": len(target_concepts),
                "num_relation_types": len(relation_types),
            }

            if avg_confidence > 0:
                evaluations[ke.agent]["avg_confidence"] = avg_confidence

            if ke.semantic_roles and len(ke.semantic_roles) > 0:
                evaluations[ke.agent]["semantic_roles"] = len(ke.semantic_roles)
            if ke.amr_parses and len(ke.amr_parses) > 0:
                evaluations[ke.agent]["amr_graphs"] = len(ke.amr_parses)
        return evaluations


def _flatten_mentions(concepts):
    mention_spans = []
    mention_index = set()
    for concept in tqdm(concepts, desc="Merging and sorting mentions"):
        for instance in concept.instances:
            index_tuple = (instance.start, instance.end, concept.idx)
            if index_tuple not in mention_index:
                mention_spans.append((instance.start, instance.end, concept.idx))
                mention_index.add(index_tuple)
    mention_spans = list(mention_spans)
    return sorted(mention_spans, key=lambda x: x[0])


def _index_relation_concept_mentions(relations):
    return {
        (source_mention.start, source_mention.end): (
            target_concept_mention.start,
            target_concept_mention.end,
        )
        for relation in relations
        for target_concept_mention in relation.target.instances
        for source_mention in relation.source.instances
    }


def _at_least_one_equal_pair(mention_pairs, index_1, index_2):
    return any(
        m[0] in index_1 and m[1] in index_2 and index_1[m[0]] == index_2[m[1]]
        for m in mention_pairs
    )


def _compute_overlaps(mention_spans_1, mention_spans_2):
    common_full = {}
    common_partial_1 = {}
    common_partial_2 = {}

    last_end_1 = mention_spans_1[-1][1]
    last_end_2 = mention_spans_2[-1][1]

    logger.debug("Matching mention spans...")
    logger.debug(f"Last offsets: {last_end_1} {last_end_2}")
    i = j = 0
    while mention_spans_1[i][1] < last_end_1 and mention_spans_2[j][1] < last_end_2:
        if mention_spans_1[i][0] == mention_spans_2[j][0]:
            if mention_spans_1[i][1] == mention_spans_2[j][1]:
                if (mention_spans_1[i][2], mention_spans_2[j][2]) not in common_full:
                    common_full[(mention_spans_1[i][2], mention_spans_2[j][2])] = []

                common_full[(mention_spans_1[i][2], mention_spans_2[j][2])].append(
                    (mention_spans_1[i][:2], mention_spans_2[j][:2])
                )
                i += 1
                j += 1
            elif mention_spans_1[i][1] < mention_spans_2[j][1]:
                if (
                    mention_spans_1[i][2],
                    mention_spans_2[j][2],
                ) not in common_partial_2:
                    common_partial_2[
                        (mention_spans_1[i][2], mention_spans_2[j][2])
                    ] = []
                common_partial_2[(mention_spans_1[i][2], mention_spans_2[j][2])].append(
                    (
                        mention_spans_1[i][:2],
                        mention_spans_2[j][:2],
                    )
                )
                i += 1
            elif mention_spans_1[i][1] > mention_spans_2[j][1]:
                if (
                    mention_spans_1[i][2],
                    mention_spans_2[j][2],
                ) not in common_partial_1:
                    common_partial_1[
                        (mention_spans_1[i][2], mention_spans_2[j][2])
                    ] = []

                common_partial_1[(mention_spans_1[i][2], mention_spans_2[j][2])].append(
                    (
                        mention_spans_1[i][:2],
                        mention_spans_2[j][:2],
                    )
                )
                j += 1
        elif mention_spans_1[i][0] < mention_spans_2[j][0]:
            i += 1
        elif mention_spans_1[i][0] > mention_spans_2[j][0]:
            j += 1
            # logger.debug(f"{i} {j}")
    return common_full, common_partial_1, common_partial_2


def _compute_overlapping_relations(common_full, relations_1, relations_2):
    indexed_relations_a = _index_relation_concept_mentions(relations_1)
    indexed_relations_b = _index_relation_concept_mentions(relations_2)
    overlapping_relations = set()
    for concept_pair in common_full:
        mentions = common_full[concept_pair]
        if _at_least_one_equal_pair(mentions, indexed_relations_a, indexed_relations_b):
            overlapping_relations.add(concept_pair)
            
    return overlapping_relations


class OverlapEvaluator(ExtractedKnowledgeEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, extracted: dict[str, ExtractedKnowledge]):
        evaluations = {}
        if len(extracted) > 1:
            pairs = list(combinations(list(extracted.values()), 2))
            for pair in tqdm(pairs, desc="Iterating over extraction pairs"):

                concepts_1 = pair[0].concepts
                concepts_2 = pair[1].concepts
                logger.debug("Merging and sorting mentions...")
                mention_spans_1 = _flatten_mentions(concepts_1)
                logger.debug("Merging and sorting mentions...")
                mention_spans_2 = _flatten_mentions(concepts_2)

                common_full, common_partial_1, common_partial_2 = _compute_overlaps(
                    mention_spans_1, mention_spans_2
                )

                sys_1 = pair[0].agent
                sys_2 = pair[1].agent

                total_1 = len(concepts_1)
                total_2 = len(concepts_2)
                num_common = len(common_full)
                num_common_partial_1 = len(common_partial_1)
                num_common_partial_2 = len(common_partial_2)
                num_specific_1 = max(
                    total_1 - num_common - num_common_partial_1 - num_common_partial_2,
                    0,
                )
                num_specific_2 = max(
                    total_2 - num_common - num_common_partial_1 - num_common_partial_2,
                    0,
                )

                num_overlapping_relations = len(_compute_overlapping_relations(common_full, pair[0].relations, pair[1].relations))

                evaluations[f"{sys_1}_vs_{sys_2}"] = {
                    f"total_{sys_1}": str(total_1),
                    f"total_{sys_2}": str(total_2),
                    f"common (%{sys_1} %{sys_2})": f"{num_common} ({num_common/total_1*1.:.2f} {num_common/total_2*1.:.2f})",
                    f"common_partial_{sys_1}_leq_{sys_2} (%{sys_2})": f"{num_common_partial_1} ({num_common_partial_1/total_2*1.:.2f})",
                    f"common_partial_{sys_2}_leq_{sys_1} (%{sys_1})": f"{num_common_partial_2} ({num_common_partial_2/total_1*1.:.2f})",
                    f"specific_{sys_1} (%{sys_1})": f"{num_specific_1} ({num_specific_1/total_1*1.:.2f})",
                    f"specific_{sys_2} (%{sys_2})": f"{num_specific_2} ({num_specific_2/total_2*1.:.2f})",
                }

                if len(pair[0].relations) > 0 and len(pair[1].relations) > 0:
                    evaluations[f"{sys_1}_vs_{sys_2}"][
                        f"relations with shared source+target (%{sys_1} %{sys_2})"
                    ] = f"{num_overlapping_relations} ({num_overlapping_relations/len(pair[0].relations)*1.:.2f} {num_overlapping_relations/len(pair[1].relations)*1.:.2f})"
        else:
            evaluations[
                "error"
            ] = "OverlapEvaluator: Only one extracted knowledge collection available, cannot compute overlap statistics..."

        return evaluations


def evaluate_extraction_complementarity(extraction_results, threshold=0.25):
    systems = list(extraction_results.keys())
    system_pairs = combinations(systems, 2)
    for pair in system_pairs:
        print(f"Evaluating pair: {pair}")
        terms_1 = [c["term"] for c in extraction_results[pair[0]]["concepts"]]
        terms_2 = [c["term"] for c in extraction_results[pair[1]]["concepts"]]

        unique_combinations = [
            list(zip(terms_1, element))
            for element in product(terms_2, repeat=len(terms_1))
        ]

        common = []
        for combination in unique_combinations:
            dist = levenshtein_distance.normalized_distance(
                combination[0], combination[1]
            )
            if dist < threshold:
                common.append(combination)

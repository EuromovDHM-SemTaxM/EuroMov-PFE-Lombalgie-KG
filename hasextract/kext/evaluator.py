from abc import ABC, abstractmethod
from itertools import combinations, product
import logging

from jellyfish import levenshtein_distance

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
            global_evaluation[extractor.__class__.__name__] = extractor(
                extracted)
        return global_evaluation


class DescriptiveStatisticsEvaluator(ExtractedKnowledgeEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, extracted: dict[str, ExtractedKnowledge]):
        evaluations = {}

        for ke in extracted.values():
            concepts = ke.concepts
            avg_len = 0.0
            num_mentions = 0
            avg_confidence = 0
            for c in concepts:
                avg_len += len(c.matched_text)

                if c.instances:
                    num_mentions += len(c.instances)
                else:
                    num_mentions += 1

                if c.confidence_score:
                    avg_confidence += avg_confidence

            avg_len /= len(concepts) * 1.
            avg_confidence /= len(concepts) * 1.

            source_concepts = set()
            target_concepts = set()
            relation_types = set()
            for relation in ke.relations:
                source_concepts.add(relation.source.id)
                target_concepts.add(relation.target.id)
                relation_types.add(relation.name)

            evaluations[ke.agent] = {
                'num_entities': len(concepts),
                'num_mentions': num_mentions,
                'avg_len': avg_len,
                'num_relations': len(ke.relations),
                'unique_sources': len(source_concepts),
                'unique_targets': len(target_concepts),
                'num_relation_types': len(relation_types)
            }

            if avg_confidence > 0:
                evaluations[ke.agent]['avg_confidence'] = avg_confidence
        return evaluations


class OverlapEvaluator(ExtractedKnowledgeEvaluator):
    def __init__(self) -> None:
        super().__init__()

    def __call__(self, extracted: dict[str, ExtractedKnowledge]):
        evaluations = {}
        if len(extracted) > 1:
            pairs = combinations(list(extracted.values()), 2)
            for pair in pairs:

                concepts_1 = pair[0].concepts
                concepts_2 = pair[1].concepts
                mention_spans_1 = sorted([(instance[0], instance[1], concept)
                                          for concept in concepts_1
                                          for instance in concept.instances],
                                         key=lambda x: x[0])
                mention_spans_2 = sorted([(instance[0], instance[1], concept)
                                          for concept in concepts_2
                                          for instance in concept.instances],
                                         key=lambda x: x[0])

                common_full = set()
                common_partial_1 = set()
                common_partial_2 = set()

                last_end_1 = mention_spans_1[-1][1]
                last_end_2 = mention_spans_2[-1][1]

                i = j = 0
                while mention_spans_1[i][1] <= last_end_1 and mention_spans_2[
                        j][1] <= last_end_2:
                    if mention_spans_1[i][0] == mention_spans_2[j][0]:
                        if mention_spans_1[i][1] == mention_spans_2[j][1]:
                            common_full.add(
                                (mention_spans_1[i][2], mention_spans_2[j][2]))
                            i += 1
                            j += 1
                        elif mention_spans_1[i][1] < mention_spans_2[j][1]:
                            common_partial_2.add(
                                (mention_spans_1[i][2], mention_spans_2[j][2]))
                            i += 1
                        elif mention_spans_1[i][1] > mention_spans_2[j][1]:
                            common_partial_1.add(
                                (mention_spans_1[i][2], mention_spans_2[j][2]))
                            j += 1
                    elif mention_spans_1[i][0] < mention_spans_2[j][0]:
                        i += 1
                    elif mention_spans_1[i][0] > mention_spans_2[j][0]:
                        j += 1

                sys_1 = pair[0].agent
                sys_2 = pair[1].agent

                total_1 = len(concepts_1)
                total_2 = len(concepts_2)
                num_common = len(common_full)
                num_common_partial_1 = len(common_partial_1)
                num_common_partial_2 = len(common_partial_2)
                num_specific_1 = total_1 - num_common - num_common_partial_1 - num_common_partial_2
                num_specific_2 = total_2 - num_common - num_common_partial_1 - num_common_partial_2
                evaluations[f"{sys_1}_vs_{sys_2}"] = {
                    f"total_{sys_1}":
                    str(total_1),
                    f"total_B_{sys_2}":
                    str(total_2),
                    f"common (%{sys_1} %{sys_2})":
                    f"{num_common} ({num_common/total_1*1.:.2f} {100.*num_common/total_2*1.:.2f})",
                    f"common_partial_{sys_1}_leq_{sys_2} (%{sys_2})":
                    f"{num_common_partial_2} ({100.*num_common_partial_2/total_2*1.:.2f})",
                    f"common_partial_{sys_2}_leq_{sys_1} (%{sys_1})":
                    f"{num_common_partial_2} ({100.*num_common_partial_1/total_1*1.:.2f})",
                    f"specific_{sys_1} (%{sys_1})":
                    f"{num_specific_1} ({100.*num_specific_1/total_1*1.:.2f})",
                    f"specific_{sys_2} (%{sys_2})":
                    f"{num_specific_2} ({100.*num_specific_2/total_2*1.:.2f})",
                }
        else:
            evaluations[
                'error'] = "OverlapEvaluator: Only one extracted knowledge collection available, cannot compute overlap statistics..."

        return evaluations


def evaluate_extraction_complementarity(extraction_results, threshold=0.25):
    systems = list(extraction_results.keys())
    system_pairs = combinations(systems, 2)
    for pair in system_pairs:
        print(f"Evaluating pair: {pair}")
        terms_1 = [c['term'] for c in extraction_results[pair[0]]['concepts']]
        terms_2 = [c['term'] for c in extraction_results[pair[1]]['concepts']]

        unique_combinations = list(
            list(zip(terms_1, element))
            for element in product(terms_2, repeat=len(terms_1)))

        common = []
        for combi in unique_combinations:
            dist = levenshtein_distance.normalized_distance(combi[0], combi[1])
            if dist < threshold:
                common.append(combi)

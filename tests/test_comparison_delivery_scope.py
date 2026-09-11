import pytest

from core.language.relational_request import compared_subjects, comparison_sides_are_covered
from core.brain.llm.latent_cortex.output_quality import evaluate_facet_coverage


@pytest.mark.parametrize('suffix', [
    'in three numbered points, then give one everyday example of each',
    'in two short paragraphs',
    'using a table',
    'and then explain the tradeoffs',
])
def test_delivery_instruction_is_not_a_compared_subject(suffix):
    question = f'Explain the difference between evaporation and boiling {suffix}.'
    assert compared_subjects(question) == (('evaporation',), ('boiling',))
    assert comparison_sides_are_covered('Evaporation occurs at the surface.', question) is False
    assert comparison_sides_are_covered('Evaporation is surface change; boiling is bulk change.', question)


def test_subject_preposition_is_not_removed():
    assert compared_subjects('Compare water in a lake and water in a river.') == (
        ('water', 'in', 'lake'), ('water', 'in', 'river'),
    )


def test_comparison_without_connective_can_fulfill_facets():
    evidence = evaluate_facet_coverage(
        '1. Evaporation occurs at the surface of a liquid. Boiling occurs throughout it.\n'
        '2. Evaporation cools liquid because energetic molecules escape.\n'
        '3. Boiling begins when vapor pressure reaches the external pressure.\n'
        'Examples: wet clothes drying; water bubbling in a pot.',
        'Explain the difference between evaporation and boiling in three numbered points, '
        'then give one everyday example of each.',
    )
    assert set(evidence['requested']) <= set(evidence['satisfied'])

"""Reading notes preserve document bodies and keep frozen records untouched."""
from tools.render_reader_navigation import (
    BLOCK,
    CLOSE,
    OPEN,
    category,
    link,
    with_card,
)


def test_adding_a_reading_note_preserves_every_original_body_byte():
    original = '# A measured mechanism\n\nStatus: Guide\n\n## Definition\n\n`x = 2`\n'
    card = f'{OPEN}\n> An explanation.\n{CLOSE}'
    result = with_card(original, card)
    assert BLOCK.sub('', result) == original
    assert with_card(result, card) == result


def test_updating_a_reading_note_replaces_only_the_owned_block():
    original = '# Topic\n\nAn equation and its dated result.\n'
    first = with_card(original, f'{OPEN}\n> Old note.\n{CLOSE}')
    revised = with_card(first, f'{OPEN}\n> New note.\n{CLOSE}')
    assert 'Old note' not in revised
    assert BLOCK.sub('', revised) == original


def test_frozen_and_generated_records_never_receive_reader_cards():
    for name in ('docs/ISC_V6_PREREGISTRATION.md', 'docs/evidence/run.md',
                 'docs/AURA_EXECUTION_TRACKER.md', 'proof_kernel/report.md',
                 'docs/worktodo/RESULTS.md', 'docs/SETTINGS_WIRING_AUDIT.md',
                 'aura/knowledge/bryan-curated-media.md'):
        assert category(name) != 'Guide or reference'
    assert category('security/OWASP_LLM_MAPPING.md') != 'Guide or reference'
    assert category('docs/DEPENDENCY_AUDIT.md', '# Dependency Audit Policy\n') != 'Guide or reference'


def test_deep_page_links_stay_inside_the_repository():
    assert link('docs/runbooks/disk-full.md', 'docs/VARIABLES_EXPLAINED.md', 'resources') == '../VARIABLES_EXPLAINED.md#resources'
    assert link('README.md', 'docs/ISC_STATUS.md') == 'docs/ISC_STATUS.md'


def test_a_page_without_a_title_is_left_as_written():
    original = 'A preserved fragment without a page heading.\n'
    assert with_card(original, f'{OPEN}\n> Note.\n{CLOSE}') == original


def test_a_reading_card_cannot_hide_a_historical_status():
    original = '# Earlier audit\n\nStatus: Historical\n\nA retained result.\n'
    card = f'{OPEN}\n' + '> Reading note.\n' * 80 + CLOSE
    assert category('docs/earlier_audit.md', with_card(original, card)) == 'Historical or frozen record'


def test_current_obligations_are_not_labelled_a_frozen_record():
    assert category('docs/AURA_1_0_MASTER_TODO.md') == 'Living status or append-only ledger'
    assert category('docs/AURA_EXECUTION_TRACKER.md') == 'Living status or append-only ledger'


def test_the_requirement_docket_is_regenerated_rather_than_hand_edited():
    assert category('docs/AURA_REMAINING_DOCKET.md') == 'Generated reference'
    assert category('artifacts/architecture/latest.md') == 'Generated reference'

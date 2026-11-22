"""Reader navigation indexes and classifies Markdown pages."""
from tools.render_reader_navigation import (
    BLOCK,
    CLOSE,
    OPEN,
    category,
    link,
)


def test_removing_a_generated_header_preserves_every_body_byte():
    original = '# A measured mechanism\n\nStatus: Guide\n\n## Definition\n\n`x = 2`\n'
    card = f'{OPEN}\n> An explanation.\n{CLOSE}'
    with_header = original.replace('\n\n', f'\n\n{card}\n\n', 1)
    assert BLOCK.sub('', with_header) == original


def test_frozen_and_generated_records_are_classified_separately():
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


def test_a_reading_card_cannot_hide_a_historical_status():
    original = '# Earlier audit\n\nStatus: Historical\n\nA retained result.\n'
    card = f'{OPEN}\n' + '> Reading note.\n' * 80 + CLOSE
    with_header = original.replace('\n\n', f'\n\n{card}\n\n', 1)
    assert category('docs/earlier_audit.md', with_header) == 'Historical or frozen record'


def test_current_obligations_are_not_labelled_a_frozen_record():
    assert category('docs/AURA_1_0_MASTER_TODO.md') == 'Living status or append-only ledger'
    assert category('docs/AURA_EXECUTION_TRACKER.md') == 'Living status or append-only ledger'


def test_the_requirement_docket_is_regenerated_rather_than_hand_edited():
    assert category('docs/AURA_REMAINING_DOCKET.md') == 'Generated reference'
    assert category('artifacts/architecture/latest.md') == 'Generated reference'

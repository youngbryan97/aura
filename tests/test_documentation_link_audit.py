"""The link audit reads real targets while leaving examples and private hosts alone."""
from tools import audit_documentation_links as audit


def test_sentence_punctuation_does_not_become_part_of_a_paper_url():
    text = 'Read [the paper](https://arxiv.org/abs/2006.08381): a useful idea.\n'
    assert audit.markdown_targets(text) == {'https://arxiv.org/abs/2006.08381'}


def test_code_examples_are_not_requested_as_public_document_links():
    text = '```bash\ncurl https://not-an-actual-endpoint.invalid/action\n```\n[Paper](https://arxiv.org/abs/2005.11401)\n'
    assert audit.markdown_targets(text) == {'https://arxiv.org/abs/2005.11401'}


def test_local_and_credential_urls_cannot_enter_public_http_checks():
    for url in ('http://localhost:8000/api/shutdown', 'http://127.0.0.1/',
                'http://169.254.169.254/', 'http://10.1.2.3/', 'http://[::1]/',
                'https://example.com/', 'https://user:secret@host.org/',
                'https://host.org/?api_key=secret'):
        assert not audit.public_url(url)
    assert audit.public_url('https://arxiv.org/abs/2005.11401')


def test_redirects_to_local_services_are_refused():
    import urllib.error
    handler = audit.PublicRedirect()
    try:
        handler.redirect_request(None, None, 302, '', {}, 'http://127.0.0.1:8000/')
    except urllib.error.URLError:
        pass
    else:
        raise AssertionError('a public link must not redirect the audit into a local service')


def test_heading_checks_resolve_relative_to_the_document(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/page.md').write_text('# The page\n\n## A result\n')
    assert audit.local_reading('docs/page.md', '#a-result') == 'ok'
    assert audit.local_reading('docs/page.md', '#absent') == 'missing_heading'
    assert audit.local_reading('docs/page.md', 'missing.md') == 'missing_target'

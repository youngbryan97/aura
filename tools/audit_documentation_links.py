#!/usr/bin/env python3
"""Audit Markdown targets without starting Aura or loading a model.

Local targets and heading anchors are checked from the checkout. Optional
public HTTP checks use bounded HEAD requests. Restricted or timed-out sites
remain unverified instead of being declared broken.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ipaddress
import json
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
INLINE = re.compile(r'!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
REFERENCE = re.compile(r'^\s*\[[^\]]+\]:\s*(\S+)', re.M)
AUTOLINK = re.compile(r'<(https?://[^>\s]+)>')
BARE = re.compile(r'https?://[^\s<>`]+')


def markdown_targets(text: str) -> set[str]:
    lines, fenced = [], False
    for line in text.splitlines():
        if re.match(r'^\s*(```|~~~)', line):
            fenced = not fenced
            lines.append('')
        else:
            lines.append('' if fenced else line)
    body = '\n'.join(lines)
    targets = {m.group(1).strip('<>') for m in INLINE.finditer(body)}
    targets.update(m.group(1) for m in REFERENCE.finditer(body) if re.search(r'https?://|[/#]|\.[A-Za-z]{2,5}$', m.group(1)))
    targets.update(AUTOLINK.findall(body))
    for value in BARE.findall(body):
        value = value.rstrip('.,;:')
        while value.endswith(')') and value.count(')') > value.count('('):
            value = value[:-1]
        targets.add(value)
    return targets


def public_url(url: str) -> bool:
    p = urllib.parse.urlsplit(url)
    host = p.hostname or ''
    if p.scheme not in ('http', 'https') or p.username or p.password:
        return False
    try:
        if not ipaddress.ip_address(host).is_global:
            return False
    except ValueError:
        pass
    if (host in ('localhost', '0.0.0.0', '127.0.0.1', '::1')
            or host.endswith(('.local', '.internal', '.test', '.example'))
            or 'example.' in host or re.match(r'^(10\.|192\.168\.|172\.(1[6-9]|2\d|3[01])\.)', host)):
        return False
    if re.search(r'YOUR_|TOKEN|API_KEY|\{|\}|<|>', url):
        return False
    if any(re.search(r'token|secret|signature|password|api.?key', k, re.I) for k, _ in urllib.parse.parse_qsl(p.query)):
        return False
    return bool(host)


def anchors(path: Path) -> set[str]:
    def slug(value: str) -> str:
        value = re.sub(r'<[^>]+>', '', value).strip().lower()
        value = re.sub(r'[^\w\- ]', '', value)
        return value.replace(' ', '-')
    text = path.read_text(errors='replace')
    result = {slug(m.group(1)) for m in re.finditer(r'^#{1,6}\s+(.+)$', text, re.M)}
    result.update(re.findall(r'\bid=["\']([^"\']+)', text))
    return result


def local_reading(doc: str, target: str) -> str:
    if target.startswith(('mailto:', 'tel:', 'data:', 'file:', 'javascript:')):
        return 'non_web_or_local_only'
    if any(x in target for x in ('YOUR_', '{', '}', 'NNNN', 'XXXX')):
        return 'illustrative_target'
    parsed = urllib.parse.urlsplit(target)
    if parsed.scheme or parsed.netloc:
        return 'external'
    path = ROOT / doc if not parsed.path else (ROOT / doc).parent / urllib.parse.unquote(parsed.path)
    try:
        path = path.resolve()
        path.relative_to(ROOT)
    except ValueError:
        return 'outside_checkout'
    if not path.exists():
        return 'missing_target'
    if parsed.fragment and path.is_file() and path.suffix.lower() == '.md':
        return 'ok' if urllib.parse.unquote(parsed.fragment) in anchors(path) else 'missing_heading'
    return 'ok'


class PublicRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not public_url(newurl):
            raise urllib.error.URLError('non-public redirect refused')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http_reading(url: str, timeout: float) -> dict:
    opener = urllib.request.build_opener(PublicRedirect())
    headers = {'User-Agent': 'AuraDocumentationLinkAudit/1.0'}
    try:
        req = urllib.request.Request(url, method='HEAD', headers=headers)
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code not in (405, 501):
                raise
            req = urllib.request.Request(url, method='GET', headers={**headers, 'Range': 'bytes=0-0'})
            response = opener.open(req, timeout=timeout)
        with response:
            return {'status': 'reachable', 'http': response.status, 'final_url': response.url}
    except urllib.error.HTTPError as exc:
        status = 'missing' if exc.code in (404, 410) else 'unverified'
        return {'status': status, 'http': exc.code, 'reason': str(exc.reason)}
    except (OSError, ValueError, TimeoutError) as exc:
        return {'status': 'unverified', 'reason': type(exc).__name__ + ': ' + str(exc)[:160]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--external', action='store_true')
    parser.add_argument('--timeout', type=float, default=4)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--max-seconds', type=float, default=360)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    from tools.render_reader_navigation import category, inventory

    started = time.monotonic()
    paths = inventory(ROOT)
    local, urls = [], {}
    count = 0
    for doc in paths:
        p = ROOT / doc
        if not p.is_file():
            continue
        text = p.read_text(errors='replace')
        kind = category(doc, text)
        for target in sorted(markdown_targets(text)):
            count += 1
            result = local_reading(doc, target)
            if result == 'external':
                urls.setdefault(target, []).append(doc)
            elif result != 'ok':
                local.append({'document': doc, 'target': target, 'status': result, 'category': kind})
    external = []
    safe = [u for u in urls if public_url(u)]
    for url in sorted(set(urls) - set(safe)):
        external.append({'url': url, 'documents': sorted(urls[url]), 'status': 'illustrative_or_sensitive_target_not_requested'})
    if args.external:
        workers = min(2, max(1, args.workers))
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(http_reading, url, args.timeout): url for url in safe}
            done_count = 0
            for future in concurrent.futures.as_completed(pending):
                url = pending[future]
                external.append({'url': url, 'documents': sorted(urls[url]), **future.result()})
                done_count += 1
                if done_count % 25 == 0:
                    print(f'checked {done_count}/{len(safe)} public URLs', flush=True)
                if time.monotonic() - started > args.max_seconds:
                    for remaining, value in pending.items():
                        if remaining is future or any(row['url'] == value for row in external):
                            continue
                        if remaining.cancel():
                            external.append({'url': value, 'documents': sorted(urls[value]), 'status': 'unverified_time_budget'})
                    # Already running requests retain their per-request timeout.
                    break
            recorded = {row['url'] for row in external}
            for future, url in pending.items():
                if url in recorded:
                    continue
                reading = {'status': 'unverified_time_budget'} if future.cancelled() else future.result()
                external.append({'url': url, 'documents': sorted(urls[url]), **reading})
    else:
        external += [{'url': u, 'documents': sorted(urls[u]), 'status': 'not_requested'} for u in safe]
    external.sort(key=lambda x: x['url'])
    result = {'schema': 'aura.documentation.link_audit.v1', 'checked_at_utc': datetime.now(UTC).isoformat(timespec='seconds'),
              'source_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
              'markdown_pages': len(paths), 'unique_page_targets': count, 'unique_external_urls': len(urls),
              'scope_note': 'Existing files and heading anchors were checked locally. Optional public HTTP reachability does not validate page contents. Historical and frozen source text is retained; missing historical targets are reported rather than silently rewritten. Local examples, credentials and private endpoints are not requested.',
              'local_findings': local, 'external': external,
              'external_counts': dict(Counter(x['status'] for x in external))}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
    current_bad = [x for x in local if x['category'] == 'Guide or reference' and x['status'] in ('missing_target','missing_heading')]
    print(json.dumps({'pages': len(paths), 'targets': count, 'current_missing': len(current_bad), 'historical_or_other_findings':len(local)-len(current_bad), 'external': result['external_counts'], 'out': str(args.out)}), flush=True)
    return 1 if current_bad else 0


if __name__ == '__main__':
    raise SystemExit(main())

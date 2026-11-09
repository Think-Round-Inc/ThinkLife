"""Release smoke checks; HTTP or semantic failures return nonzero."""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlparse
from urllib.request import Request, build_opener, HTTPRedirectHandler
from release_common import ROOT, write_json


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def base_url(value):
    p = urlparse(value)
    if p.scheme not in ('http', 'https') or not p.hostname or p.username or p.password or p.query or p.fragment:
        raise argparse.ArgumentTypeError('Use an HTTP(S) URL without credentials, query or fragment')
    return value.rstrip('/')


def check(url, predicate, attempts=3, timeout=10, delay=2):
    for attempt in range(1, attempts + 1):
        try:
            req = Request(url, headers={'Accept': 'application/json', 'Cache-Control': 'no-cache'})
            with build_opener(NoRedirect()).open(req, timeout=timeout) as res:
                if res.status != 200:
                    raise ValueError('Unexpected HTTP status')
                raw = res.read(1024 * 1024 + 1)
                if len(raw) > 1024 * 1024:
                    raise ValueError('Response exceeds limit')
                data = json.loads(raw)
                if not isinstance(data, dict) or not predicate(data):
                    raise ValueError('Response contract mismatch')
            return {'passed': True, 'attempts': attempt}
        except HTTPError as exc:
            reason = 'HTTP ' + str(exc.code)
            exc.close()
        except (URLError, TimeoutError, OSError):
            reason = 'Connection or TLS failure'
        except (ValueError, TypeError, AttributeError, KeyError):
            reason = 'Response contract mismatch'
        if attempt < attempts:
            time.sleep(delay)
    return {'passed': False, 'attempts': attempts, 'reason': reason}


def checks(frontend, backend, keycloak, realm, commit, release_id):
    issuer = keycloak + '/realms/' + quote(realm, safe='')
    return [
        ('frontend_health', frontend + '/api/health', lambda d:d.get('status') == 'healthy' and d.get('service') == 'ThinkLife Frontend'),
        ('frontend_release', frontend + '/release.json', lambda d:d.get('source_commit') == commit and d.get('release_id') == release_id and d.get('working_tree_dirty') is False),
        ('backend_identity', backend + '/', lambda d:d.get('message') == 'ThinkLife Backend' and d.get('version') == '0.7.0'),
        ('backend_health', backend + '/health', lambda d:d.get('status') == 'healthy' and d.get('brain') is True and d.get('zoe') is True),
        ('brain_health', backend + '/api/brain/health', lambda d:d.get('status') == 'healthy' and d.get('brain_status', {}).get('overall') == 'healthy'),
        ('zoe_health', backend + '/api/zoe/health', lambda d:d.get('service') == 'zoe' and all(d.get(k) is True for k in ('initialized','zoe_core_available','zoe_agent_available')) and d.get('zoe_agent_health', {}).get('status') == 'healthy'),
        ('keycloak_oidc', issuer + '/.well-known/openid-configuration', lambda d:d.get('issuer') == issuer and all(isinstance(d.get(k), str) and d[k].startswith(issuer + '/') for k in ('authorization_endpoint','token_endpoint','jwks_uri'))),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frontend-url', type=base_url, default='http://localhost:3000')
    p.add_argument('--backend-url', type=base_url, default='http://localhost:8000')
    p.add_argument('--keycloak-url', type=base_url, default='http://localhost:8080')
    p.add_argument('--realm', default='thinklife')
    p.add_argument('--expected-commit', required=True)
    p.add_argument('--release-id', required=True)
    p.add_argument('--attempts', type=int, default=3)
    p.add_argument('--timeout', type=float, default=10)
    p.add_argument('--delay', type=float, default=2)
    p.add_argument('--report', type=Path, default=ROOT / '.release/smoke-results.json')
    args = p.parse_args()
    if len(args.expected_commit) != 40 or any(c not in '0123456789abcdef' for c in args.expected_commit):
        p.error('--expected-commit must be a full lowercase SHA-1 commit')
    if not 1 <= args.attempts <= 10 or not 0 < args.timeout <= 60 or not 0 <= args.delay <= 30:
        p.error('attempts 1..10, timeout (0,60], delay [0,30] required')
    report = {'release_id': args.release_id, 'expected_commit': args.expected_commit, 'checked_at': datetime.now(timezone.utc).isoformat(), 'checks': []}
    for name, url, predicate in checks(args.frontend_url, args.backend_url, args.keycloak_url, args.realm, args.expected_commit, args.release_id):
        result = {'name': name, 'url': url, **check(url, predicate, args.attempts, args.timeout, args.delay)}
        report['checks'].append(result)
        print(('PASS ' if result['passed'] else 'FAIL ') + name + ': ' + result.get('reason', 'OK'), flush=True)
    report['passed'] = all(x['passed'] for x in report['checks'])
    report['scope'] = 'Availability and frontend commit check; not LLM generation, login, or backend commit attestation.'
    write_json(args.report, report)
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

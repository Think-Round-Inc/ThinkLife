"""Validate release environment contracts and compare protected prior env files.

Reports names only, never values or hashes of secrets. Does not load app code.
"""
import argparse
import json
import math
import os
import re
from pathlib import Path
from urllib.parse import urlparse
from release_common import ROOT, env_references, read_env, write_json


def validate(scope, values, contract):
    errors, warnings = [], []
    rules = contract[scope]
    for key, rule in rules.items():
        value = values.get(key, rule.get('default', ''))
        if not value.strip():
            if rule.get('required') or ('default' in rule and key in values) or (rule.get('type') in ('int', 'float') and key in values):
                errors.append(key + ': required or explicitly configured value is empty')
            continue
        if re.search(r'your[_-]|placeholder|change[_-]?me|replace[_-]?me|<[^>]+>', value, re.I):
            errors.append(key + ': placeholder value')
        kind = rule.get('type', 'string')
        if kind == 'url':
            parsed = urlparse(value)
            if parsed.scheme not in ('http', 'https') or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
                errors.append(key + ': requires an HTTP(S) URL without credentials, query or fragment')
        if kind in ('int', 'float'):
            try:
                number = int(value) if kind == 'int' else float(value)
                if not math.isfinite(number) or number < rule.get('min', -math.inf) or number > rule.get('max', math.inf):
                    raise ValueError()
            except ValueError:
                errors.append(key + ': invalid ' + kind + ' or out of range')
        if key == 'NEXTAUTH_SECRET' and len(value) < 32:
            errors.append(key + ': requires at least 32 characters')
    for key in sorted(set(values) - set(rules)):
        warnings.append(key + ': not in release contract; verify whether the application uses it')
    for pair in (('LANGFUSE_PUBLIC_KEY', 'LANGFUSE_SECRET_KEY'), ('GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET')):
        if bool(values.get(pair[0])) != bool(values.get(pair[1])):
            errors.append('/'.join(pair) + ': configure both or neither')
    for key in values:
        if key.startswith('NEXT_PUBLIC_') and re.search(r'SECRET|PASSWORD|API_KEY|TOKEN', key):
            errors.append(key + ': secret must not be public')
    return errors, warnings


def changes(previous, current):
    return {'added': sorted(current.keys() - previous.keys()), 'removed': sorted(previous.keys() - current.keys()), 'changed': sorted(k for k in current.keys() & previous.keys() if current[k] != previous[k])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-contract', action='store_true', help='Offline check: all consumed env names documented; no deployment validation')
    parser.add_argument('--backend-env', type=Path, default=ROOT / 'backend/.env')
    parser.add_argument('--frontend-env', type=Path, default=ROOT / 'frontend/.env.local')
    parser.add_argument('--previous-backend-env', type=Path)
    parser.add_argument('--previous-frontend-env', type=Path)
    parser.add_argument('--process-env', action='store_true', help='Use process variables as overrides, matching python-dotenv/Next.js precedence')
    parser.add_argument('--strict', action='store_true', help='Warnings also fail the release gate')
    parser.add_argument('--report', type=Path, default=ROOT / '.release/env-validation.json')
    args = parser.parse_args()
    contract = json.loads((ROOT / 'docs/release/environment-contract.json').read_text())
    report = {'mode': 'contract-only' if args.check_contract else 'deployment', 'errors': [], 'warnings': [], 'changes': {}}
    refs = env_references()
    for scope, entries in refs.items():
        for name in set(entries) - set(contract[scope]):
            report['errors'].append(scope + '.' + name + ': consumed but missing from contract')
    effective = {}
    if not args.check_contract:
        for scope in ('backend', 'frontend'):
            try:
                values = read_env(getattr(args, scope + '_env'))
                if args.process_env:
                    values.update({k:os.environ[k] for k in contract[scope] if k in os.environ})
                effective[scope] = values
                errors, warnings = validate(scope, values, contract)
                report['errors'] += [scope + '.' + s for s in errors]
                report['warnings'] += [scope + '.' + s for s in warnings]
                previous = getattr(args, 'previous_' + scope + '_env')
                if previous:
                    report['changes'][scope] = changes(read_env(previous), values)
                else:
                    report['warnings'].append(scope + ': no previous environment supplied; change comparison not performed')
            except (ValueError, OSError) as exc:
                # OSError contains filenames only, no file values.
                report['errors'].append(scope + ': ' + str(exc))
        if len(effective) == 2:
            for name in ('URL', 'REALM', 'CLIENT_ID'):
                b, f = 'KEYCLOAK_' + name, 'NEXT_PUBLIC_KEYCLOAK_' + name
                bv = effective['backend'].get(b, contract['backend'][b].get('default'))
                fv = effective['frontend'].get(f, contract['frontend'][f].get('default'))
                if bv and fv and bv.rstrip('/') != fv.rstrip('/'):
                    report['errors'].append('backend/frontend Keycloak ' + name + ' mismatch')
    report['passed'] = not report['errors'] and not (args.strict and report['warnings'])
    write_json(args.report, report)
    for s in report['errors']:
        print('FAIL ' + s)
    for s in report['warnings']:
        print('WARN ' + s)
    print(('PASS' if report['passed'] else 'FAIL') + ' environment ' + report['mode'])
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())

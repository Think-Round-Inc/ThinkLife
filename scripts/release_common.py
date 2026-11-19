"""Release tooling only; standard library, no application imports or API calls."""
import ast
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')


def sources():
    return [ROOT / p for p in git('ls-files', '--cached', '--others', '--exclude-standard').splitlines()
            if p.endswith(('.py', '.ts', '.tsx')) and p.startswith(('backend/', 'frontend/', 'faq-bot-rag/'))]


def env_references():
    refs = {'backend': {}, 'frontend': {}, 'faq': {}}
    for path in sources():
        rel = path.relative_to(ROOT).as_posix()
        scope = 'frontend' if rel.startswith('frontend/') else ('faq' if rel.startswith('faq-bot-rag/') else 'backend')
        content = path.read_text(encoding='utf-8-sig')
        pattern = r'''(?:os\.getenv\(\s*|os\.environ(?:\.get\(\s*|\[\s*))["']([A-Z][A-Z0-9_]*)["']|process\.env\.([A-Z][A-Z0-9_]*)'''
        for match in re.finditer(pattern, content):
            name = match.group(1) or match.group(2)
            locations = refs[scope].setdefault(name, [])
            if rel not in locations:
                locations.append(rel)
    # Provider credentials are looked up through key_map rather than a literal getenv.
    for name in ('ANTHROPIC_API_KEY', 'GEMINI_API_KEY'):
        refs['backend'][name] = ['backend/brain/cortex/workflow_engine.py (dynamic key_map)']
    return refs


def read_env(path):
    """Read simple dotenv assignments without executing or interpolating secrets.

    Multiline/expansion syntax deliberately fails instead of validating a wrong value.
    """
    result = {}
    if not Path(path).is_file():
        raise ValueError('Environment file is missing')
    for number, raw in enumerate(Path(path).read_text(encoding='utf-8-sig').splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith('#'):
            continue
        match = re.fullmatch(r'(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)', line)
        if not match:
            raise ValueError('Invalid dotenv assignment at line %s' % number)
        key, value = match.groups()
        if key in result:
            raise ValueError('Duplicate variable %s at line %s' % (key, number))
        if value.startswith(('"', "'")):
            quoted = re.fullmatch(r'''(["'])(.*?)\1\s*(?:#.*)?''', value)
            if not quoted:
                raise ValueError('Unsupported quoted value at line %s' % number)
            value = quoted.group(2)
        else:
            value = re.split(r'\s+#', value, maxsplit=1)[0].strip()
        if '${' in value:
            raise ValueError('Resolve dotenv expansion before validation at line %s' % number)
        result[key] = value
    return result


def prompt_inventory():
    # Hash whole source containers: catches changes to templates AND their assembly.
    paths = set()
    for path in sources():
        rel = path.relative_to(ROOT).as_posix()
        if path.suffix == '.py' and 'test' not in rel and re.search(r'prompt|"role":\s*"system"', path.read_text(encoding='utf-8-sig'), re.I):
            paths.add(path)
    paths.update(ROOT / p for p in ('backend/agents/zoe/personality.py', 'backend/agents/zoe/helpers.py', 'backend/brain/guardrails/fallback_handler.py'))
    return [{'path': p.relative_to(ROOT).as_posix(), 'version': 'sha256:' + digest(p), 'scope': 'whole-source-container'} for p in sorted(paths)]


def model_inventory():
    entries = []
    for path in sources():
        if path.suffix != '.py' or '/tests/' in path.as_posix():
            continue
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and re.fullmatch(r'(?:gpt-|claude-|gemini-|text-embedding-|tts-|dall-e-)[a-zA-Z0-9_.-]+', node.value):
                entries.append({'model': node.value, 'path': path.relative_to(ROOT).as_posix(), 'line': node.lineno, 'kind': 'source-reference-not-proof-of-runtime-use'})
    return sorted(entries, key=lambda x: (x['path'], x['line']))

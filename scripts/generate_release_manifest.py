""" Generate traceable release evidence and frontend build metadata """
import argparse
import ast
import json
import re
from pathlib import Path
from release_common import ROOT, digest, git, model_inventory, prompt_inventory, write_json


def dependency_inventory():
    result = {}
    for name in ('requirements.txt', 'backend/requirements.txt', 'faq-bot-rag/requirements.txt'):
        p = ROOT / name
        requirements = [s.strip() for s in p.read_text().splitlines() if s.strip() and not s.lstrip().startswith('#')]
        result[name] = {'sha256': digest(p), 'declared': requirements, 'fully_pinned': all(re.fullmatch(r'[A-Za-z0-9_.\[\],-]+==[^*\s;]+', s) for s in requirements)}
    package = json.loads((ROOT / 'frontend/package.json').read_text())
    lock = json.loads((ROOT / 'frontend/package-lock.json').read_text())
    result['frontend'] = {'package_sha256': digest(ROOT / 'frontend/package.json'), 'lock_sha256': digest(ROOT / 'frontend/package-lock.json'), 'engines': package.get('engines', {}), 'dependencies': package['dependencies'], 'devDependencies': package['devDependencies'], 'resolved': {k: v.get('version') for k, v in lock.get('packages', {}).items() if k}}
    result['docker-compose.yml'] = {'sha256': digest(ROOT / 'docker-compose.yml'), 'images': re.findall(r'^\s*image:\s*(\S+)', (ROOT / 'docker-compose.yml').read_text(), re.M)}
    return result


def build(release_id, openai_model):
    if not re.fullmatch(r'[A-Za-z0-9_.-]+', release_id):
        raise ValueError('release-id must contain only letters, digits, dots, underscores or hyphens')
    if not re.fullmatch(r'[A-Za-z0-9_.:-]+', openai_model):
        raise ValueError('Invalid model identifier')
    package = json.loads((ROOT / 'frontend/package.json').read_text())
    references = model_inventory()
    def model_at(path, prefix):
        matches = {x['model'] for x in references if x['path'] == path and x['model'].startswith(prefix)}
        if len(matches) != 1:
            raise ValueError('Review model inventory: ambiguous or missing model in ' + path)
        return matches.pop()
    tree = ast.parse((ROOT / 'backend/main.py').read_text())
    root_fn = next(n for n in tree.body if isinstance(n, ast.AsyncFunctionDef) and n.name == 'root')
    root_return = next(n for n in ast.walk(root_fn) if isinstance(n, ast.Return))
    backend_version = ast.literal_eval(root_return.value)['version']
    dirty = bool(git('status', '--porcelain', '--untracked-files=normal'))
    return {'schema_version': 1, 'release_id': release_id, 'source_commit': git('rev-parse', 'HEAD'), 'working_tree_dirty': dirty,
            'frontend_version': package['version'], 'backend_root_version': backend_version,
            'models': {'brain_openai_configured': openai_model, 'zoe_chat_hardcoded': model_at('backend/plugins/zoe_agent.py', 'gpt-'), 'faq_chat_hardcoded': model_at('faq-bot-rag/app/rag_chat.py', 'gpt-'), 'judge_default': model_at('backend/evaluators/base.py', 'gpt-'), 'zoe_tts': model_at('backend/agents/zoe/tts_service.py', 'tts-'), 'vector_default': model_at('backend/brain/data_sources/vector_db.py', 'text-embedding-'), 'faq_embedding': 'SDK default; resolve from captured langchain-openai version', 'references': references},
            'prompts': prompt_inventory(), 'dependencies': dependency_inventory(),
            'limitations': ['Model aliases do not pin a provider snapshot.', 'Source references include optional providers and defaults, not only active models.', 'Python requirements and container tags are not immutable locks; archive installed versions and image digests.', 'No backend commit endpoint exists; verify backend artifact digest in deployment evidence.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release-id', required=True)
    parser.add_argument('--openai-model', default='gpt-4o-mini', help='Actual deployment OPENAI_MODEL, or application default')
    parser.add_argument('--output-dir', type=Path, default=ROOT / '.release')
    parser.add_argument('--publish-frontend', action='store_true', help='Write public/release.json BEFORE next build')
    parser.add_argument('--require-clean', action='store_true')
    args = parser.parse_args()
    try:
        data = build(args.release_id, args.openai_model)
        if args.require_clean and data['working_tree_dirty']:
            raise ValueError('Release requires a committed, clean working tree')
        write_json(args.output_dir / 'manifest.json', data)
        lines = ['# Release ' + args.release_id, '', 'Source commit: `' + data['source_commit'] + '`', '', 'Working tree dirty: ' + str(data['working_tree_dirty']), '', '## Models', '', '```json', json.dumps({k:v for k,v in data['models'].items() if k != 'references'}, indent=2), '```', '', '## Prompt versions', '', 'Whole source files are versioned conservatively; non-prompt edits may also change a version.', '']
        lines += ['- `' + p['path'] + '`: `' + p['version'] + '`' for p in data['prompts']]
        lines += ['', '## Dependencies', '', 'Python declarations (not installed-version locks):', '']
        for name, info in data['dependencies'].items():
            if 'declared' in info:
                lines += ['### ' + name, '', '```text', '\n'.join(info['declared']), '```', '']
        front = data['dependencies']['frontend']
        lines += ['### Frontend', '', '| Package | Declared | Lockfile root resolution |', '| --- | --- | --- |']
        for name, version in {**front['dependencies'], **front['devDependencies']}.items():
            lines.append('| ' + name + ' | ' + version + ' | ' + str(front['resolved'].get('node_modules/' + name, 'not resolved')) + ' |')
        lines += ['', '### Containers', '', *['- `' + x + '` (tag; record deployed digest)' for x in data['dependencies']['docker-compose.yml']['images']], '', '## Limitations', '', *['- ' + x for x in data['limitations']], '', 'Complete transitive frontend versions and file hashes: manifest.json. Deployment and rollback sign-off: docs/release/RELEASE_RECORD.md.']
        (args.output_dir / 'release.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        if args.publish_frontend:
            write_json(ROOT / 'frontend/public/release.json', {k:data[k] for k in ('schema_version','release_id','source_commit','working_tree_dirty')})
        print('Generated manifest.json and release.md; dirty=%s' % data['working_tree_dirty'])
        return 0
    except (ValueError, OSError) as exc:
        print('FAIL: ' + str(exc))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())

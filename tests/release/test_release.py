"""Release gates: realistic bad responses, secret handling and source provenance."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts'))
from release_common import digest, env_references, prompt_inventory, read_env
from generate_release_manifest import build, dependency_inventory
from validate_release_env import changes, validate
from post_release_smoke import check, checks, main as smoke_main

CONTRACT = json.loads((ROOT / 'docs/release/environment-contract.json').read_text())


class EnvironmentTests(unittest.TestCase):
    def test_source_contract_complete(self):
        for scope, refs in env_references().items():
            self.assertFalse(set(refs) - set(CONTRACT[scope]))

    def test_required_key_and_placeholder_fail_without_value(self):
        self.assertTrue(validate('backend', {}, CONTRACT)[0])
        secret = 'your_openai_api_key_here'
        errors, _ = validate('backend', {'OPENAI_API_KEY': secret}, CONTRACT)
        self.assertTrue(errors)
        self.assertNotIn(secret, str(errors))

    def test_invalid_numbers(self):
        for value in ('', 'nan', 'inf', '-1', '2.5', 'oops'):
            self.assertTrue(validate('backend', {'OPENAI_API_KEY':'private-test-value', 'OPENAI_TEMPERATURE':value}, CONTRACT)[0])

    def test_legacy_settings_warn(self):
        errors, warnings = validate('backend', {'OPENAI_API_KEY':'private-test-value', 'PORT':'9000'}, CONTRACT)
        self.assertFalse(errors)
        self.assertTrue(any('PORT' in x for x in warnings))

    def test_credential_pairs_and_public_secret(self):
        self.assertTrue(validate('backend', {'OPENAI_API_KEY':'private-test-value','LANGFUSE_PUBLIC_KEY':'pk-example'}, CONTRACT)[0])
        self.assertTrue(validate('frontend', {'NEXT_PUBLIC_API_KEY':'private-test-value'}, CONTRACT)[0])

    def test_diff_reports_names_only(self):
        result = changes({'SECRET':'private-old','REMOVED':'x'}, {'SECRET':'private-new','ADDED':'x'})
        self.assertEqual(result, {'added':['ADDED'],'removed':['REMOVED'],'changed':['SECRET']})
        self.assertNotIn('private', json.dumps(result))

    def test_dotenv_parsing_and_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'test.env'
            p.write_text('export A="a#b" # comment\nB=x # comment\nC=\'xyz\'\n')
            self.assertEqual(read_env(p), {'A':'a#b','B':'x','C':'xyz'})
            for invalid in ('A=x\nA=y', 'A="not closed', 'A=${SECRET}', 'invalid-secret-line'):
                p.write_text(invalid)
                with self.assertRaises(ValueError) as caught:
                    read_env(p)
                self.assertNotIn(invalid, str(caught.exception))

    def test_cli_missing_files_fail(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/validate_release_env.py'), '--backend-env', d + '/missing', '--frontend-env', d + '/missing', '--report', d + '/report.json'], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1)
            self.assertFalse(json.loads(Path(d + '/report.json').read_text())['passed'])

    def test_cli_matching_env_and_changed_secret(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p/'backend').write_text('OPENAI_API_KEY=private-new\n')
            (p/'previous').write_text('OPENAI_API_KEY=private-old\n')
            (p/'frontend').write_text('NEXT_PUBLIC_API_URL=http://localhost:8000\nNEXT_PUBLIC_KEYCLOAK_URL=http://localhost:8080\nNEXT_PUBLIC_KEYCLOAK_REALM=thinklife\nNEXT_PUBLIC_KEYCLOAK_CLIENT_ID=thinklife-frontend\n')
            args = [sys.executable, str(ROOT/'scripts/validate_release_env.py'), '--backend-env', str(p/'backend'), '--frontend-env', str(p/'frontend'), '--previous-backend-env', str(p/'previous'), '--previous-frontend-env', str(p/'frontend'), '--report', str(p/'report'), '--strict']
            result = subprocess.run(args, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            report = (p/'report').read_text()
            self.assertNotIn('private-', report + result.stdout + result.stderr)
            self.assertEqual(json.loads(report)['changes']['backend']['changed'], ['OPENAI_API_KEY'])
            (p/'frontend').write_text((p/'frontend').read_text().replace('REALM=thinklife','REALM=wrong'))
            self.assertEqual(subprocess.run(args, capture_output=True).returncode, 1)


class ManifestTests(unittest.TestCase):
    def test_models_separate_and_dependency_coverage(self):
        m = build('test-release','gpt-configured-test')
        self.assertEqual(m['models']['brain_openai_configured'],'gpt-configured-test')
        self.assertEqual(m['models']['zoe_chat_hardcoded'],'gpt-4o-mini')
        self.assertEqual(len(m['source_commit']),40)
        for name in ('requirements.txt','backend/requirements.txt','faq-bot-rag/requirements.txt','frontend','docker-compose.yml'):
            self.assertIn(name,m['dependencies'])
        self.assertFalse(m['dependencies']['backend/requirements.txt']['fully_pinned'])
        self.assertIn('node_modules/next',m['dependencies']['frontend']['resolved'])

    def test_prompt_versions_use_actual_content(self):
        entries = {x['path']:x for x in prompt_inventory()}
        for p in ('backend/agents/zoe/zoe_core.py','backend/brain/cortex/workflow_engine.py','faq-bot-rag/app/rag_chat.py','backend/brain/guardrails/fallback_handler.py'):
            self.assertEqual(entries[p]['version'], 'sha256:' + digest(ROOT/p))

    def test_generator_works_outside_repo_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            result = subprocess.run([sys.executable,str(ROOT/'scripts/generate_release_manifest.py'),'--release-id','test','--output-dir',d],cwd=d,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertTrue((Path(d)/'release.md').is_file())


class SmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                cls.calls += 1
                status, body = cls.routes.get(self.path,(404,{}))
                if self.path == '/retry' and cls.calls == 1:
                    status = 503
                self.send_response(status)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(body if isinstance(body,bytes) else json.dumps(body).encode())
            def log_message(self,*args): pass
        cls.server = ThreadingHTTPServer(('127.0.0.1',0),Handler)
        cls.base = 'http://127.0.0.1:' + str(cls.server.server_port)
        cls.thread = threading.Thread(target=cls.server.serve_forever,daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join()

    def setUp(self):
        self.__class__.routes = {}
        self.__class__.calls = 0

    def test_200_unhealthy_or_html_or_array_fails(self):
        for body in ({'status':'unhealthy'},b'<html>Sign in</html>',[]):
            self.__class__.routes = {'/health':(200,body)}
            self.assertFalse(check(self.base+'/health',lambda d:d.get('status')=='healthy',1)['passed'])

    def test_server_error_and_redirect_fail(self):
        for status in (500,302,401):
            self.__class__.routes = {'/health':(status,{'detail':'private-secret'})}
            result = check(self.base+'/health',lambda d:True,1)
            self.assertFalse(result['passed'])
            self.assertNotIn('private-secret',str(result))

    def test_transient_failure_retries(self):
        self.__class__.routes = {'/retry':(200,{'status':'healthy'})}
        result = check(self.base+'/retry',lambda d:d.get('status')=='healthy',2,2,0)
        self.assertTrue(result['passed'])
        self.assertEqual(result['attempts'],2)

    def test_version_mismatch_and_dirty_fail(self):
        predicate = checks(self.base,self.base,self.base,'thinklife','a'*40,'rc1')[1][2]
        self.assertTrue(predicate({'source_commit':'a'*40,'release_id':'rc1','working_tree_dirty':False}))
        self.assertFalse(predicate({'source_commit':'b'*40,'release_id':'rc1','working_tree_dirty':False}))
        self.assertFalse(predicate({'source_commit':'a'*40,'release_id':'rc1','working_tree_dirty':True}))

    def test_cli_all_pass_and_brain_failure_exit(self):
        issuer = self.base+'/realms/thinklife'
        self.__class__.routes = {
            '/api/health':(200,{'status':'healthy','service':'ThinkLife Frontend'}),
            '/release.json':(200,{'source_commit':'a'*40,'release_id':'rc1','working_tree_dirty':False}),
            '/':(200,{'message':'ThinkLife Backend','version':'0.7.0'}),
            '/health':(200,{'status':'healthy','brain':True,'zoe':True}),
            '/api/brain/health':(200,{'status':'healthy','brain_status':{'overall':'healthy'}}),
            '/api/zoe/health':(200,{'service':'zoe','initialized':True,'zoe_core_available':True,'zoe_agent_available':True,'zoe_agent_health':{'status':'healthy'}}),
            '/realms/thinklife/.well-known/openid-configuration':(200,{'issuer':issuer, 'authorization_endpoint':issuer+'/auth','token_endpoint':issuer+'/token','jwks_uri':issuer+'/keys'})}
        with tempfile.TemporaryDirectory() as d:
            args = ['smoke','--frontend-url',self.base,'--backend-url',self.base,'--keycloak-url',self.base,'--expected-commit','a'*40,'--release-id','rc1','--attempts','1','--report',d+'/report']
            with patch.object(sys,'argv',args), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(smoke_main(),0)
                self.__class__.routes['/api/brain/health'] = (500,{'detail':'missing method'})
                self.assertEqual(smoke_main(),1)
            report=json.loads(Path(d+'/report').read_text())
            self.assertEqual(len(report['checks']),7)
            self.assertFalse(report['passed'])


if __name__ == '__main__':
    unittest.main()

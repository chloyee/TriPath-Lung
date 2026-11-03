"""Scan source files, optional ZIP snapshots and Git history; print locations only."""
import argparse
import os
from pathlib import Path
import re
import subprocess
import zipfile


RULES = {
    'private-key': rb'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----',
    'github-token': rb'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})',
    'openai-key': rb'\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_-]{20,}',
    'huggingface-token': rb'\bhf_[A-Za-z0-9]{25,}',
    'aws-access-key': rb'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b',
    'slack-token': rb'\bxox[baprs]-[A-Za-z0-9-]{20,}',
    'google-api-key': rb'\bAIza[A-Za-z0-9_-]{35}',
    'credential-url': rb'https?://[^\s/:@]+:[^\s/@]+@',
    'credential-literal': rb'''(?i)["']?(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password)["']?\s*[:=]\s*["']([^"'\r\n]{12,})["']''',
}
EXCLUDE = {'.git', '.venv', 'venv', '__pycache__', '.pytest_cache', '.mypy_cache',
           '.ruff_cache', 'build', 'dist', 'data', 'features', 'outputs', 'logs'}


def scan_bytes(name, data):
    found = []
    for rule, pattern in RULES.items():
        for match in re.finditer(pattern, data):
            if rule == 'credential-literal' and match.group(1).lower() in {
                    b'your_api_key_here', b'your_token_here', b'replace_with_token'}:
                continue
            line = data[:match.start()].count(b'\n') + 1
            found.append(f'{name}:{line}: {rule}')
    return found


def git(root, *args):
    return subprocess.check_output(['git', '-C', str(root), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive', type=Path, action='append', default=[])
    parser.add_argument('--history', action='store_true')
    args = parser.parse_args()
    findings, scanned = [], 0
    for directory, names, files in os.walk(args.root):
        names[:] = [n for n in names if n not in EXCLUDE and not n.endswith('.egg-info')]
        for name in files:
            path = Path(directory)/name
            if path.is_symlink():
                continue
            findings.extend(scan_bytes(str(path.relative_to(args.root)), path.read_bytes()))
            scanned += 1
    for archive in args.archive:
        with zipfile.ZipFile(archive) as bundle:
            for item in bundle.infolist():
                if not item.is_dir():
                    findings.extend(scan_bytes(f'{archive.name}!{item.filename}', bundle.read(item)))
                    scanned += 1
    if args.history:
        objects = git(args.root, 'rev-list', '--objects', '--all').splitlines()
        for entry in objects:
            oid, _, path = entry.partition(b' ')
            kind = git(args.root, 'cat-file', '-t', oid.decode()).strip()
            if kind in {b'blob', b'commit', b'tag'}:
                content = git(args.root, 'cat-file', '-p', oid.decode())
                label = f'history:{oid.decode()}:{path.decode(errors="replace")}'
                findings.extend(scan_bytes(label, content))
                scanned += 1
    for finding in findings:
        print(finding)
    print(f'Scanned {scanned} source/history/archive items; {len(findings)} credential-pattern matches.')
    raise SystemExit(bool(findings))


if __name__ == '__main__':
    main()

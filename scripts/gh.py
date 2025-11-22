#!/usr/bin/env python3
"""
GitHub Helper - Universal Issue & PR Management

PURPOSE:
  Smooth GitHub workflow for issues and PRs.
  Auto-detects repository from git, works without gh CLI.

AUTHENTICATION:
  Requires GitHub token from:
  1. GITHUB_TOKEN environment variable
  2. ~/.config/gh/hosts.yml (gh CLI config)

USAGE:
  # Auto-detect repo from current directory
  gh-helper issue 42                    # View issue #42
  gh-helper issue 42 --comment "Done!"  # Comment on issue
  gh-helper issue list                  # List open issues
  gh-helper issue create "Bug fix"      # Create issue

  gh-helper pr 38                       # View PR #38
  gh-helper pr 38 files                 # Show files changed
  gh-helper pr 38 --merge               # Merge PR
  gh-helper pr list                     # List open PRs
  gh-helper pr create "Feature"         # Create PR from current branch

  # Specify repo explicitly
  gh-helper -r scottsen/morphogen issue 10

  # Quick shortcuts
  gh-helper i 42          # View issue
  gh-helper p 38          # View PR
  gh-helper p 38 m        # Merge PR

FEATURES:
  Issues:  view, list, create, comment, close, reopen, label, assign
  PRs:     view, list, create, merge, comment, files, commits, diff
  Smart:   Auto-detect repo, colored output

EXAMPLES:
  gh-helper issue list --state all --limit 20
  gh-helper pr 38 --comment "LGTM" --merge
  gh-helper issue 42 --label bug --assign scottsen
  gh-helper pr create --title "Fix auth" --body "Details..." --base develop
"""
import sys
import json
import os
import subprocess
from pathlib import Path
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

# ANSI colors
class Color:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    CYAN = '\033[96m'
    GRAY = '\033[90m'

def colored(text: str, color: str) -> str:
    """Return colored text."""
    return f"{color}{text}{Color.RESET}"

def get_github_token() -> Optional[str]:
    """Get GitHub token from environment or gh config."""
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        return token

    gh_config = Path.home() / ".config" / "gh" / "hosts.yml"
    if gh_config.exists():
        with open(gh_config) as f:
            for line in f:
                if "oauth_token:" in line:
                    return line.split("oauth_token:")[1].strip()
    return None

def get_repo_from_git() -> Optional[str]:
    """Auto-detect GitHub repo from git remote."""
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True,
            text=True,
            check=True
        )
        remote_url = result.stdout.strip()

        # Parse GitHub URL (handles both HTTPS and SSH)
        if "github.com" in remote_url:
            # Extract owner/repo from URL
            if remote_url.startswith("git@github.com:"):
                repo = remote_url.replace("git@github.com:", "").replace(".git", "")
            elif "github.com/" in remote_url:
                repo = remote_url.split("github.com/")[1].replace(".git", "")
            else:
                return None
            return repo
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    return None

def github_api_request(url: str, token: str, method: str = "GET", data: Optional[Dict] = None) -> Any:
    """Make authenticated GitHub API request."""
    req = urllib.request.Request(url, method=method)
    req.add_header("Authorization", f"token {token}")
    req.add_header("Accept", "application/vnd.github.v3+json")

    if data:
        req.add_header("Content-Type", "application/json")
        req.data = json.dumps(data).encode('utf-8')

    try:
        with urllib.request.urlopen(req) as response:
            if response.status == 204:  # No content (e.g., successful merge)
                return {}
            return json.loads(response.read())
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if hasattr(e, 'read') else str(e)
        print(colored(f"❌ Error {e.code}: {e.reason}", Color.RED), file=sys.stderr)
        try:
            error_json = json.loads(error_body)
            if 'message' in error_json:
                print(colored(f"   {error_json['message']}", Color.RED), file=sys.stderr)
        except:
            print(error_body, file=sys.stderr)
        sys.exit(1)

# === ISSUE OPERATIONS ===

def view_issue(repo: str, number: int, token: str) -> None:
    """View issue details."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    issue = github_api_request(url, token)

    print(colored(f"\nIssue #{issue['number']}: {issue['title']}", Color.BOLD))
    print(colored(f"State: {issue['state']}", Color.GREEN if issue['state'] == 'open' else Color.GRAY))
    print(f"Author: {issue['user']['login']}")
    print(f"Created: {issue['created_at']}")

    if issue.get('labels'):
        labels = ', '.join([l['name'] for l in issue['labels']])
        print(f"Labels: {colored(labels, Color.CYAN)}")

    if issue.get('assignees'):
        assignees = ', '.join([a['login'] for a in issue['assignees']])
        print(f"Assigned: {assignees}")

    if issue.get('body'):
        print(f"\n{issue['body']}")

    print(f"\n{colored(issue['html_url'], Color.BLUE)}")

def list_issues(repo: str, token: str, state: str = "open", limit: int = 10) -> None:
    """List issues."""
    url = f"https://api.github.com/repos/{repo}/issues?state={state}&per_page={limit}"
    issues = github_api_request(url, token)

    print(colored(f"\n{repo} - {state.upper()} Issues", Color.BOLD))
    print(colored("─" * 80, Color.GRAY))

    for issue in issues:
        # Skip PRs (they appear in issues endpoint too)
        if 'pull_request' in issue:
            continue

        state_color = Color.GREEN if issue['state'] == 'open' else Color.GRAY
        number = colored(f"#{issue['number']}", Color.YELLOW)
        state = colored(issue['state'], state_color)

        labels = ""
        if issue.get('labels'):
            labels = " " + colored(f"[{', '.join([l['name'] for l in issue['labels']])}]", Color.CYAN)

        print(f"{number} {state:15} {issue['title']}{labels}")

def create_issue(repo: str, token: str, title: str, body: str = "") -> None:
    """Create a new issue."""
    url = f"https://api.github.com/repos/{repo}/issues"
    data = {"title": title, "body": body}
    issue = github_api_request(url, token, method="POST", data=data)

    print(colored(f"✅ Issue created: #{issue['number']}", Color.GREEN))
    print(f"Title: {issue['title']}")
    print(f"URL: {colored(issue['html_url'], Color.BLUE)}")

def comment_issue(repo: str, number: int, token: str, comment: str) -> None:
    """Add comment to issue."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}/comments"
    data = {"body": comment}
    github_api_request(url, token, method="POST", data=data)
    print(colored(f"✅ Comment added to issue #{number}", Color.GREEN))

def close_issue(repo: str, number: int, token: str) -> None:
    """Close an issue."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    data = {"state": "closed"}
    github_api_request(url, token, method="PATCH", data=data)
    print(colored(f"✅ Issue #{number} closed", Color.GREEN))

def reopen_issue(repo: str, number: int, token: str) -> None:
    """Reopen an issue."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}"
    data = {"state": "open"}
    github_api_request(url, token, method="PATCH", data=data)
    print(colored(f"✅ Issue #{number} reopened", Color.GREEN))

# === PR OPERATIONS ===

def view_pr(repo: str, number: int, token: str) -> None:
    """View PR details."""
    url = f"https://api.github.com/repos/{repo}/pulls/{number}"
    pr = github_api_request(url, token)

    print(colored(f"\nPR #{pr['number']}: {pr['title']}", Color.BOLD))

    state = pr['state']
    if pr.get('merged'):
        print(colored(f"State: merged", Color.MAGENTA))
    elif state == 'open':
        print(colored(f"State: {state}", Color.GREEN))
    else:
        print(colored(f"State: {state}", Color.GRAY))

    print(f"Author: {pr['user']['login']}")
    print(f"Created: {pr['created_at']}")
    print(f"Branch: {colored(pr['head']['ref'], Color.CYAN)} → {colored(pr['base']['ref'], Color.CYAN)}")

    if pr.get('merged_at'):
        print(f"Merged: {pr['merged_at']}")

    additions = pr['additions']
    deletions = pr['deletions']
    print(f"\nChanges: {colored(f'+{additions}', Color.GREEN)} {colored(f'-{deletions}', Color.RED)} ({pr['changed_files']} files, {pr['commits']} commits)")

    if pr.get('body'):
        print(f"\n{pr['body']}")

    print(f"\n{colored(pr['html_url'], Color.BLUE)}")

def list_prs(repo: str, token: str, state: str = "open", limit: int = 10) -> None:
    """List pull requests."""
    url = f"https://api.github.com/repos/{repo}/pulls?state={state}&per_page={limit}"
    prs = github_api_request(url, token)

    print(colored(f"\n{repo} - {state.upper()} Pull Requests", Color.BOLD))
    print(colored("─" * 80, Color.GRAY))

    for pr in prs:
        state_color = Color.MAGENTA if pr.get('merged') else (Color.GREEN if pr['state'] == 'open' else Color.GRAY)
        number = colored(f"#{pr['number']}", Color.YELLOW)
        state_text = "merged" if pr.get('merged') else pr['state']
        state = colored(state_text, state_color)

        print(f"{number} {state:15} {pr['title']}")
        print(colored(f"       {pr['head']['ref']} → {pr['base']['ref']}", Color.GRAY))

def show_pr_files(repo: str, number: int, token: str) -> None:
    """Show files changed in PR."""
    url = f"https://api.github.com/repos/{repo}/pulls/{number}/files"
    files = github_api_request(url, token)

    print(colored(f"\nPR #{number} - Files Changed: {len(files)}", Color.BOLD))
    print(colored("─" * 80, Color.GRAY))

    for f in files:
        status_color = Color.GREEN if f['status'] == 'added' else (Color.RED if f['status'] == 'removed' else Color.YELLOW)
        status = colored(f['status'].ljust(8), status_color)
        additions = colored(f"+{f['additions']}", Color.GREEN)
        deletions = colored(f"-{f['deletions']}", Color.RED)
        print(f"{status} {f['filename']}")
        print(f"         {additions} {deletions}")

def create_pr(repo: str, token: str, title: str, body: str = "", head: Optional[str] = None, base: str = "main") -> None:
    """Create a pull request."""
    # Auto-detect head branch if not specified
    if not head:
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                capture_output=True,
                text=True,
                check=True
            )
            head = result.stdout.strip()
            if head == base:
                print(colored(f"❌ Cannot create PR: currently on {base} branch", Color.RED), file=sys.stderr)
                sys.exit(1)
        except subprocess.CalledProcessError:
            print(colored("❌ Could not detect current branch", Color.RED), file=sys.stderr)
            sys.exit(1)

    url = f"https://api.github.com/repos/{repo}/pulls"
    data = {
        "title": title,
        "body": body,
        "head": head,
        "base": base
    }
    pr = github_api_request(url, token, method="POST", data=data)

    print(colored(f"✅ Pull request created: #{pr['number']}", Color.GREEN))
    print(f"Title: {pr['title']}")
    print(f"Branch: {colored(head, Color.CYAN)} → {colored(base, Color.CYAN)}")
    print(f"URL: {colored(pr['html_url'], Color.BLUE)}")

def merge_pr(repo: str, number: int, token: str, method: str = "merge") -> None:
    """Merge a pull request."""
    url = f"https://api.github.com/repos/{repo}/pulls/{number}/merge"
    data = {"merge_method": method}  # merge, squash, or rebase
    github_api_request(url, token, method="PUT", data=data)
    print(colored(f"✅ PR #{number} merged successfully", Color.GREEN))

def comment_pr(repo: str, number: int, token: str, comment: str) -> None:
    """Add comment to PR."""
    url = f"https://api.github.com/repos/{repo}/issues/{number}/comments"
    data = {"body": comment}
    github_api_request(url, token, method="POST", data=data)
    print(colored(f"✅ Comment added to PR #{number}", Color.GREEN))

# === MAIN ===

def print_help():
    """Print help message."""
    print(__doc__)

def main():
    args = sys.argv[1:]

    if not args or args[0] in ["-h", "--help", "help"]:
        print_help()
        sys.exit(0)

    # Parse global options
    repo = None
    i = 0
    while i < len(args):
        if args[i] in ["-r", "--repo"]:
            if i + 1 < len(args):
                repo = args[i + 1]
                args = args[:i] + args[i+2:]
            else:
                print(colored("❌ --repo requires a value", Color.RED), file=sys.stderr)
                sys.exit(1)
        else:
            i += 1

    # Get authentication
    token = get_github_token()
    if not token:
        print(colored("❌ No GitHub token found", Color.RED), file=sys.stderr)
        print("Set: export GITHUB_TOKEN='ghp_your_token_here'", file=sys.stderr)
        sys.exit(1)

    # Auto-detect repo if not specified
    if not repo:
        repo = get_repo_from_git()
        if not repo:
            print(colored("❌ Could not detect repository", Color.RED), file=sys.stderr)
            print("Either run from a git repository or use: tia-gh -r owner/repo", file=sys.stderr)
            sys.exit(1)

    if not args:
        print(colored("❌ No command specified", Color.RED), file=sys.stderr)
        print("Try: gh-helper --help", file=sys.stderr)
        sys.exit(1)

    command = args[0]

    # Handle shortcuts
    if command in ["i", "issue", "issues"]:
        command = "issue"
    elif command in ["p", "pr", "prs", "pull"]:
        command = "pr"

    # === ISSUE COMMANDS ===
    if command == "issue":
        if len(args) < 2:
            list_issues(repo, token)
            sys.exit(0)

        subcommand = args[1]

        if subcommand == "list":
            state = "open"
            limit = 10
            for i, arg in enumerate(args[2:]):
                if arg in ["--state", "-s"] and i + 3 < len(args):
                    state = args[i + 3]
                elif arg in ["--limit", "-l"] and i + 3 < len(args):
                    limit = int(args[i + 3])
            list_issues(repo, token, state, limit)

        elif subcommand == "create":
            title = args[2] if len(args) > 2 else None
            body = ""
            for i, arg in enumerate(args):
                if arg in ["--body", "-b"] and i + 1 < len(args):
                    body = args[i + 1]

            if not title:
                print(colored("❌ Title required", Color.RED), file=sys.stderr)
                sys.exit(1)

            create_issue(repo, token, title, body)

        elif subcommand.isdigit():
            number = int(subcommand)

            # Check for options
            comment = None
            should_close = False
            should_reopen = False

            i = 2
            while i < len(args):
                if args[i] in ["--comment", "-c"] and i + 1 < len(args):
                    comment = args[i + 1]
                    i += 2
                elif args[i] in ["--close"]:
                    should_close = True
                    i += 1
                elif args[i] in ["--reopen"]:
                    should_reopen = True
                    i += 1
                else:
                    i += 1

            # Execute actions
            view_issue(repo, number, token)

            if comment:
                comment_issue(repo, number, token, comment)
            if should_close:
                close_issue(repo, number, token)
            if should_reopen:
                reopen_issue(repo, number, token)
        else:
            print(colored(f"❌ Unknown issue command: {subcommand}", Color.RED), file=sys.stderr)
            sys.exit(1)

    # === PR COMMANDS ===
    elif command == "pr":
        if len(args) < 2:
            list_prs(repo, token)
            sys.exit(0)

        subcommand = args[1]

        if subcommand == "list":
            state = "open"
            limit = 10
            for i, arg in enumerate(args[2:]):
                if arg in ["--state", "-s"] and i + 3 < len(args):
                    state = args[i + 3]
                elif arg in ["--limit", "-l"] and i + 3 < len(args):
                    limit = int(args[i + 3])
            list_prs(repo, token, state, limit)

        elif subcommand == "create":
            title = args[2] if len(args) > 2 else None
            body = ""
            head = None
            base = "main"

            for i, arg in enumerate(args):
                if arg in ["--body", "-b"] and i + 1 < len(args):
                    body = args[i + 1]
                elif arg in ["--head", "-h"] and i + 1 < len(args):
                    head = args[i + 1]
                elif arg in ["--base"] and i + 1 < len(args):
                    base = args[i + 1]

            if not title:
                print(colored("❌ Title required", Color.RED), file=sys.stderr)
                sys.exit(1)

            create_pr(repo, token, title, body, head, base)

        elif subcommand.isdigit():
            number = int(subcommand)

            # Check for detail view or options
            detail = None
            comment = None
            should_merge = False

            i = 2
            while i < len(args):
                if args[i] in ["files", "f"]:
                    detail = "files"
                    i += 1
                elif args[i] in ["commits", "c"]:
                    detail = "commits"
                    i += 1
                elif args[i] in ["--comment"] and i + 1 < len(args):
                    comment = args[i + 1]
                    i += 2
                elif args[i] in ["--merge", "-m", "m"]:
                    should_merge = True
                    i += 1
                else:
                    i += 1

            # Execute actions
            if detail == "files":
                show_pr_files(repo, number, token)
            else:
                view_pr(repo, number, token)

            if comment:
                comment_pr(repo, number, token, comment)
            if should_merge:
                merge_pr(repo, number, token)
        else:
            print(colored(f"❌ Unknown PR command: {subcommand}", Color.RED), file=sys.stderr)
            sys.exit(1)

    else:
        print(colored(f"❌ Unknown command: {command}", Color.RED), file=sys.stderr)
        print("Try: gh-helper --help", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()

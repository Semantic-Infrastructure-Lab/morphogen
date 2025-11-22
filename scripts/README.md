# Scripts & Tools

Developer tools and utilities for the Morphogen project.

## Available Tools

### reveal.sh + reveal.py - Progressive File Explorer

**Purpose**: Explore large files incrementally at 4 levels of detail to manage token usage and understand structure before reading full content.

**Components**:
- `reveal.py` - Local Python implementation (no installation required)
- `reveal.sh` - Wrapper script (tries external reveal, falls back to reveal.py)

**No Installation Needed**: The local `reveal.py` works out of the box with Python 3.

**Optional External Tool**:
```bash
# Only if you want the full-featured gist version
pip install git+https://gist.github.com/scottsen/ee3fff354a79032f1c6d9d46991c8400
```

**Usage**:
```bash
# Level 0: Metadata (filename, size, type, line count, hash)
./scripts/reveal.sh 0 morphogen/domains/audio.py

# Level 1: Structure (imports, classes, functions for .py; headings for .md)
./scripts/reveal.sh 1 morphogen/domains/audio.py

# Level 2: Preview (representative sample with beginning/middle/end)
./scripts/reveal.sh 2 docs/specifications/chemistry.md

# Level 3: Full content (with line numbers)
./scripts/reveal.sh 3 SPECIFICATION.md

# Or use reveal.py directly
python3 scripts/reveal.py --level 1 README.md
```

**When to Use**:
- Before reading large files (1000+ lines)
- Surveying domain implementations
- Understanding documentation structure
- Exploring test file organization
- Conserving tokens (get 80% info at 20% token cost)
- Creating documentation maps and indexes

**Supported File Types** (reveal.py):
- **Python** (.py): Imports, classes, functions with line numbers
- **Markdown** (.md): Heading hierarchy, code blocks, lists
- **All others**: Generic preview with line numbers

**reveal.py Features**:
- Level 0: Metadata (size, lines, hash)
- Level 1: Structure (imports, classes, functions, headings)
- Level 2: Preview (beginning + middle + end samples)
- Level 3: Full content with line numbers
- No external dependencies (pure Python 3 stdlib)

---

### gh.py - GitHub Issue & PR Manager

**Purpose**: Manage GitHub issues and PRs from the command line without requiring `gh` CLI.

**Installation**:
```bash
# Add to PATH or use directly
alias gh-helper='python /home/user/morphogen/scripts/gh.py'
```

**Authentication**:
Requires GitHub token from:
1. `GITHUB_TOKEN` environment variable
2. `~/.config/gh/hosts.yml` (gh CLI config)

**Usage**:
```bash
# Issues
gh-helper issue 42                     # View issue #42
gh-helper issue list                   # List open issues
gh-helper issue create "Bug fix"       # Create new issue
gh-helper issue 42 --comment "Done!"   # Comment on issue
gh-helper issue 42 --close             # Close issue

# Pull Requests
gh-helper pr 38                        # View PR #38
gh-helper pr list                      # List open PRs
gh-helper pr 38 files                  # Show changed files
gh-helper pr 38 --merge                # Merge PR
gh-helper pr create "Feature"          # Create PR from current branch

# Shortcuts
gh-helper i 42                         # View issue
gh-helper p 38                         # View PR
gh-helper p 38 m                       # Merge PR
```

**Features**:
- Auto-detects repository from git config
- Colored terminal output
- Works without `gh` CLI installed
- Supports all major GitHub operations

---

## Development Workflow

### Exploring the Codebase

1. **Start with structure** (Level 1):
   ```bash
   ./scripts/reveal.sh 1 morphogen/domains/audio.py
   ```

2. **Preview if needed** (Level 2):
   ```bash
   ./scripts/reveal.sh 2 morphogen/domains/audio.py
   ```

3. **Read full content** only when necessary (Level 3):
   ```bash
   ./scripts/reveal.sh 3 morphogen/domains/audio.py
   ```

4. **Explore documentation**:
   ```bash
   # Map all specifications
   for f in docs/specifications/*.md; do
       ./scripts/reveal.sh 1 "$f"
   done

   # See complete documentation index
   ./scripts/reveal.sh 1 docs/DOCUMENTATION_INDEX.md
   ```

### Working with Issues

```bash
# List current issues
gh-helper issue list

# View specific issue
gh-helper issue 123

# Close issue with comment
gh-helper issue 123 --comment "Fixed in PR #125" --close
```

### Creating Pull Requests

```bash
# Create PR from current branch
gh-helper pr create --title "Add reveal tool documentation" \
                 --body "Documents the progressive reveal CLI for codebase exploration"

# Check PR status
gh-helper pr list

# Merge when ready
gh-helper pr 126 --merge
```

---

## Adding New Tools

When adding new developer tools:

1. **Create the script** in `scripts/`
2. **Make it executable**: `chmod +x scripts/your_tool.sh`
3. **Document it here** in this README
4. **Update claude.md** if Claude should know about it
5. **Add tests** if applicable

### Tool Template

```bash
#!/usr/bin/env bash
# Tool Name - Brief Description
#
# Detailed description of what this tool does and why it exists.
#
# Usage:
#   ./scripts/tool_name.sh <args>

set -euo pipefail

# Your implementation here
```

---

## Tips for Claude (and Humans!)

When working with this codebase:

- **Use reveal for exploration**: Check file structure before reading full content
- **Level 0 for quick checks**: File size, line count, type detection
- **Level 1 is often enough**: Imports, classes, and functions give good context
- **Level 2 for sampling**: Get representative content without full file
- **Level 3 sparingly**: Only when you need complete file contents

**Token Conservation Strategy**:
```bash
# Instead of reading 2000-line file directly:
./scripts/reveal.sh 1 large_file.py         # Structure only (~5-10% tokens)
./scripts/reveal.sh 2 large_file.py         # Preview (~20-30% tokens)
# Then read specific sections with regular Read tool if needed
```

**Documentation Exploration**:
```bash
# Map all specifications
for f in docs/specifications/*.md; do
    ./scripts/reveal.sh 1 "$f" | grep -E "(File:|STRUCTURE|headings)"
done

# Find large documents
for f in docs/**/*.md; do
    ./scripts/reveal.sh 0 "$f" 2>/dev/null | grep "Lines:"
done | sort -t: -k2 -n

# Use the comprehensive documentation index
./scripts/reveal.sh 1 docs/DOCUMENTATION_INDEX.md
```

---

**Maintained by**: Morphogen Development Team
**Last Updated**: 2025-11-21

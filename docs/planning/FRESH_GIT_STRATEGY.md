# Fresh Git Strategy: Morphogen Repository Transition

**Status:** Planning Document
**Date:** 2025-11-21
**Purpose:** Document the strategy for transitioning from `kairo` to `morphogen` repository with clean history

---

## Current State Assessment

### Naming Inconsistencies

**What's Correct:**
- ✅ Package name: `morphogen` (in pyproject.toml)
- ✅ Primary documentation: References "Morphogen"
- ✅ Python module structure: `morphogen/`
- ✅ Branding decision: ADR-011 established "Morphogen" as official name

**What Needs Transition:**
- ❌ Git repository: `git@github.com:scottsen/kairo.git`
- ❌ Local directory: `/home/scottsen/src/projects/kairo`
- ❌ Legacy references: "kairo" scattered in examples, tests, comments
- ❌ Git history: Contains pre-rename artifacts and naming confusion

### Why Fresh Git?

1. **Clean identity** - New repository name matches project name
2. **Historical clarity** - No confusion about "kairo" vs "morphogen" in git history
3. **Strategic positioning** - Professional presentation for v1.0 launch
4. **Community onboarding** - New contributors see consistent naming from day one
5. **Release preparation** - PyPI release should point to correctly-named repository

---

## Transition Strategy

### Phase 1: Repository Creation (Day 1)

**1.1 Create New Repository**
```bash
# On GitHub: Create new repo `morphogen`
# URL: git@github.com:scottsen/morphogen.git
# Description: "Universal deterministic computation platform unifying audio, physics, circuits, and optimization"
# Visibility: Public
# Initialize: No README, .gitignore, or license (we'll migrate existing)
```

**1.2 Prepare Clean Export**
```bash
# Current location: /home/scottsen/src/projects/kairo

# Create clean snapshot (exclude git history)
cd /home/scottsen/src/projects/
mkdir morphogen-fresh
rsync -av --exclude='.git' \
          --exclude='__pycache__' \
          --exclude='*.pyc' \
          --exclude='.pytest_cache' \
          --exclude='*.egg-info' \
          --exclude='.venv' \
          morphogen/ morphogen-fresh/
```

**1.3 Final Naming Cleanup**
```bash
cd morphogen-fresh

# Replace remaining "kairo" references
find . -type f \( -name "*.py" -o -name "*.md" -o -name "*.txt" \) \
  -exec sed -i 's/morphogen/morphogen/g' {} +

# Verify changes
grep -r "kairo" . --include="*.py" --include="*.md" | wc -l
# Should be minimal (only intentional references like "hello.kairo" file extension)
```

**1.4 Initialize Fresh Git**
```bash
cd morphogen-fresh
git init
git remote add origin git@github.com:scottsen/morphogen.git

# Initial commit
git add .
git commit -m "chore: Initial commit - Morphogen v0.11.0

Morphogen is a universal, deterministic computation platform that unifies
audio synthesis, physics simulation, circuit design, geometry, and optimization.

Current status:
- 40 production-ready computational domains
- 900+ comprehensive tests (all passing)
- MLIR compilation pipeline complete
- Python runtime with NumPy backend

This repository represents a fresh start with consistent naming throughout.
Previous development history under 'kairo' name preserved separately."

git push -u origin main
```

### Phase 2: Historical Preservation (Day 1)

**2.1 Archive Old Repository**
```bash
# Keep old kairo repo as archive
cd /home/scottsen/src/projects/kairo
git remote rename origin kairo-archive
git remote set-url kairo-archive git@github.com:scottsen/kairo-archive.git

# Add note about transition
echo "# ARCHIVED: This repository has been superseded by morphogen

This repository has been archived. All active development has moved to:

**https://github.com/scottsen/morphogen**

This archive preserves the historical development under the 'kairo' name.
" > ARCHIVE_NOTICE.md

git add ARCHIVE_NOTICE.md
git commit -m "docs: Archive notice - Development moved to morphogen repository"
git push kairo-archive main
```

**2.2 Update Old Repo Settings**
```bash
# On GitHub for scottsen/kairo:
# - Set as archived
# - Update description: "ARCHIVED - Development moved to scottsen/morphogen"
# - Disable issues, PRs, wiki
```

### Phase 3: Local Development Setup (Day 1)

**3.1 Move to New Location**
```bash
# Move fresh repo to permanent location
mv /home/scottsen/src/projects/morphogen-fresh /home/scottsen/src/projects/morphogen

# Update working directory
cd /home/scottsen/src/projects/morphogen
```

**3.2 Verify Installation**
```bash
# Install in editable mode
pip install -e ".[dev,audio,viz]"

# Run tests
pytest tests/ -v

# Verify CLI
morphogen --version
# Should output: morphogen 0.11.0
```

**3.3 Update Development Environment**
```bash
# Update any project management tools or IDE configurations
# From: /home/scottsen/src/projects/kairo
# To:   /home/scottsen/src/projects/morphogen
```

### Phase 4: Documentation Updates (Day 2)

**4.1 Update Critical Docs**
```markdown
Files to review and update:
- README.md - Verify all links point to new repo
- docs/README.md - Update navigation
- CONTRIBUTING.md (if exists) - Update git clone instructions
- claude.md - Update paths and repository references
- docs/planning/* - Update any references to old repo
```

**4.2 Add Migration Note**
```bash
# Create migration history document
touch docs/planning/MIGRATION_FROM_KAIRO.md
```

**4.3 Update External References**
```bash
# Check for hardcoded URLs
grep -r "github.com/scottsen/kairo" . --include="*.md" --include="*.py"

# Replace with new repo URL
find . -type f \( -name "*.md" -o -name "*.py" \) \
  -exec sed -i 's|github.com/scottsen/kairo|github.com/scottsen/morphogen|g' {} +
```

### Phase 5: Release Preparation (Day 3-7)

**5.1 Pre-release Checklist**
- [ ] All tests passing (`pytest tests/ -v`)
- [ ] Documentation consistent
- [ ] GitHub repo properly configured
- [ ] README.md compelling and accurate
- [ ] CHANGELOG.md up to date
- [ ] LICENSE file present
- [ ] pyproject.toml repository URL updated

**5.2 GitHub Repository Setup**
```bash
# Configure branch protection on main
# Require PR reviews for main branch
# Enable GitHub Actions (if using CI/CD)
# Set up GitHub Topics: python, dsl, compiler, audio, physics, mlir
```

**5.3 Tag Initial Release**
```bash
git tag -a v0.11.0 -m "Morphogen v0.11.0 - Fresh repository start

Initial release under unified 'morphogen' naming.

Highlights:
- 40 production-ready computational domains
- 900+ comprehensive tests
- MLIR compilation pipeline
- Cross-domain composition
"
git push origin v0.11.0
```

---

## Risk Mitigation

### Risks and Mitigations

**Risk 1: Breaking existing installations**
- Mitigation: Old repo archived but still accessible
- Mitigation: PyPI package name already `morphogen` (no breaking change)

**Risk 2: Lost contributors/community**
- Mitigation: Redirect notice on old repo
- Mitigation: Communicate transition clearly
- Mitigation: GitHub will auto-redirect old URL → new URL

**Risk 3: Documentation links break**
- Mitigation: Comprehensive grep/replace before migration
- Mitigation: Test all documentation links post-migration

**Risk 4: External integrations break**
- Mitigation: Update project paths in all development tools immediately
- Mitigation: Symlink if needed: `ln -s morphogen kairo`

---

## Post-Migration Checklist

### Week 1
- [ ] Fresh repo created and initialized
- [ ] Old repo archived with redirect notice
- [ ] All tests passing in new repo
- [ ] Documentation links verified
- [ ] Development environment updated
- [ ] Local development workflow confirmed

### Week 2
- [ ] GitHub repository properly branded
- [ ] README compelling and professional
- [ ] CONTRIBUTING guide updated
- [ ] First PR merged on new repo (validates workflow)

### Month 1
- [ ] Community (if any) notified and migrated
- [ ] No critical issues from migration
- [ ] Development velocity maintained
- [ ] Prepare for PyPI release with correct repo URL

---

## Long-term Repository Management

### Branch Strategy
```
main           - Stable, releases tagged here
develop        - Active development (optional, or direct to main)
feature/*      - Feature branches
claude/*       - AI assistant session branches
hotfix/*       - Critical bug fixes
```

### Commit Conventions
```
type(scope): Brief description

Types: feat, fix, docs, refactor, test, chore, perf, ci
Scope: domain name, component, or subsystem

Examples:
feat(audio): Add waveguide synthesis operators
fix(field): Correct boundary handling in diffusion
docs(planning): Add fresh git strategy document
refactor(mlir): Simplify lowering passes
```

### Release Tagging
```
v0.11.0 - Current (fresh repo start)
v0.12.0 - Next minor (feature additions)
v1.0.0  - Major milestone (full v1.0 release plan)

Tag format: vMAJOR.MINOR.PATCH
Always include annotated tag with release notes
```

---

## Communication Plan

### Internal (Development Team)
- Update project tracking systems
- Notify any collaborators
- Update bookmarks and IDE workspaces

### External (Community)
- Blog post or announcement (if applicable)
- Update any documentation sites
- Social media update (if applicable)

### PyPI Release
```toml
# In pyproject.toml, ensure:
[project.urls]
Homepage = "https://github.com/scottsen/morphogen"
Documentation = "https://github.com/scottsen/morphogen/tree/main/docs"
Repository = "https://github.com/scottsen/morphogen"
"Bug Tracker" = "https://github.com/scottsen/morphogen/issues"
```

---

## Success Criteria

### Technical Success
- [ ] All tests pass in new repository
- [ ] Clean `git log` with consistent naming
- [ ] No `kairo` references except intentional (file extensions)
- [ ] Documentation fully consistent

### Process Success
- [ ] Migration completed within 1 week
- [ ] Zero downtime for active development
- [ ] All integrations (CI/CD) functional
- [ ] Development velocity maintained

### Strategic Success
- [ ] Professional repository presentation
- [ ] Clear identity for v1.0 launch
- [ ] New contributors face no naming confusion
- [ ] Ready for PyPI release with correct branding

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| Phase 1: Repository Creation | Day 1 | New repo initialized, clean export |
| Phase 2: Historical Preservation | Day 1 | Old repo archived |
| Phase 3: Local Setup | Day 1 | Development environment functional |
| Phase 4: Documentation | Day 2 | All docs updated and verified |
| Phase 5: Release Prep | Day 3-7 | Repository professionally configured |
| **Total** | **1 week** | **Fresh, professional repository** |

---

## Next Steps

1. **Review this strategy** with team/stakeholders
2. **Choose migration date** (low-activity period preferred)
3. **Execute Phase 1-3** in single session (Day 1)
4. **Validate thoroughly** before announcing
5. **Proceed with Phase 4-5** methodically

---

**Document Status:** Ready for Execution
**Approval Required:** Repository owner (scottsen)
**Estimated Effort:** 1 week (mostly Day 1 for core migration)
**Risk Level:** Low (old repo preserved, package name already correct)

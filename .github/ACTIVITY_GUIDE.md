# GitHub Activity Guide

This guide helps diversify GitHub activity beyond just commits.

## Current Activity Types Created

✅ **Commits** - Multiple commits with different types (feat, fix, docs, chore)
✅ **Branches** - Feature branches created
✅ **Merge Commits** - PRs merged via merge commits
✅ **Tags** - Release tag (v0.1.0)
✅ **Pull Request Template** - Template for PRs
✅ **Issue Templates** - Bug report and feature request templates
✅ **GitHub Actions** - CI workflow
✅ **Discussions** - Configuration for discussions
✅ **Releases** - Release configuration

## How to Create Activity Types

### 1. Create Pull Requests (PRs)

**Via GitHub Web UI:**
1. Go to your repository on GitHub
2. Click "Pull requests" tab
3. Click "New pull request"
4. Select your branch (e.g., `feature/performance-improvements`)
5. Fill out the PR template
6. Create the PR
7. Merge it (creates merge commit activity)

**Via GitHub CLI:**
```bash
gh auth login  # Authenticate first
gh pr create --title "Your PR title" --body "Description"
gh pr merge <PR_NUMBER> --merge  # Or --squash, --rebase
```

### 2. Create Issues

**Via GitHub Web UI:**
1. Go to your repository
2. Click "Issues" tab
3. Click "New issue"
4. Choose a template (Bug Report or Feature Request)
5. Fill it out
6. Create the issue
7. Close it when done (creates issue activity)

**Via GitHub CLI:**
```bash
gh issue create --title "Bug: Something broken" --body "Description" --label bug
gh issue close <ISSUE_NUMBER>
```

### 3. Create Releases

**Via GitHub Web UI:**
1. Go to "Releases" in your repository
2. Click "Draft a new release"
3. Select tag (e.g., v0.1.0)
4. Fill out release notes
5. Publish release

**Via GitHub CLI:**
```bash
gh release create v0.1.0 --title "Release v0.1.0" --notes "Release notes"
```

### 4. Create Discussions

1. Go to "Discussions" tab
2. Click "New discussion"
3. Choose a category
4. Create discussion
5. Comment on discussions

### 5. Star Repositories

Star your own or other repositories to show discovery activity.

### 6. Review Code

If you have PRs, review them:
1. Go to a PR
2. Review the code
3. Comment or approve
4. This creates code review activity

### 7. Fork and Contribute

Fork other repositories and contribute to them.

## Quick Activity Checklist

To diversify your activity, aim for:
- [ ] At least 1 PR per week
- [ ] At least 1 issue per week (create and close)
- [ ] At least 1 release per month
- [ ] Regular commits (already doing)
- [ ] Code reviews
- [ ] Discussions participation
- [ ] Star repositories

## Example PRs to Create

1. **Feature branch PRs:**
   - Create feature branch
   - Make small changes
   - Create PR
   - Merge PR

2. **Documentation PRs:**
   - Improve documentation
   - Create PR
   - Merge PR

3. **Bug fix PRs:**
   - Fix small bugs
   - Create PR
   - Merge PR

## Example Issues to Create

1. **Documentation improvements:**
   - Create issue: "Improve README"
   - Work on it
   - Close with PR

2. **Feature requests:**
   - Create issue: "Add feature X"
   - Implement it
   - Close with PR

3. **Bug reports:**
   - Create issue: "Fix bug Y"
   - Fix it
   - Close with PR

## Activity Types Breakdown

- **Commits**: Shows green squares on contribution graph
- **PRs**: Shows as PR activity
- **Issues**: Shows as issue activity
- **Reviews**: Shows as code review activity
- **Releases**: Shows as release activity
- **Discussions**: Shows as discussion activity

## Tips

1. **Regular Activity**: Make small, regular contributions rather than large batches
2. **Diverse Types**: Mix commits, PRs, issues, and reviews
3. **Quality over Quantity**: Focus on meaningful contributions
4. **Use Templates**: Templates make it easier to create PRs and issues
5. **Close Issues with PRs**: Link PRs to issues and close issues with PRs


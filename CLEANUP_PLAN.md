# Nanobrain Framework Cleanup Plan

## 🎯 Executive Summary

This plan addresses the comprehensive cleanup of the Nanobrain HPC agentic AI framework, focusing on:
1. **Repository structure cleanup** - Remove bloat, organize properly
2. **Demo consolidation** - Fix viral_pssm and rag_database_creation, remove broken demos
3. **Core framework stabilization** - Fix imports, dependencies, and hardcoded paths
4. **Documentation and packaging** - Proper README, setup, and deployment guides

## 🚨 Phase 1: Backup and Safety (MANDATORY FIRST STEP)

### 1.1 Create Full Backup
```bash
# Create timestamped backup
cp -r . ../nanobrain_backup_$(date +%Y%m%d_%H%M%S)

# Create git backup branch
git checkout -b backup_before_cleanup_$(date +%Y%m%d)
git add -A
git commit -m "Full backup before comprehensive cleanup"
git push origin backup_before_cleanup_$(date +%Y%m%d)
```

### 1.2 Document Current State
- [ ] Inventory all working demos
- [ ] Document external dependencies
- [ ] List hardcoded paths and configurations
- [ ] Identify critical functionality that must be preserved

## 🧹 Phase 2: Repository Structure Cleanup

### 2.1 Remove Temporary and Generated Files
```bash
# Remove build artifacts
rm -rf build/ dist/ *.egg-info/
rm -rf __pycache__/ .pytest_cache/
find . -name "*.pyc" -delete
find . -name "*.pyo" -delete

# Remove logs and temporary files
rm -rf logs/ runinfo/ executed_workflows/
rm -f *.err *.out *.log
rm -f cmd_parsl.* parsl.*

# Remove backup files
find . -name "*.backup*" -delete
find . -name "*_backup_*" -delete
```

### 2.2 Demo Directory Consolidation

**KEEP (Working/Important):**
- `demos/viral_pssm_workflow/` - Target for fixes
- `demos/rag_database_creation/` - Target for fixes  
- `demos/academylink_aurora_demo/` - Main showcase demo
- `demos/simple_demo/` - Basic example

**ARCHIVE (Move to archive/ directory):**
- All other demo directories (18 directories)
- Keep for reference but remove from main structure

**DELETE (Broken/Duplicate):**
- `demos/academylink_aurora_demo_backup_*`
- `demos/bck-academylink/`
- `demos/duplicate_transfer_fix_demo/`

### 2.3 Target Directory Structure
```
nanobrain/
├── README.md                 # Updated comprehensive guide
├── pyproject.toml           # Cleaned dependencies
├── setup.py                 # Simplified setup
├── nanobrain/               # Core framework
│   ├── __init__.py
│   ├── core/               # Core components
│   ├── agents/             # Agent implementations
│   ├── workflows/          # Workflow orchestration
│   ├── executors/          # Execution backends
│   └── utils/              # Utilities
├── demos/                   # Cleaned demo directory
│   ├── simple_demo/        # Basic example
│   ├── viral_pssm_workflow/ # Fixed viral PSSM demo
│   ├── rag_database_creation/ # Fixed RAG demo
│   └── academylink_aurora_demo/ # Showcase demo
├── tests/                   # Comprehensive tests
├── docs/                    # Documentation
├── config/                  # Configuration templates
└── archive/                 # Archived demos and experiments
```

## 🔧 Phase 3: Core Framework Fixes

### 3.1 Import and Dependency Issues
- [ ] Fix circular imports in core modules
- [ ] Remove hardcoded paths (replace with configurable paths)
- [ ] Clean up pyproject.toml dependencies
- [ ] Add proper __init__.py files where missing
- [ ] Fix relative imports throughout codebase

### 3.2 Configuration System
- [ ] Create centralized configuration management
- [ ] Remove environment-specific hardcoded values
- [ ] Add configuration validation
- [ ] Create configuration templates for different environments

### 3.3 Error Handling and Logging
- [ ] Implement consistent error handling patterns
- [ ] Add proper logging configuration
- [ ] Remove debug print statements
- [ ] Add graceful degradation for missing dependencies

## 🎯 Phase 4: Demo Fixes (Priority Targets)

### 4.1 Viral PSSM Workflow Demo
**Current Issues:**
- Complex directory structure with many test files
- Hardcoded BV-BRC API dependencies
- Missing proper error handling

**Fixes:**
- [ ] Simplify directory structure
- [ ] Add mock data for testing without API access
- [ ] Create clear README with setup instructions
- [ ] Add comprehensive tests
- [ ] Fix configuration management

### 4.2 RAG Database Creation Demo
**Current Issues:**
- Multiple similar files (create_*, final_*, simple_*)
- Complex distributed processing setup
- Missing clear entry points

**Fixes:**
- [ ] Consolidate into single main script with options
- [ ] Add local/simple mode for testing
- [ ] Clear documentation of requirements
- [ ] Simplified configuration
- [ ] Better error messages for missing dependencies

### 4.3 Simple Demo Enhancement
- [ ] Ensure it works out of the box
- [ ] Add comprehensive comments
- [ ] Create step-by-step tutorial
- [ ] Test with minimal dependencies

## 📚 Phase 5: Documentation and Packaging

### 5.1 Documentation Overhaul
- [ ] Rewrite README.md with honest assessment
- [ ] Create INSTALLATION.md with clear requirements
- [ ] Add CONTRIBUTING.md guidelines
- [ ] Document API with proper docstrings
- [ ] Create troubleshooting guide

### 5.2 Packaging Improvements
- [ ] Clean pyproject.toml with proper optional dependencies
- [ ] Add proper entry points for CLI tools
- [ ] Create Docker containers for different use cases
- [ ] Add GitHub Actions for CI/CD
- [ ] Create release process documentation

### 5.3 Testing Infrastructure
- [ ] Add comprehensive unit tests
- [ ] Create integration tests for demos
- [ ] Add performance benchmarks
- [ ] Set up automated testing pipeline

## 🚀 Phase 6: Validation and Deployment

### 6.1 Testing Protocol
- [ ] Test installation from scratch on clean environment
- [ ] Validate all demos work with documented requirements
- [ ] Performance testing of core components
- [ ] Documentation accuracy verification

### 6.2 Release Preparation
- [ ] Version tagging and changelog
- [ ] Create release notes
- [ ] Package for PyPI (optional)
- [ ] Create deployment guides for different environments

## ⚠️ Risk Mitigation

### High-Risk Operations
1. **Demo directory removal** - Could break existing workflows
2. **Import restructuring** - May cause temporary breakage
3. **Configuration changes** - Could affect existing deployments

### Mitigation Strategies
1. **Incremental approach** - One phase at a time with testing
2. **Backup everything** - Multiple backup strategies
3. **Rollback plan** - Clear steps to revert changes
4. **Testing at each phase** - Validate before proceeding

## 📊 Success Metrics

### Phase Completion Criteria
- [ ] **Phase 1**: Backups created and verified
- [ ] **Phase 2**: Repository size reduced by >50%
- [ ] **Phase 3**: Core framework imports work without errors
- [ ] **Phase 4**: Target demos run successfully with clear instructions
- [ ] **Phase 5**: Documentation is comprehensive and accurate
- [ ] **Phase 6**: Fresh installation works on clean environment

### Quality Gates
- All tests pass
- Documentation is complete and accurate
- Installation process is streamlined
- Core functionality preserved
- Performance maintained or improved

## 🕐 Estimated Timeline

- **Phase 1**: 1-2 hours (backup and assessment)
- **Phase 2**: 4-6 hours (cleanup and restructuring)
- **Phase 3**: 8-12 hours (core framework fixes)
- **Phase 4**: 12-16 hours (demo fixes)
- **Phase 5**: 6-8 hours (documentation)
- **Phase 6**: 4-6 hours (validation)

**Total Estimated Time**: 35-50 hours over 1-2 weeks

## 🎯 Immediate Next Steps

1. **Get approval** for this cleanup plan
2. **Create backups** (Phase 1)
3. **Start with Phase 2** (low-risk cleanup)
4. **Focus on viral_pssm and rag_database_creation** demos as requested
5. **Iterate and validate** at each phase

This plan balances thoroughness with practicality, ensuring the framework becomes maintainable while preserving its core functionality and the specific demos you need working.
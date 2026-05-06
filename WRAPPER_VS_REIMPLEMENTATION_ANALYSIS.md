# BRUTAL TRUTH: Perl Wrapper vs Python Reimplementation

## Discovery: Modules ARE Available

**Repository:** https://github.com/TheSEED/seed_gjo
**License:** Not specified (need to check)
**Modules Found:**
- ✅ `lib/gjoseqlib.pm` (3,117 lines)
- ✅ `lib/BlastInterface.pm` (3,629 lines)
- ✅ 19 other supporting modules

**This changes EVERYTHING.**

---

## Option 1: Perl Wrapper Approach

### What You Get

**PROS:**
1. ✅ **Proven algorithm** - Jim Davis's code is battle-tested for viral annotations
2. ✅ **Exact behavior** - No risk of reimplementation bugs
3. ✅ **Faster to deploy** - Bundle modules, not rewrite logic
4. ✅ **Reference output** - Can compare against known-good results
5. ✅ **Maintainability** - Upstream bug fixes benefit you

**CONS:**
1. ❌ **21 Perl modules to bundle** - Not just 2, the whole dependency tree
2. ❌ **Perl ecosystem complexity** - Module paths, @INC configuration
3. ❌ **Black box debugging** - Harder to debug Perl from Python
4. ❌ **Conda-pack size** - Large tarball with Perl + modules + tools
5. ❌ **STDIN/STDOUT coupling** - Must pipe data correctly
6. ❌ **Error handling** - Parse stderr from Perl script
7. ❌ **Hardcoded threading** - Still need to patch for Aurora

### Implementation Complexity

**Conda Environment:**
```yaml
name: viral-pssm-perl
dependencies:
  - perl >=5.26
  - mmseqs2 >=14.0
  - mafft >=7.487
  - blast >=2.13.0
  # Perl modules (need to bundle from seed_gjo)
```

**Bundling seed_gjo Modules:**
```bash
# Clone repository
git clone https://github.com/TheSEED/seed_gjo.git

# Copy modules to conda env
cp -r seed_gjo/lib/* $CONDA_PREFIX/lib/perl5/site_perl/

# Test import
perl -c -I$CONDA_PREFIX/lib/perl5/site_perl fasta-cluster-pssm.pl
```

**Python Wrapper:**
```python
class FastaClusterPSSMStep(Step):
    """Wrapper for fasta-cluster-pssm.pl Perl script."""

    async def process(self, input_data: Dict, **kwargs) -> Dict:
        # 1. Unpack conda environment (Perl + modules + tools)
        env_dir = await self._unpack_conda_env()

        # 2. Write input FASTA
        fasta_path = self._write_fasta(input_data['sequences'])

        # 3. Execute Perl script via subprocess
        result = await self._run_perl_script(
            env_dir, fasta_path, input_data['parameters']
        )

        # 4. Parse output files
        return self._parse_results(result)

    async def _run_perl_script(self, env_dir, fasta_path, params):
        """Execute Perl script with proper environment."""
        perl_path = env_dir / "bin/perl"
        perl_script = self.config_dir / "fasta-cluster-pssm.pl"

        # Set PERL5LIB to include bundled modules
        env = os.environ.copy()
        env['PERL5LIB'] = f"{env_dir}/lib/perl5/site_perl"

        cmd = [
            str(perl_path),
            "-I", f"{env_dir}/lib/perl5/site_perl",
            str(perl_script),
            "-f", str(params['column_fraction']),
            "-n", str(params['nterm_conservation']),
            "-c", str(params['cterm_conservation']),
            "-mi", str(params['mmseqs_identity']),
            "-mc", str(params['mmseqs_coverage'])
        ]

        # Pipe FASTA to STDIN
        with open(fasta_path, 'r') as stdin_file:
            result = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=stdin_file,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self.temp_dir  # Perl script creates subdirectories
            )

            stdout, stderr = await result.communicate()

        if result.returncode != 0:
            raise RuntimeError(f"Perl script failed: {stderr.decode()}")

        return self._parse_perl_output()
```

**Estimated Implementation Time:** 2-3 days
- Day 1: Bundle modules, test Perl script runs (8 hours)
- Day 2: Implement Python wrapper (6 hours)
- Day 3: Test, debug, validate (6 hours)

**Estimated Conda Pack Size:** 300-500 MB (Perl + modules + BLAST+ + MMseqs2 + MAFFT)

---

## Option 2: Python Reimplementation

### What You Get

**PROS:**
1. ✅ **Full control** - Modify algorithm, fix bugs, optimize
2. ✅ **Type safety** - Pydantic models, static analysis
3. ✅ **Better error messages** - Python exceptions with context
4. ✅ **Easier debugging** - Step through Python code
5. ✅ **Smaller footprint** - No Perl runtime needed
6. ✅ **Native integration** - Pure Python, no subprocess complexity
7. ✅ **Aurora-optimized** - Parameterize everything properly

**CONS:**
1. ❌ **800-1000 lines to write** - Significant development effort
2. ❌ **Potential bugs** - Risk of deviating from Perl behavior
3. ❌ **No reference** - Hard to validate correctness
4. ❌ **Algorithm ownership** - You maintain the curation logic
5. ❌ **Time investment** - 5-6 days vs 2-3 days

### Implementation Complexity

**Already Designed:**
- ✅ Data models documented
- ✅ Alignment utilities spec'd out
- ✅ Main step structure defined
- ✅ Test strategy planned

**Estimated Implementation Time:** 5-6 days (as documented)

**Estimated Conda Pack Size:** 200-300 MB (Python + BioPython + tools, no Perl)

---

## Side-by-Side Comparison

| Aspect | Perl Wrapper | Python Reimplementation |
|--------|--------------|------------------------|
| **Development Time** | 2-3 days | 5-6 days |
| **Lines of Code** | ~200 (wrapper) | ~1,000 (full impl) |
| **Correctness** | ✅ Proven | ⚠️ Needs validation |
| **Debuggability** | ❌ Black box | ✅ Full visibility |
| **Maintainability** | ⚠️ Depends on upstream | ✅ You control |
| **Error Handling** | ❌ Parse stderr | ✅ Python exceptions |
| **Type Safety** | ❌ None | ✅ Pydantic |
| **Performance** | ✅ Same (uses same tools) | ✅ Same (uses same tools) |
| **Conda Pack Size** | 300-500 MB | 200-300 MB |
| **Testing Complexity** | ⚠️ Integration only | ✅ Unit + Integration |
| **Aurora Optimization** | ⚠️ Still hardcoded threads | ✅ Fully parameterized |

---

## BRUTAL TRUTH Analysis

### Wrapper Approach Risks

1. **Perl Module Hell**
   - Need ALL 21 modules from seed_gjo
   - Possible circular dependencies
   - PERL5LIB path configuration fragile
   - Version conflicts possible

2. **Hardcoded Threading**
   - Perl script: `mafft --thread 24`
   - **MUST patch the script anyway** to parameterize
   - So you're not truly using it "as-is"

3. **Temp Directory Management**
   - Script does `mkdir($tmp); chdir($tmp);`
   - Concurrent Parsl workers might collide
   - Need unique temp dir per execution

4. **STDIN/STDOUT Coupling**
   - Must pipe FASTA correctly
   - Perl script expects specific format
   - Error messages go to stderr (need parsing)

5. **Output Parsing**
   - Still need to parse PSSM files
   - Still need to parse Curation_Report
   - Same effort as reimplementation

### Reimplementation Approach Risks

1. **Algorithm Deviation**
   - Easy to introduce subtle bugs in curation logic
   - Conservation calculation might differ slightly
   - Column trimming edge cases

2. **PSSM Format Differences**
   - psiblast might generate different PSSM format than BlastInterface
   - **CRITICAL TO TEST THIS**

3. **Time Investment**
   - 5-6 days is significant
   - Might discover issues late

---

## My Recommendation: **HYBRID APPROACH**

**DON'T** choose purely wrapper or reimplementation.

**DO** this hybrid:

### Phase 1: Test Wrapper (1 day)
1. **Bundle seed_gjo modules** into test conda env
2. **Run fasta-cluster-pssm.pl** with sample data
3. **Validate it works** end-to-end
4. **Compare psiblast vs BlastInterface PSSMs**

**Decision Point:** If Perl wrapper works cleanly → use it. If problematic → reimplement.

### Phase 2a: If Wrapper Works (2 days)
1. **Patch script** to parameterize threading: `--thread $ENV{MAFFT_THREADS}`
2. **Create Python wrapper** as designed above
3. **Test on Aurora**

### Phase 2b: If Wrapper Has Issues (5 days)
1. **Proceed with Python reimplementation**
2. **Use Perl script as reference** for validation
3. **Compare outputs** at each step

---

## Recommendation Summary

**START WITH WRAPPER TEST** (1 day investment)

**Reasons:**
1. ✅ **Low risk** - 1 day to prove viability
2. ✅ **Definitive answer** - Will know if modules work
3. ✅ **Reference validation** - Can test against known-good
4. ✅ **Fast deployment** - If it works, 2 more days to production
5. ✅ **Fallback option** - Can still reimplement if needed

**Test Criteria:**
- ✅ Can import all modules without errors
- ✅ Script runs on sample data
- ✅ Generates valid PSSM files
- ✅ Output format matches expectations
- ✅ Works in conda-packed environment

**IF wrapper test passes:** Deploy wrapper (total: 3 days)
**IF wrapper test fails:** Reimplement in Python (total: 6 days)

**Either way, maximum risk is 1 day to find out.**

---

## Next Immediate Action

**TEST THE WRAPPER APPROACH** (4-8 hours)

```bash
# 1. Create test conda environment
conda create -n test-perl-wrapper perl mmseqs2 mafft blast

# 2. Clone seed_gjo
git clone https://github.com/TheSEED/seed_gjo.git

# 3. Install modules
conda activate test-perl-wrapper
cp -r seed_gjo/lib/* $CONDA_PREFIX/lib/perl5/site_perl/

# 4. Download Perl script
curl -O https://raw.githubusercontent.com/jimdavis1/Viral_Annotation/master/Other_Scripts/fasta-cluster-pssm.pl

# 5. Test with sample data
echo ">seq1
MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGKQDKEVLKEQKLISEEDL
>seq2
MKVLWAALLVTFLAGCQAKVEQAVETEPEPELRQQTEWQSGQRWELALGKQDKEVLKEQKLISEEDL" | \
perl -I$CONDA_PREFIX/lib/perl5/site_perl fasta-cluster-pssm.pl -f 0.7 -n 0.8 -c 0.8

# 6. Check if it works
ls -la # Look for pssms/, alis/, corrected_alis/, Curation_Report
```

**If this works → Wrapper is viable**
**If this fails → Reimplementation required**

**Should I proceed with the wrapper test?**

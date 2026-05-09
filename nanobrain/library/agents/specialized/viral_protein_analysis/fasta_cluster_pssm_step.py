"""
FastaClusterPSSMStep - Distributed PSSM Generation via Parsl

Executes fasta-cluster-pssm.pl on Aurora compute nodes using Parsl distributed execution.
Handles:
- Environment setup (Perl + PSSM tools on compute nodes)
- Parameter grid execution for PSSM generation
- Output parsing (pssms/, alis/, Curation_Report)
- Result aggregation

Validated Infrastructure:
- Environment: 421 MB conda-packed tarball with Perl 5.32.1, psiblast 2.13.0+, mmseqs, mafft
- Executor: ParslExecutor with PBS submission to Aurora debug/workq
- Worker init: Layered Python+Parsl and PSSM environments (tested job 8180824)
"""

import os
import asyncio
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.executor import ParslExecutor


class FastaClusterPSSMStep(Step):
    """
    Step for distributed PSSM generation using fasta-cluster-pssm.pl.

    This step:
    - Takes clustered FASTA sequences as input
    - Executes fasta-cluster-pssm.pl via Parsl on Aurora compute nodes
    - Configures threading via MAFFT_THREADS environment variable
    - Parses PSSM outputs (pssms/, alis/, Curation_Report)
    - Returns structured results with quality metrics

    Architecture:
    - Uses validated pssm_parsl_executor.yml config
    - Leverages tested worker init (Python + PSSM environments layered)
    - Supports parameter grid execution for optimization
    """

    REQUIRED_CONFIG_FIELDS = ['name']
    OPTIONAL_CONFIG_FIELDS = {
        'description': 'PSSM generation via fasta-cluster-pssm.pl',
        'executor_config_path': 'config/pssm_parsl_executor.yml',
        'default_threads': 4,
        'timeout': 3600,
        'auto_initialize': True,
        'debug_mode': False,
        'enable_logging': True
    }

    @classmethod
    def _get_config_class(cls):
        """UNIFIED PATTERN: Return StepConfig"""
        return StepConfig

    @classmethod
    def extract_component_config(cls, config: StepConfig) -> Dict[str, Any]:
        """Extract FastaClusterPSSMStep configuration"""
        base_config = super().extract_component_config(config)

        # Access config fields directly from StepConfig object (not from config.config!)
        # Use model_dump() to get dict representation for easy access
        step_config_dict = config.model_dump() if hasattr(config, 'model_dump') else {}

        return {
            **base_config,
            'executor_config_path': step_config_dict.get('executor_config_path', 'config/pssm_parsl_executor.yml'),
            'default_threads': step_config_dict.get('default_threads', 4),
            'timeout': step_config_dict.get('timeout', 3600),
            'output_dir': step_config_dict.get('output_dir', '/tmp/pssm_output'),
            'keep_intermediate': step_config_dict.get('keep_intermediate', False)
        }

    def _init_from_config(self, config: StepConfig, component_config: Dict[str, Any],
                         dependencies: Dict[str, Any]) -> None:
        """Initialize FastaClusterPSSMStep with resolved dependencies"""
        super()._init_from_config(config, component_config, dependencies)

        # Extract configuration
        self.executor_config_path = component_config.get('executor_config_path')
        self.default_threads = component_config.get('default_threads', 4)
        self.timeout = component_config.get('timeout', 3600)
        self.output_dir = component_config.get('output_dir', '/tmp/pssm_output')
        self.keep_intermediate = component_config.get('keep_intermediate', False)

        # Initialize executor (will be loaded in initialize())
        self.executor = None

        self.nb_logger.info("🧬 FastaClusterPSSMStep initialized")
        self.nb_logger.info(f"   Executor config path: {self.executor_config_path}")
        self.nb_logger.info(f"   Default threads: {self.default_threads}")
        self.nb_logger.info(f"   Timeout: {self.timeout}s")

    async def initialize(self) -> None:
        """Initialize the Parsl executor"""
        if self.executor is not None:
            return

        self.nb_logger.info("🔧 Initializing ParslExecutor...")

        try:
            # Load executor from config path
            self.executor = ParslExecutor.from_config(self.executor_config_path)
            await self.executor.initialize()

            self.nb_logger.info(f"✅ ParslExecutor initialized: {self.executor.config.name}")

        except Exception as e:
            self.nb_logger.error(f"❌ Failed to initialize ParslExecutor: {e}")
            raise

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.

        Input format:
        {
            'fasta_file': '/path/to/clustered.fasta',  # Required
            'threads': 4,                               # Optional (default: from config)
            'output_dir': '/path/to/output',           # Optional (default: /tmp/pssm_output)
            'parameters': {...}                         # Optional parameter grid overrides
        }

        Returns:
        {
            'success': bool,
            'pssms': List[Dict],        # Parsed PSSM matrices
            'alignments': List[Dict],   # Alignment information
            'curation_report': Dict,    # Quality metrics from Curation_Report
            'execution_time': float,
            'output_dir': str
        }
        """
        self.nb_logger.info("🔄 Starting PSSM generation")

        # Ensure executor is initialized
        if self.executor is None:
            await self.initialize()

        # Extract FASTA input - support multiple input formats
        fasta_file = input_data.get('fasta_file')
        fasta_content = input_data.get('fasta_content') or input_data.get('fasta_sequences')

        # Create temp FASTA file if content provided instead of file path
        if fasta_content and not fasta_file:
            import tempfile
            temp_fasta = tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False)
            temp_fasta.write(fasta_content)
            temp_fasta.close()
            fasta_file = temp_fasta.name
            self.nb_logger.info(f"   Created temp FASTA file from content: {fasta_file}")
        elif not fasta_file:
            raise ValueError("Input must contain 'fasta_file', 'fasta_content', or 'fasta_sequences' key")

        if not Path(fasta_file).exists():
            raise FileNotFoundError(f"FASTA file not found: {fasta_file}")

        threads = input_data.get('threads', self.default_threads)
        output_dir = Path(input_data.get('output_dir', self.output_dir))
        parameters = input_data.get('parameters', {})

        self.nb_logger.info(f"   FASTA: {fasta_file}")
        self.nb_logger.info(f"   Threads: {threads}")
        self.nb_logger.info(f"   Output: {output_dir}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Execute PSSM generation
            result = await self._execute_pssm_generation(
                fasta_file=fasta_file,
                output_dir=output_dir,
                threads=threads,
                parameters=parameters
            )

            self.nb_logger.info("✅ PSSM generation completed successfully")
            return result

        except Exception as e:
            self.nb_logger.error(f"❌ PSSM generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'fasta_file': fasta_file,
                'output_dir': str(output_dir)
            }

    async def _execute_pssm_generation(self, fasta_file: str, output_dir: Path,
                                      threads: int, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fasta-cluster-pssm.pl via Parsl.

        The Perl script expects FASTA input via STDIN and generates outputs in current directory:
        - pssms/           - PSSM matrix files
        - alis/            - Alignment files
        - Curation_Report  - Quality metrics and statistics
        """
        import time
        start_time = time.time()

        # Read FASTA content
        with open(fasta_file, 'r') as f:
            fasta_content = f.read()

        # Create Parsl python_app for PSSM generation
        from parsl.app.app import python_app

        @python_app
        def run_pssm_generation(fasta_content: str, output_dir: str, threads: int):
            """
            Execute fasta-cluster-pssm.pl on compute node.

            This function runs on the Aurora compute node where:
            - PSSM environment is unpacked to /tmp/ by worker_init
            - Environment variables are set (PSSM_ENV_ROOT, PERL5LIB, PSSM_SCRIPT, MAFFT_THREADS)
            - Both Python (Parsl) and PSSM tools are available
            """
            from pathlib import Path

            # Get PSSM script path from environment (set by worker_init)
            pssm_script = os.environ.get('PSSM_SCRIPT')
            if not pssm_script or not os.path.exists(pssm_script):
                return {
                    'success': False,
                    'error': f'PSSM script not found: {pssm_script}'
                }

            # Create working directory on compute node
            work_dir = Path(output_dir)
            work_dir.mkdir(parents=True, exist_ok=True)

            try:
                # Set environment variables for script
                env = os.environ.copy()
                env['MAFFT_THREADS'] = str(threads)

                # Write FASTA to temporary file (script reads from STDIN)
                with tempfile.NamedTemporaryFile(mode='w', suffix='.fasta', delete=False) as fasta_tmp:
                    fasta_tmp.write(fasta_content)
                    fasta_tmp_path = fasta_tmp.name

                # Execute fasta-cluster-pssm.pl
                # Script reads FASTA from STDIN and generates outputs in current directory
                with open(fasta_tmp_path, 'r') as stdin:
                    result = subprocess.run(
                        ['perl', pssm_script],
                        stdin=stdin,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        cwd=str(work_dir),
                        env=env,
                        timeout=3600  # 1 hour timeout
                    )

                # Clean up temp file
                os.unlink(fasta_tmp_path)

                # Check result
                if result.returncode != 0:
                    return {
                        'success': False,
                        'error': f'Script failed with return code {result.returncode}',
                        'stdout': result.stdout.decode('utf-8', errors='replace'),
                        'stderr': result.stderr.decode('utf-8', errors='replace')
                    }

                # Parse outputs
                pssms_dir = work_dir / 'pssms'
                alis_dir = work_dir / 'alis'
                curation_report = work_dir / 'Curation_Report'

                # Count generated files
                pssm_count = len(list(pssms_dir.glob('*'))) if pssms_dir.exists() else 0
                ali_count = len(list(alis_dir.glob('*'))) if alis_dir.exists() else 0

                # Read curation report if exists
                curation_data = {}
                if curation_report.exists():
                    with open(curation_report, 'r') as f:
                        curation_text = f.read()
                    # Parse curation report (simple text parsing)
                    curation_data = {'raw_report': curation_text}
                    # TODO: Add structured parsing of curation report

                return {
                    'success': True,
                    'pssm_count': pssm_count,
                    'alignment_count': ali_count,
                    'curation_report': curation_data,
                    'output_dir': str(work_dir),
                    'stdout': result.stdout.decode('utf-8', errors='replace')[:1000],  # First 1KB
                    'stderr': result.stderr.decode('utf-8', errors='replace')[:1000]
                }

            except subprocess.TimeoutExpired:
                return {
                    'success': False,
                    'error': 'Script execution timed out after 1 hour'
                }
            except Exception as e:
                return {
                    'success': False,
                    'error': f'Execution error: {str(e)}'
                }

        # Submit task to Parsl
        self.nb_logger.info("📤 Submitting PSSM generation task to Parsl...")
        future = run_pssm_generation(fasta_content, str(output_dir), threads)

        # Wait for result
        self.nb_logger.info("⏳ Waiting for compute node execution...")
        task_result = await asyncio.get_event_loop().run_in_executor(
            None, future.result
        )

        execution_time = time.time() - start_time

        if not task_result.get('success'):
            self.nb_logger.error(f"❌ Task failed: {task_result.get('error')}")
            return {
                'success': False,
                **task_result,
                'execution_time': execution_time
            }

        self.nb_logger.info("✅ Task completed:")
        self.nb_logger.info(f"   PSSMs generated: {task_result['pssm_count']}")
        self.nb_logger.info(f"   Alignments: {task_result['alignment_count']}")
        self.nb_logger.info(f"   Execution time: {execution_time:.2f}s")

        # Parse detailed outputs
        pssm_data = await self._parse_pssm_outputs(output_dir)
        alignment_data = await self._parse_alignment_outputs(output_dir)

        return {
            'success': True,
            'pssms': pssm_data,
            'alignments': alignment_data,
            'curation_report': task_result.get('curation_report', {}),
            'execution_time': execution_time,
            'output_dir': str(output_dir),
            'pssm_count': task_result['pssm_count'],
            'alignment_count': task_result['alignment_count'],
            'threads_used': threads,
            'fasta_file': fasta_file,
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'threads': threads,
                'output_dir': str(output_dir)
            }
        }

    async def _parse_pssm_outputs(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Parse PSSM files from pssms/ directory"""
        pssms = []
        pssms_dir = output_dir / 'pssms'

        if not pssms_dir.exists():
            return pssms

        for pssm_file in pssms_dir.glob('*'):
            try:
                # Read PSSM file
                with open(pssm_file, 'r') as f:
                    pssm_content = f.read()

                pssms.append({
                    'filename': pssm_file.name,
                    'size': pssm_file.stat().st_size,
                    'content': pssm_content[:1000],  # First 1KB for preview
                    # TODO: Add structured PSSM matrix parsing
                })
            except Exception as e:
                self.nb_logger.warning(f"Failed to parse PSSM file {pssm_file}: {e}")

        return pssms

    async def _parse_alignment_outputs(self, output_dir: Path) -> List[Dict[str, Any]]:
        """Parse alignment files from alis/ directory"""
        alignments = []
        alis_dir = output_dir / 'alis'

        if not alis_dir.exists():
            return alignments

        for ali_file in alis_dir.glob('*'):
            try:
                alignments.append({
                    'filename': ali_file.name,
                    'size': ali_file.stat().st_size,
                    # TODO: Add alignment statistics parsing
                })
            except Exception as e:
                self.nb_logger.warning(f"Failed to parse alignment file {ali_file}: {e}")

        return alignments

    async def cleanup(self) -> None:
        """Cleanup Parsl executor"""
        if self.executor:
            await self.executor.shutdown()
            self.executor = None
            self.nb_logger.info("🧹 ParslExecutor cleaned up")

r"""
BV-BRC Command Pipeline

Implements the exact 4-step BV-BRC CLI command sequence:
1. Get all genomes for taxon: p3-all-genomes --eq taxon_id,<taxon_id> > <taxon_id>.tsv
2. Get genome features: cut -f1 <taxon_id>.tsv | p3-get-genome-features --attr patric_id --attr product > <taxon_id>.id_md5
3. Filter unique md5s: grep "CDS\|mat" <taxon_id>.id_md5 |cut -f2 | sort -u | perl -e 'while (<>){chomp; if ($_ =~ /\w/){print "$_\n";}}' > <taxon_id>.uniqe.md5
4. Get sequences: p3-get-feature-sequence --input <taxon_id>.uniqe.md5 --col 0 > <taxon_id>.unique.seq

Based on user requirements and BV-BRC CLI documentation.
"""

import asyncio
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
import shutil

from nanobrain.core.logging_system import get_logger


@dataclass
class CommandResult:
    """Result of a single BV-BRC command execution"""
    success: bool
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    command: str


@dataclass
class PipelineFiles:
    """Files created during pipeline execution"""
    working_dir: Path
    genomes_tsv: Path
    features_id_md5: Path
    unique_md5: Path
    sequences_fasta: Path
    
    def cleanup(self):
        """Clean up all pipeline files (if temporary)"""
        if self.working_dir.name.startswith('tmp'):
            shutil.rmtree(self.working_dir, ignore_errors=True)


@dataclass
class PipelineResult:
    """Complete result of BV-BRC pipeline execution"""
    success: bool
    taxon_id: str
    execution_time: float
    files: Optional[PipelineFiles] = None
    commands_executed: List[CommandResult] = None
    error_message: Optional[str] = None
    genome_count: int = 0
    feature_count: int = 0
    unique_md5_count: int = 0
    sequence_count: int = 0


class BVBRCCommandPipeline:
    """
    Executes the exact BV-BRC CLI command sequence for protein extraction.
    
    This implements the user-specified 4-step command sequence with:
    - Temporary working directories per taxon
    - Fail-fast error handling
    - Preserved intermediate files for debugging
    - Proper shell command execution with timeouts
    """
    
    def __init__(self, 
                 bvbrc_cli_path: str = "/Applications/BV-BRC.app/deployment/bin",
                 timeout_seconds: int = 300,
                 preserve_files: bool = True):
        """
        Initialize BV-BRC command pipeline.
        
        Args:
            bvbrc_cli_path: Path to BV-BRC CLI tools directory
            timeout_seconds: Timeout for each command execution
            preserve_files: Whether to preserve intermediate files for debugging
        """
        self.cli_path = Path(bvbrc_cli_path)
        self.timeout_seconds = timeout_seconds
        self.preserve_files = preserve_files
        self.logger = get_logger("bvbrc_command_pipeline")
        
        # Verify CLI tools exist
        self._verify_cli_tools()
    
    def _verify_cli_tools(self) -> None:
        """Verify that required BV-BRC CLI tools are available"""
        required_tools = [
            "p3-all-genomes",
            "p3-get-genome-features", 
            "p3-get-feature-sequence"
        ]
        
        missing_tools = []
        for tool in required_tools:
            tool_path = self.cli_path / tool
            if not tool_path.exists():
                missing_tools.append(tool)
        
        if missing_tools:
            raise FileNotFoundError(
                f"Required BV-BRC CLI tools not found at {self.cli_path}: {missing_tools}"
            )
        
        self.logger.info(f"✅ BV-BRC CLI tools verified at {self.cli_path}")
    
    async def execute_pipeline(self, taxon_id: str) -> PipelineResult:
        """
        Execute complete BV-BRC pipeline for a specific taxon.
        
        Args:
            taxon_id: Taxon ID for genome extraction (e.g., "11020")
            
        Returns:
            PipelineResult with execution details and file paths
        """
        start_time = time.time()
        self.logger.info(f"🚀 Starting BV-BRC pipeline for taxon {taxon_id}")
        
        # Create temporary working directory for this taxon
        working_dir = Path(tempfile.mkdtemp(prefix=f"bvbrc_taxon_{taxon_id}_"))
        self.logger.info(f"📁 Working directory: {working_dir}")
        
        try:
            # Setup file paths
            files = PipelineFiles(
                working_dir=working_dir,
                genomes_tsv=working_dir / f"{taxon_id}.tsv",
                features_id_md5=working_dir / f"{taxon_id}.id_md5",
                unique_md5=working_dir / f"{taxon_id}.uniqe.md5",
                sequences_fasta=working_dir / f"{taxon_id}.unique.seq"
            )
            
            commands_executed = []
            
            # Step 1: Get all genomes for taxon
            step1_result = await self._execute_step1_get_genomes(taxon_id, files.genomes_tsv)
            commands_executed.append(step1_result)
            
            if not step1_result.success:
                return PipelineResult(
                    success=False,
                    taxon_id=taxon_id,
                    execution_time=time.time() - start_time,
                    files=files,
                    commands_executed=commands_executed,
                    error_message=f"Step 1 failed: {step1_result.stderr}"
                )
            
            # Count genomes
            genome_count = await self._count_data_lines(files.genomes_tsv)
            self.logger.info(f"📊 Found {genome_count} genomes for taxon {taxon_id}")
            
            # Step 2: Get genome features
            step2_result = await self._execute_step2_get_features(files.genomes_tsv, files.features_id_md5)
            commands_executed.append(step2_result)
            
            if not step2_result.success:
                return PipelineResult(
                    success=False,
                    taxon_id=taxon_id,
                    execution_time=time.time() - start_time,
                    files=files,
                    commands_executed=commands_executed,
                    error_message=f"Step 2 failed: {step2_result.stderr}"
                )
            
            # Count features
            feature_count = await self._count_data_lines(files.features_id_md5)
            self.logger.info(f"📊 Found {feature_count} features for taxon {taxon_id}")
            
            # Step 3: Filter unique MD5s
            step3_result = await self._execute_step3_filter_unique_md5s(files.features_id_md5, files.unique_md5)
            commands_executed.append(step3_result)
            
            if not step3_result.success:
                return PipelineResult(
                    success=False,
                    taxon_id=taxon_id,
                    execution_time=time.time() - start_time,
                    files=files,
                    commands_executed=commands_executed,
                    error_message=f"Step 3 failed: {step3_result.stderr}"
                )
            
            # Count unique MD5s
            unique_md5_count = await self._count_data_lines(files.unique_md5)
            self.logger.info(f"📊 Found {unique_md5_count} unique protein MD5s for taxon {taxon_id}")
            
            # Step 4: Get feature sequences
            step4_result = await self._execute_step4_get_sequences(files.unique_md5, files.sequences_fasta)
            commands_executed.append(step4_result)
            
            if not step4_result.success:
                return PipelineResult(
                    success=False,
                    taxon_id=taxon_id,
                    execution_time=time.time() - start_time,
                    files=files,
                    commands_executed=commands_executed,
                    error_message=f"Step 4 failed: {step4_result.stderr}"
                )
            
            # Count sequences (approximate - count FASTA headers)
            sequence_count = await self._count_fasta_sequences(files.sequences_fasta)
            self.logger.info(f"📊 Retrieved {sequence_count} protein sequences for taxon {taxon_id}")
            
            total_time = time.time() - start_time
            self.logger.info(f"✅ BV-BRC pipeline completed for taxon {taxon_id} in {total_time:.2f}s")
            
            return PipelineResult(
                success=True,
                taxon_id=taxon_id,
                execution_time=total_time,
                files=files,
                commands_executed=commands_executed,
                genome_count=genome_count,
                feature_count=feature_count,
                unique_md5_count=unique_md5_count,
                sequence_count=sequence_count
            )
            
        except Exception as e:
            self.logger.error(f"❌ Pipeline execution failed for taxon {taxon_id}: {e}")
            return PipelineResult(
                success=False,
                taxon_id=taxon_id,
                execution_time=time.time() - start_time,
                files=PipelineFiles(working_dir, Path(), Path(), Path(), Path()),
                error_message=str(e)
            )
    
    async def _execute_step1_get_genomes(self, taxon_id: str, output_file: Path) -> CommandResult:
        """
        Step 1: Get all genomes for specified taxon
        Command: p3-all-genomes --eq taxon_id,<taxon_id> > <taxon_id>.tsv
        """
        command = f"{self.cli_path}/p3-all-genomes --eq taxon_id,{taxon_id}"
        self.logger.info(f"🔄 Step 1: Getting genomes for taxon {taxon_id}")
        
        return await self._execute_shell_command_with_output(command, output_file)
    
    async def _execute_step2_get_features(self, genomes_file: Path, output_file: Path) -> CommandResult:
        """
        Step 2: Get genome features with patric_id and product attributes
        Command: cut -f1 <taxon_id>.tsv | p3-get-genome-features --attr patric_id --attr product > <taxon_id>.id_md5
        """
        # Build the pipeline command
        cut_cmd = f"cut -f1 {genomes_file}"
        p3_cmd = f"{self.cli_path}/p3-get-genome-features --attr patric_id --attr product"
        command = f"{cut_cmd} | {p3_cmd}"
        
        self.logger.info(f"🔄 Step 2: Getting genome features from {genomes_file.name}")
        
        return await self._execute_shell_command_with_output(command, output_file)
    
    async def _execute_step3_filter_unique_md5s(self, features_file: Path, output_file: Path) -> CommandResult:
        r"""
        Step 3: Filter non-unique MD5 hashes for CDS and mat features
        Command: grep "CDS\|mat" <taxon_id>.id_md5 |cut -f2 | sort -u | perl -e 'while (<>){chomp; if ($_ =~ /\w/){print "$_\n";}}' > <taxon_id>.uniqe.md5
        """
        # Build the complex pipeline command (note: keeping "uniqe" typo as in user specification)
        grep_cmd = f"grep \"CDS\\|mat\" {features_file}"
        cut_cmd = "cut -f2"
        sort_cmd = "sort -u"
        perl_cmd = "perl -e 'while (<>){chomp; if ($_ =~ /\\w/){print \"$_\\n\";}}'"
        
        command = f"{grep_cmd} | {cut_cmd} | {sort_cmd} | {perl_cmd}"
        
        self.logger.info(f"🔄 Step 3: Filtering unique MD5s from {features_file.name}")
        
        return await self._execute_shell_command_with_output(command, output_file)
    
    async def _execute_step4_get_sequences(self, md5_file: Path, output_file: Path) -> CommandResult:
        """
        Step 4: Get feature sequences for unique MD5s
        Command: p3-get-feature-sequence --input <taxon_id>.uniqe.md5 --col 0 > <taxon_id>.unique.seq
        """
        command = f"{self.cli_path}/p3-get-feature-sequence --input {md5_file} --col 0"
        
        self.logger.info(f"🔄 Step 4: Getting feature sequences from {md5_file.name}")
        
        return await self._execute_shell_command_with_output(command, output_file)
    
    async def _execute_shell_command_with_output(self, command: str, output_file: Path) -> CommandResult:
        """
        Execute shell command with output redirection and timeout handling.
        
        Args:
            command: Shell command to execute
            output_file: File to redirect stdout to
            
        Returns:
            CommandResult with execution details
        """
        start_time = time.time()
        self.logger.debug(f"Executing: {command} > {output_file}")
        
        try:
            # Execute command with shell=True to handle pipes and redirections
            process = await asyncio.create_subprocess_shell(
                f"{command} > {output_file}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=output_file.parent
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout_seconds
                )
                
                return_code = process.returncode
                execution_time = time.time() - start_time
                
                # Read actual output from file
                file_content = ""
                if output_file.exists():
                    try:
                        file_content = output_file.read_text()
                    except Exception as e:
                        self.logger.warning(f"Could not read output file {output_file}: {e}")
                
                success = (return_code == 0)
                
                if success:
                    self.logger.debug(f"✅ Command completed in {execution_time:.2f}s")
                else:
                    self.logger.error(f"❌ Command failed with return code {return_code}")
                    self.logger.error(f"stderr: {stderr.decode()}")
                
                return CommandResult(
                    success=success,
                    stdout=file_content,
                    stderr=stderr.decode(),
                    return_code=return_code,
                    execution_time=execution_time,
                    command=command
                )
                
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                
                self.logger.error(f"❌ Command timed out after {self.timeout_seconds}s")
                
                return CommandResult(
                    success=False,
                    stdout="",
                    stderr=f"Command timed out after {self.timeout_seconds} seconds",
                    return_code=-1,
                    execution_time=self.timeout_seconds,
                    command=command
                )
                
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ Command execution failed: {e}")
            
            return CommandResult(
                success=False,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=execution_time,
                command=command
            )
    
    async def _count_data_lines(self, file_path: Path) -> int:
        """Count data lines in a file (excluding headers)"""
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, 'r') as f:
                lines = f.readlines()
            
            # Skip header line if present
            data_lines = [line for line in lines if line.strip() and not line.startswith('#')]
            if data_lines and '\t' in data_lines[0]:  # Likely has header
                return max(0, len(data_lines) - 1)
            else:
                return len(data_lines)
        except Exception as e:
            self.logger.warning(f"Could not count lines in {file_path}: {e}")
            return 0
    
    async def _count_fasta_sequences(self, file_path: Path) -> int:
        """Count FASTA sequences by counting headers"""
        if not file_path.exists():
            return 0
        
        try:
            with open(file_path, 'r') as f:
                content = f.read()
            
            # Count FASTA headers (lines starting with '>')
            return content.count('>')
        except Exception as e:
            self.logger.warning(f"Could not count sequences in {file_path}: {e}")
            return 0 
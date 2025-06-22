"""
Sequence Curation Step

Re-architected to inherit from NanoBrain Step base class.
"""

import asyncio
import time
import numpy as np
from typing import Dict, Any, List, Optional

from nanobrain.core.step import Step, StepConfig
from nanobrain.core.logging_system import get_logger


class SequenceCurationStep(Step):
    """
    Sequence curation functionality
    
    Re-architected to inherit from NanoBrain Step base class.
    """
    
    def __init__(self, config: StepConfig, curation_config: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(config, **kwargs)
        
        # Extract configuration from step config or provided curation_config
        step_config_dict = config.config if hasattr(config, 'config') else {}
        if curation_config:
            step_config_dict.update(curation_config)
        
        self.curation_config = step_config_dict.get('curation_config', {})
        self.step_config = step_config_dict
        
        self.nb_logger.info(f"🧬 SequenceCurationStep initialized")
        
    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process method required by Step base class.
        
        This implements the NanoBrain framework interface while calling the
        original execute method that contains the sequence curation logic.
        """
        self.nb_logger.info("🔄 Processing sequence curation step")
        
        # Call the original execute method
        result = await self.execute(input_data)
        
        self.nb_logger.info(f"✅ Sequence curation step completed successfully")
        return result

    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute sequence curation steps
        
        Args:
            input_data: Contains sequences and annotations
            
        Returns:
            Dict with curation results and quality analysis
        """
        
        step_start_time = time.time()
        
        try:
            self.nb_logger.info("🧹 Starting sequence curation and quality control")
            
            sequences = input_data.get('sequences', [])
            annotations = input_data.get('annotations', [])
            
            # Step 9: Create FASTA file of selected proteins (already done in data acquisition)
            self.nb_logger.info("Step 9: FASTA file creation completed in previous step")
            
            # Step 10: Analyze length distribution
            self.nb_logger.info("Step 10: Analyzing protein length distribution")
            length_analysis = await self._analyze_length_distribution(sequences)
            
            # Step 11: Identify mangled sequences
            self.nb_logger.info("Step 11: Identifying mangled or problematic sequences")
            curation_report = await self._identify_mangled_sequences(sequences)
            
            execution_time = time.time() - step_start_time
            self.nb_logger.info(f"✅ Sequence curation completed in {execution_time:.2f} seconds")
            
            return {
                'length_analysis': length_analysis,
                'curation_report': curation_report,
                'problematic_sequences': curation_report.get('problematic_sequences', []),
                'execution_time': execution_time,
                'quality_statistics': {
                    'total_sequences': len(sequences),
                    'problematic_sequences': len(curation_report.get('problematic_sequences', [])),
                    'quality_score': curation_report.get('overall_quality_score', 0.8)
                }
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Sequence curation failed: {e}")
            raise
            
    async def _analyze_length_distribution(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 10: Analyze protein length distribution
        
        Expected Alphavirus protein lengths (from implementation plan):
        - nsP1: ~535 aa, nsP2: ~800 aa, nsP3: ~530 aa, nsP4: ~610 aa
        - Capsid: ~260 aa, E3: ~60 aa, E2: ~420 aa, 6K: ~55 aa, E1: ~440 aa
        """
        
        length_stats = {}
        
        # Group sequences by type (if available) or analyze all together
        all_lengths = []
        for seq in sequences:
            if isinstance(seq, dict) and 'aa_sequence' in seq:
                length = len(seq['aa_sequence'])
                all_lengths.append(length)
                
        if all_lengths:
            length_stats = {
                'mean': float(np.mean(all_lengths)),
                'median': float(np.median(all_lengths)),
                'std': float(np.std(all_lengths)),
                'min': int(min(all_lengths)),
                'max': int(max(all_lengths)),
                'total_sequences': len(all_lengths),
                'outliers': self._identify_length_outliers(all_lengths)
            }
        else:
            length_stats = {
                'mean': 0.0,
                'median': 0.0,
                'std': 0.0,
                'min': 0,
                'max': 0,
                'total_sequences': 0,
                'outliers': []
            }
            
        return {
            'statistics': length_stats,
            'expected_ranges': self.step_config.get('expected_lengths', {}),
            'analysis_method': 'basic_statistics'
        }
        
    def _identify_length_outliers(self, lengths: List[int]) -> List[Dict[str, Any]]:
        """Identify length outliers using statistical methods"""
        if len(lengths) < 3:
            return []
            
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        outliers = []
        for i, length in enumerate(lengths):
            z_score = abs(length - mean_length) / std_length if std_length > 0 else 0
            if z_score > 2.0:  # More than 2 standard deviations
                outliers.append({
                    'sequence_index': i,
                    'length': length,
                    'z_score': float(z_score),
                    'type': 'length_outlier'
                })
                
        return outliers
        
    async def _identify_mangled_sequences(self, sequences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Step 11: Identify mangled or problematic sequences
        
        Issues to detect:
        - Sequences with unusual amino acid composition
        - Sequences with stop codons (*) 
        - Sequences with ambiguous amino acids (X)
        - Sequences that are too short or too long
        """
        
        problematic_sequences = []
        total_sequences = len(sequences)
        
        for i, seq in enumerate(sequences):
            if not isinstance(seq, dict) or 'aa_sequence' not in seq:
                continue
                
            sequence = seq['aa_sequence']
            issues = []
            
            # Check for stop codons
            if '*' in sequence:
                issues.append(f"Contains {sequence.count('*')} stop codons")
            
            # Check for ambiguous amino acids
            ambiguous_count = sequence.count('X')
            if ambiguous_count > len(sequence) * 0.05:  # >5% ambiguous
                issues.append(f"High ambiguous AA content: {ambiguous_count}/{len(sequence)} ({100*ambiguous_count/len(sequence):.1f}%)")
            
            # Check length against reasonable ranges
            if len(sequence) < 50:
                issues.append(f"Very short sequence: {len(sequence)} amino acids")
            elif len(sequence) > 2000:
                issues.append(f"Very long sequence: {len(sequence)} amino acids")
            
            # Check amino acid composition
            composition_issues = self._check_amino_acid_composition(sequence)
            issues.extend(composition_issues)
            
            if issues:
                severity = self._assess_issue_severity(issues)
                problematic_sequences.append({
                    'sequence_index': i,
                    'sequence_id': seq.get('aa_sequence_md5', f'seq_{i}'),
                    'issues': issues,
                    'severity': severity,
                    'sequence_length': len(sequence)
                })
        
        # Calculate overall quality metrics
        problem_rate = len(problematic_sequences) / total_sequences if total_sequences > 0 else 0
        overall_quality_score = 1.0 - problem_rate
        
        return {
            'total_sequences': total_sequences,
            'problematic_sequences': problematic_sequences,
            'problem_rate': problem_rate,
            'overall_quality_score': overall_quality_score,
            'curation_recommendations': self._generate_curation_recommendations(problematic_sequences)
        }
        
    def _check_amino_acid_composition(self, sequence: str) -> List[str]:
        """Check for unusual amino acid composition"""
        issues = []
        
        if len(sequence) == 0:
            return ["Empty sequence"]
        
        # Check for extremely biased composition
        aa_counts = {}
        for aa in sequence:
            aa_counts[aa] = aa_counts.get(aa, 0) + 1
            
        # Find most common amino acid
        if aa_counts:
            max_count = max(aa_counts.values())
            max_fraction = max_count / len(sequence)
            
            if max_fraction > 0.5:  # More than 50% is one amino acid
                most_common = [aa for aa, count in aa_counts.items() if count == max_count][0]
                issues.append(f"Extreme bias: {max_fraction*100:.1f}% is '{most_common}'")
                
        return issues
        
    def _assess_issue_severity(self, issues: List[str]) -> str:
        """Assess severity of sequence issues"""
        high_severity_keywords = ['stop codon', 'empty sequence', 'extreme bias']
        medium_severity_keywords = ['high ambiguous', 'very short', 'very long']
        
        for issue in issues:
            issue_lower = issue.lower()
            if any(keyword in issue_lower for keyword in high_severity_keywords):
                return 'high'
            elif any(keyword in issue_lower for keyword in medium_severity_keywords):
                return 'medium'
                
        return 'low'
        
    def _generate_curation_recommendations(self, problematic_sequences: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for sequence curation"""
        recommendations = []
        
        if not problematic_sequences:
            recommendations.append("No major sequence quality issues detected")
            return recommendations
            
        high_severity_count = sum(1 for seq in problematic_sequences if seq['severity'] == 'high')
        medium_severity_count = sum(1 for seq in problematic_sequences if seq['severity'] == 'medium')
        
        if high_severity_count > 0:
            recommendations.append(f"Consider removing {high_severity_count} sequences with high severity issues")
            
        if medium_severity_count > 0:
            recommendations.append(f"Review {medium_severity_count} sequences with medium severity issues")
            
        # Specific recommendations based on issue types
        all_issues = []
        for seq in problematic_sequences:
            all_issues.extend(seq['issues'])
            
        if any('stop codon' in issue for issue in all_issues):
            recommendations.append("Remove sequences with stop codons or check for frameshift errors")
            
        if any('ambiguous' in issue for issue in all_issues):
            recommendations.append("Consider filtering sequences with high ambiguous amino acid content")
            
        return recommendations 
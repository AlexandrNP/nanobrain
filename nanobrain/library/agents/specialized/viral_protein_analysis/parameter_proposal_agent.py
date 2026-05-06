"""
Parameter Proposal Agent for PSSM Generation Hyperparameter Search

This agent analyzes protein sequence characteristics and proposes an optimized
parameter search grid for fasta-cluster-pssm.pl execution. Uses LLM to make
intelligent decisions about parameter ranges based on sequence diversity,
conservation patterns, and quality requirements.

✅ FRAMEWORK COMPLIANCE:
- Inherits from SimpleSpecializedAgent (NO explicit LLM client instantiation)
- Uses from_config() pattern exclusively
- Implements required abstract methods from SpecializedAgentBase
- All prompts stored in configuration (NO hardcoded prompts)
"""

import json
import re
from typing import Dict, Any, Optional, List
from pydantic import Field

from nanobrain.library.agents.specialized.base import SimpleSpecializedAgent
from nanobrain.core.agent import AgentConfig
from nanobrain.core.logging_system import get_logger

logger = get_logger(__name__)


class ParameterProposalAgentConfig(AgentConfig):
    """
    Configuration for ParameterProposalAgent.

    Extends base AgentConfig with specialized parameters for parameter grid
    proposal based on sequence analysis.
    """

    # Parameter proposal specific configuration
    min_grid_size: int = Field(
        default=8,
        description="Minimum number of parameter combinations to propose"
    )
    max_grid_size: int = Field(
        default=50,
        description="Maximum number of parameter combinations to propose"
    )
    conservative_mode: bool = Field(
        default=True,
        description="Use conservative parameter ranges for well-conserved regions"
    )

    # Parameter ranges for fasta-cluster-pssm.pl
    column_fraction_range: List[float] = Field(
        default=[0.5, 0.6, 0.7, 0.8],
        description="Possible values for column_fraction (-f parameter)"
    )
    nterm_conservation_range: List[float] = Field(
        default=[0.6, 0.7, 0.8, 0.9],
        description="Possible values for N-terminal conservation (-n parameter)"
    )
    cterm_conservation_range: List[float] = Field(
        default=[0.6, 0.7, 0.8, 0.9],
        description="Possible values for C-terminal conservation (-c parameter)"
    )

    # Sequence analysis thresholds
    diversity_threshold: float = Field(
        default=0.3,
        description="Diversity threshold for determining parameter strategy"
    )
    min_sequence_length: int = Field(
        default=50,
        description="Minimum sequence length for analysis"
    )


class ParameterProposalAgent(SimpleSpecializedAgent):
    """
    Parameter Proposal Agent - Intelligent Hyperparameter Search Grid Generation
    ===========================================================================

    This specialized agent analyzes protein sequences and proposes optimized
    parameter search grids for PSSM generation using fasta-cluster-pssm.pl.

    **Core Capabilities:**
        * **Sequence Analysis**: Analyze protein sequence diversity and conservation
        * **Parameter Optimization**: Propose parameter ranges based on sequence characteristics
        * **Grid Generation**: Create intelligent search grids for hyperparameter optimization
        * **Conservation Focus**: Optimize for short, well-conserved regions
        * **LLM Integration**: Use LLM to make data-driven parameter decisions

    **Parameter Optimization Strategy:**
        The agent optimizes three critical parameters:

        1. **Column Fraction (-f)**: Remove columns with <fraction bases
           - Higher values → more stringent filtering → shorter, more conserved regions
           - Lower values → retain more positions → longer alignments

        2. **N-Terminal Conservation (-n)**: Trim N-terminus until ≥threshold
           - Higher values → aggressive N-terminal trimming → shorter alignments
           - Lower values → retain more N-terminal residues

        3. **C-Terminal Conservation (-c)**: Trim C-terminus until ≥threshold
           - Higher values → aggressive C-terminal trimming → shorter alignments
           - Lower values → retain more C-terminal residues

    **Usage:**
        ```python
        # Load agent from configuration
        agent = ParameterProposalAgent.from_config('config/parameter_proposal_agent.yml')

        # Analyze sequences and propose parameter grid
        result = await agent.process({
            'sequences': sequences_dict,
            'target_grid_size': 12,
            'optimization_goal': 'short_conserved_regions'
        })

        # Extract proposed parameter grid
        parameter_grid = result['parameter_grid']
        # {
        #     'column_fraction': [0.6, 0.7, 0.8],
        #     'nterm_conservation': [0.7, 0.8],
        #     'cterm_conservation': [0.7, 0.8]
        # }
        ```
    """

    # Component configuration
    COMPONENT_TYPE = "parameter_proposal_agent"
    REQUIRED_CONFIG_FIELDS = ['name', 'model']
    OPTIONAL_CONFIG_FIELDS = {
        'min_grid_size': 8,
        'max_grid_size': 50,
        'conservative_mode': True
    }

    @classmethod
    def _get_config_class(cls):
        """Return ParameterProposalAgent config class"""
        return ParameterProposalAgentConfig

    def _init_from_config(self, config: ParameterProposalAgentConfig,
                          component_config: Dict[str, Any],
                          dependencies: Dict[str, Any]) -> None:
        """Initialize ParameterProposalAgent with configuration"""
        # Call parent initialization
        super()._init_from_config(config, component_config, dependencies)

        # Store parameter proposal specific configuration
        self.min_grid_size = config.min_grid_size
        self.max_grid_size = config.max_grid_size
        self.conservative_mode = config.conservative_mode

        self.column_fraction_range = config.column_fraction_range
        self.nterm_conservation_range = config.nterm_conservation_range
        self.cterm_conservation_range = config.cterm_conservation_range

        self.diversity_threshold = config.diversity_threshold
        self.min_sequence_length = config.min_sequence_length

        self.agent_logger = get_logger(f"agents.parameter_proposal.{self.name}")

    async def process(self, input_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process sequence data and propose parameter search grid.

        Args:
            input_data: Dictionary containing:
                - sequences: Dict of sequence_id -> sequence_string
                - target_grid_size: Optional target number of combinations
                - optimization_goal: Optional goal ('short_conserved_regions', 'comprehensive')
            **kwargs: Additional parameters

        Returns:
            Dictionary containing:
                - parameter_grid: Proposed parameter ranges
                - analysis: Sequence analysis results
                - rationale: Explanation of parameter choices
                - estimated_combinations: Number of parameter combinations
        """
        try:
            self.agent_logger.info("Starting parameter proposal process")

            # Validate input
            if 'sequences' not in input_data:
                raise ValueError("Input must contain 'sequences' dictionary")

            sequences = input_data['sequences']
            target_grid_size = input_data.get('target_grid_size', 12)
            optimization_goal = input_data.get(
                'optimization_goal', 'short_conserved_regions')

            # Analyze sequences
            self.agent_logger.info(f"Analyzing {len(sequences)} sequences")
            sequence_analysis = self._analyze_sequences(sequences)

            # Build prompt for LLM
            prompt = self._build_parameter_proposal_prompt(
                sequence_analysis=sequence_analysis,
                target_grid_size=target_grid_size,
                optimization_goal=optimization_goal
            )

            # Use inherited _process_with_llm() - NO EXPLICIT LLM CLIENT
            self.agent_logger.info("Requesting parameter grid proposal from LLM")
            llm_response = await self._process_with_llm(prompt)

            # Parse and validate LLM response
            result = self._parse_and_validate_proposal(
                llm_response=llm_response,
                sequence_analysis=sequence_analysis
            )

            self.agent_logger.info(
                f"Generated parameter grid with {result['estimated_combinations']} combinations"
            )

            return result

        except Exception as e:
            self.agent_logger.error(f"Error in parameter proposal: {e}", exc_info=True)
            raise

    def _analyze_sequences(self, sequences: Dict[str, str]) -> Dict[str, Any]:
        """
        Analyze sequence characteristics relevant for parameter selection.

        Args:
            sequences: Dictionary of sequence_id -> sequence_string

        Returns:
            Dictionary containing sequence analysis metrics
        """
        if not sequences:
            return {
                'num_sequences': 0,
                'avg_length': 0,
                'min_length': 0,
                'max_length': 0,
                'diversity_estimate': 0.0
            }

        sequence_list = list(sequences.values())
        lengths = [len(seq) for seq in sequence_list]

        # Calculate basic statistics
        analysis = {
            'num_sequences': len(sequences),
            'avg_length': sum(lengths) / len(lengths) if lengths else 0,
            'min_length': min(lengths) if lengths else 0,
            'max_length': max(lengths) if lengths else 0,
            'length_variance': self._calculate_variance(lengths)
        }

        # Estimate sequence diversity (simple pairwise comparison of first 10 sequences)
        diversity = self._estimate_diversity(sequence_list[:10])
        analysis['diversity_estimate'] = diversity

        # Determine suggested strategy based on analysis
        if diversity > self.diversity_threshold:
            analysis['suggested_strategy'] = 'diverse_sequences'
        else:
            analysis['suggested_strategy'] = 'conserved_sequences'

        return analysis

    def _estimate_diversity(self, sequences: List[str]) -> float:
        """
        Estimate sequence diversity using simple pairwise comparison.

        Args:
            sequences: List of sequence strings (max 10 for performance)

        Returns:
            Diversity estimate (0.0 = identical, 1.0 = completely different)
        """
        if len(sequences) < 2:
            return 0.0

        # Compare pairs of sequences
        differences = []
        for i in range(min(len(sequences), 5)):
            for j in range(i + 1, min(len(sequences), 5)):
                seq1 = sequences[i]
                seq2 = sequences[j]
                min_len = min(len(seq1), len(seq2))

                if min_len == 0:
                    continue

                # Count mismatches in overlapping region
                mismatches = sum(
                    1 for k in range(min_len) if seq1[k] != seq2[k]
                )
                differences.append(mismatches / min_len)

        return sum(differences) / len(differences) if differences else 0.0

    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of a list of values."""
        if not values:
            return 0.0

        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance

    def _build_parameter_proposal_prompt(
        self,
        sequence_analysis: Dict[str, Any],
        target_grid_size: int,
        optimization_goal: str
    ) -> str:
        """
        Build prompt for LLM to propose parameter grid.

        Args:
            sequence_analysis: Results from sequence analysis
            target_grid_size: Target number of parameter combinations
            optimization_goal: Optimization goal

        Returns:
            Prompt string for LLM
        """
        prompt = f"""You are a bioinformatics parameter optimization expert specializing in PSSM generation.

Analyze the following protein sequence characteristics and propose an optimized parameter search grid for the fasta-cluster-pssm.pl tool.

**Sequence Analysis:**
- Number of sequences: {sequence_analysis['num_sequences']}
- Average length: {sequence_analysis['avg_length']:.1f} amino acids
- Length range: {sequence_analysis['min_length']}-{sequence_analysis['max_length']} amino acids
- Sequence diversity estimate: {sequence_analysis['diversity_estimate']:.2f} (0.0=identical, 1.0=very different)
- Suggested strategy: {sequence_analysis['suggested_strategy']}

**Optimization Goal:** {optimization_goal}

**Parameter Descriptions:**
1. **column_fraction (-f)**: Remove alignment columns with fewer than this fraction of sequences having a residue.
   - Higher values (0.7-0.8) → More stringent filtering → Shorter, more conserved regions
   - Lower values (0.5-0.6) → Retain more positions → Longer alignments
   - Available range: {self.column_fraction_range}

2. **nterm_conservation (-n)**: Trim N-terminus until this conservation threshold is met.
   - Higher values (0.8-0.9) → Aggressive trimming → Shorter alignments focusing on core
   - Lower values (0.6-0.7) → Retain more N-terminal residues
   - Available range: {self.nterm_conservation_range}

3. **cterm_conservation (-c)**: Trim C-terminus until this conservation threshold is met.
   - Higher values (0.8-0.9) → Aggressive trimming → Shorter alignments focusing on core
   - Lower values (0.6-0.7) → Retain more C-terminal residues
   - Available range: {self.cterm_conservation_range}

**Target Grid Size:** {target_grid_size} parameter combinations (range: {self.min_grid_size}-{self.max_grid_size})

**Task:**
Based on the sequence analysis, propose an intelligent parameter search grid that:
1. Focuses on finding SHORT but WELL-CONSERVED regions (this is the primary goal)
2. Balances comprehensive exploration with computational efficiency
3. Adjusts parameter ranges based on sequence diversity
4. Stays within the target grid size

**Required Output Format (JSON):**
{{
    "parameter_grid": {{
        "column_fraction": [list of 2-4 values from {self.column_fraction_range}],
        "nterm_conservation": [list of 2-3 values from {self.nterm_conservation_range}],
        "cterm_conservation": [list of 2-3 values from {self.cterm_conservation_range}]
    }},
    "rationale": "Brief explanation of parameter choices based on sequence characteristics",
    "expected_outcome": "Description of what type of regions these parameters will identify",
    "estimated_combinations": <number>
}}

Respond ONLY with the JSON object, no additional text.
"""

        return prompt

    def _parse_and_validate_proposal(
        self,
        llm_response: str,
        sequence_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Parse and validate LLM's parameter proposal.

        Args:
            llm_response: Raw LLM response
            sequence_analysis: Sequence analysis results

        Returns:
            Validated parameter proposal
        """
        try:
            # Extract JSON from response (handle markdown code blocks)
            json_match = re.search(r'```json\s*(.*?)\s*```', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON object directly
                json_match = re.search(r'\{.*\}', llm_response, re.DOTALL)
                if json_match:
                    json_str = json_match.group(0)
                else:
                    raise ValueError("No JSON object found in LLM response")

            # Parse JSON
            proposal = json.loads(json_str)

            # Validate structure
            if 'parameter_grid' not in proposal:
                raise ValueError("Missing 'parameter_grid' in proposal")

            param_grid = proposal['parameter_grid']
            required_params = ['column_fraction', 'nterm_conservation', 'cterm_conservation']

            for param in required_params:
                if param not in param_grid:
                    raise ValueError(f"Missing parameter '{param}' in grid")
                if not isinstance(param_grid[param], list):
                    raise ValueError(f"Parameter '{param}' must be a list")
                if len(param_grid[param]) == 0:
                    raise ValueError(f"Parameter '{param}' list is empty")

            # Calculate actual number of combinations
            num_combinations = (
                len(param_grid['column_fraction']) *
                len(param_grid['nterm_conservation']) *
                len(param_grid['cterm_conservation'])
            )

            # Validate grid size
            if num_combinations < self.min_grid_size:
                self.agent_logger.warning(
                    f"Grid size {num_combinations} below minimum {self.min_grid_size}"
                )
            if num_combinations > self.max_grid_size:
                self.agent_logger.warning(
                    f"Grid size {num_combinations} exceeds maximum {self.max_grid_size}, "
                    f"consider reducing parameter ranges"
                )

            # Update estimated combinations
            proposal['estimated_combinations'] = num_combinations

            # Add sequence analysis to result
            proposal['sequence_analysis'] = sequence_analysis

            return proposal

        except json.JSONDecodeError as e:
            self.agent_logger.error(f"Failed to parse JSON from LLM response: {e}")
            self.agent_logger.debug(f"LLM response: {llm_response}")
            raise ValueError(f"Invalid JSON in LLM response: {e}")

        except Exception as e:
            self.agent_logger.error(f"Error validating proposal: {e}")
            raise

    async def _process_specialized_request(self, input_data: Dict[str, Any], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Process specialized parameter proposal requests.

        This method is called by the base class for specialized handling.
        For ParameterProposalAgent, all requests go through the main process() method.
        """
        # Delegate to main process method
        return await self.process(input_data, **kwargs)

    def _should_handle_specialized(self, input_data: Any, **kwargs) -> bool:
        """
        Determine if request should be handled by specialized logic.

        For ParameterProposalAgent, all requests are specialized.
        """
        return True

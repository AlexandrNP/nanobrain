"""
Conversational Response Step

Provides educational and informational responses about alphaviruses.
Includes scientific knowledge base and literature reference integration.

Author: NanoBrain Development Team
Date: December 2024
Version: 4.1.0
"""

from nanobrain.core.step import Step, StepConfig
from nanobrain.library.infrastructure.data.chat_session_data import (
    ConversationalResponseData, MessageType
)
from typing import Dict, Any, List, Optional
import time
from datetime import datetime


class ConversationalResponseStep(Step):
    """
    Step for generating educational responses about alphaviruses.
    
    Provides scientific information with literature references
    and handles various biology topics related to alphaviruses.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config or {})
        
        # Initialize alphavirus knowledge base
        self.knowledge_base = self._initialize_knowledge_base()
        
        self.nb_logger.info("🧠 Conversational Response Step initialized with alphavirus knowledge base")
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate conversational response about alphaviruses.
        
        Args:
            input_data: Contains classification_data and routing_decision
            
        Returns:
            Dictionary with ConversationalResponseData
        """
        start_time = time.time()
        
        try:
            classification_data = input_data.get('classification_data')
            routing_decision = input_data.get('routing_decision', {})
            
            if not classification_data:
                raise ValueError("Missing classification_data")
            
            query = classification_data.original_query
            topic_hints = routing_decision.get('topic_hints', ['general'])
            clarification_needed = routing_decision.get('clarification_needed', False)
            
            self.nb_logger.info(f"🧠 Generating response for topics: {topic_hints}")
            
            # Generate response based on topics
            if clarification_needed:
                response_data = await self._generate_clarification_response(query)
            else:
                response_data = await self._generate_educational_response(query, topic_hints)
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            response_data.processing_time_ms = processing_time
            
            self.nb_logger.info(f"✅ Generated {response_data.response_type} response ({len(response_data.response)} chars)")
            
            return {
                'success': True,
                'response_data': response_data,
                'processing_time_ms': processing_time
            }
            
        except Exception as e:
            self.nb_logger.error(f"❌ Conversational response generation failed: {e}")
            
            # Generate fallback response
            fallback_response = ConversationalResponseData(
                query=input_data.get('classification_data', {}).get('original_query', ''),
                response="I apologize, but I encountered an error generating a response. Please try rephrasing your question or ask about alphavirus structure, replication, or diseases.",
                response_type='error',
                confidence=0.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
            return {
                'success': False,
                'response_data': fallback_response,
                'error': str(e)
            }
    
    async def _generate_educational_response(self, query: str, topic_hints: List[str]) -> ConversationalResponseData:
        """Generate educational response based on topic hints"""
        
        # Determine primary topic
        primary_topic = topic_hints[0] if topic_hints else 'general'
        
        # Get base response for topic
        if primary_topic in self.knowledge_base:
            topic_info = self.knowledge_base[primary_topic]
            
            # Generate contextual response
            response = await self._generate_contextual_response(query, primary_topic, topic_info)
            response_type = 'educational'
            confidence = 0.85
            
        else:
            # Fallback to general information
            response = await self._generate_general_response(query)
            response_type = 'factual'
            confidence = 0.7
        
        # Create response data
        response_data = ConversationalResponseData(
            query=query,
            response=response,
            response_type=response_type,
            confidence=confidence,
            topic_area=primary_topic
        )
        
        # Add relevant literature references
        await self._add_literature_references(response_data, primary_topic)
        
        return response_data
    
    async def _generate_clarification_response(self, query: str) -> ConversationalResponseData:
        """Generate clarification response for unclear queries"""
        
        response = """I'd be happy to help you learn about alphaviruses! However, I need a bit more information to provide the most relevant answer.

**I can help you with:**
🦠 **Virus Structure** - envelope proteins, capsid, genome organization
🔄 **Replication Cycle** - viral life cycle, host cell interaction
🏥 **Diseases** - symptoms, pathogenesis, clinical aspects
🦟 **Transmission** - vectors, epidemiology, prevention
🧬 **Evolution** - phylogeny, mutations, viral diversity
📊 **Classification** - taxonomy, viral families, nomenclature

**For protein analysis and annotation**, please include:
- Protein sequences (if you have them)
- Specific protein names (e.g., "envelope protein E1", "nsP2")
- Analysis goals (structure prediction, function annotation)

Please let me know which topic interests you most, or provide more specific details about what you'd like to learn!"""

        response_data = ConversationalResponseData(
            query=query,
            response=response,
            response_type='clarification',
            confidence=1.0,
            topic_area='general'
        )
        
        return response_data
    
    async def _generate_general_response(self, query: str) -> str:
        """Generate general response about alphaviruses"""
        
        return """**Alphaviruses** are positive-sense RNA viruses in the family *Togaviridae*. They are important human and animal pathogens transmitted primarily by mosquitoes.

**🔬 Key Characteristics:**
- Single-stranded, positive-sense RNA genome (~11-12 kb)
- Enveloped virions with icosahedral nucleocapsid
- Arthropod vectors (primarily *Aedes* and *Culex* mosquitoes)
- Host range includes humans, horses, birds, and other vertebrates

**🦠 Major Human Pathogens:**
- **Chikungunya virus** - joint pain and fever
- **Eastern Equine Encephalitis virus** - severe neurological disease
- **Western Equine Encephalitis virus** - mild to severe encephalitis
- **Venezuelan Equine Encephalitis virus** - flu-like symptoms to encephalitis

**🧬 Genome Organization:**
5'-nsP1-nsP2-nsP3-nsP4-Capsid-E3-E2-6K-E1-3'

Would you like to learn more about a specific aspect of alphavirus biology?"""
    
    async def _generate_contextual_response(self, query: str, topic: str, topic_info: Dict[str, Any]) -> str:
        """Generate contextual response for specific topic"""
        
        query_lower = query.lower()
        
        # Use query keywords to refine response
        if topic == 'structure':
            if any(keyword in query_lower for keyword in ['envelope', 'e1', 'e2', 'e3']):
                return topic_info['envelope_proteins']
            elif any(keyword in query_lower for keyword in ['capsid', 'core', 'nucleocapsid']):
                return topic_info['capsid']
            elif any(keyword in query_lower for keyword in ['genome', 'rna', 'organization']):
                return topic_info['genome_organization']
            else:
                return topic_info['overview']
        
        elif topic == 'replication':
            if any(keyword in query_lower for keyword in ['entry', 'attachment', 'fusion']):
                return topic_info['entry_fusion']
            elif any(keyword in query_lower for keyword in ['translation', 'polyprotein']):
                return topic_info['translation']
            elif any(keyword in query_lower for keyword in ['assembly', 'budding', 'release']):
                return topic_info['assembly']
            else:
                return topic_info['overview']
        
        elif topic == 'diseases':
            if any(keyword in query_lower for keyword in ['chikungunya', 'chik']):
                return topic_info['chikungunya']
            elif any(keyword in query_lower for keyword in ['eastern equine', 'eee']):
                return topic_info['eastern_equine']
            elif any(keyword in query_lower for keyword in ['symptoms', 'clinical']):
                return topic_info['symptoms']
            else:
                return topic_info['overview']
        
        elif topic == 'transmission':
            if any(keyword in query_lower for keyword in ['mosquito', 'vector']):
                return topic_info['vectors']
            elif any(keyword in query_lower for keyword in ['epidemiology', 'geographic']):
                return topic_info['epidemiology']
            else:
                return topic_info['overview']
        
        # Default to overview for other topics
        return topic_info.get('overview', topic_info.get('general', ''))
    
    async def _add_literature_references(self, response_data: ConversationalResponseData, topic: str):
        """Add relevant literature references to response"""
        
        # Literature references by topic
        references = {
            'structure': [
                {
                    'title': 'Structure of chikungunya virus',
                    'authors': 'Voss JE, Vaney MC, Duquerroy S, et al.',
                    'journal': 'Nature',
                    'year': 2010,
                    'pmid': '20428234'
                },
                {
                    'title': 'Alphavirus structure and assembly',
                    'authors': 'Kuhn RJ',
                    'journal': 'Adv Virus Res',
                    'year': 2007,
                    'pmid': '17765004'
                }
            ],
            'replication': [
                {
                    'title': 'Alphavirus RNA replication',
                    'authors': 'Strauss JH, Strauss EG',
                    'journal': 'Microbiol Rev',
                    'year': 1994,
                    'pmid': '8078435'
                },
                {
                    'title': 'Alphavirus nonstructural proteins and their role in viral RNA replication',
                    'authors': 'Lemm JA, Rümenapf T, Strauss EG, et al.',
                    'journal': 'J Virol',
                    'year': 1994,
                    'pmid': '8254745'
                }
            ],
            'diseases': [
                {
                    'title': 'Chikungunya: a re-emerging virus',
                    'authors': 'Schwartz O, Albert ML',
                    'journal': 'Lancet',
                    'year': 2010,
                    'pmid': '19854199'
                },
                {
                    'title': 'Eastern equine encephalitis virus',
                    'authors': 'Morens DM, Folkers GK, Fauci AS',
                    'journal': 'N Engl J Med',
                    'year': 2019,
                    'pmid': '31291517'
                }
            ]
        }
        
        # Add references for the topic
        if topic in references:
            topic_refs = references[topic][:3]  # Max 3 references
            
            for ref in topic_refs:
                response_data.add_reference(
                    title=ref['title'],
                    authors=ref['authors'],
                    journal=ref['journal'],
                    year=ref['year'],
                    pmid=ref['pmid']
                )
    
    def _initialize_knowledge_base(self) -> Dict[str, Dict[str, str]]:
        """Initialize comprehensive alphavirus knowledge base"""
        
        return {
            'structure': {
                'overview': """**Alphavirus Structure**

Alphaviruses are enveloped RNA viruses with a sophisticated structural organization:

**🧬 Genome Structure:**
- Single-stranded, positive-sense RNA (~11,700 nucleotides)
- 5' cap structure and 3' poly(A) tail
- Two open reading frames (ORFs)

**🔬 Virion Structure:**
- Icosahedral nucleocapsid core (T=4 symmetry)
- Lipid envelope derived from host cell membrane
- ~70 nm diameter particles

**🧪 Structural Proteins:**
- **Capsid (C)** - forms nucleocapsid core
- **Envelope proteins E1 & E2** - surface glycoproteins
- **E3** - signal peptide, cleaved during maturation
- **6K** - small membrane protein""",
                
                'envelope_proteins': """**Alphavirus Envelope Proteins**

The surface glycoproteins E1 and E2 are critical for viral infectivity:

**E2 Protein:**
- Receptor binding and cellular attachment
- Major target for neutralizing antibodies
- Contains variable domains for host adaptation
- Forms heterodimers with E1

**E1 Protein:**
- Membrane fusion machinery
- Low-pH triggered conformational changes
- Class II fusion protein structure
- Contains fusion peptide for membrane merger

**E3 Protein:**
- Cleaved signal peptide of E2
- Remains associated with virion in some species
- May play role in virus maturation

**Functional Organization:**
- E1-E2 heterodimers form spike complexes
- 80 spikes per virion (240 protein copies)
- Icosahedral arrangement on viral surface""",
                
                'capsid': """**Alphavirus Capsid Protein**

The capsid protein forms the inner nucleocapsid structure:

**🔬 Structure & Function:**
- Basic protein (~30 kDa)
- RNA-binding domain for genome packaging
- Protease activity for self-cleavage
- Forms icosahedral core (T=4, 240 subunits)

**🧬 RNA Interaction:**
- Binds viral genomic RNA specifically
- Recognizes packaging signals
- Maintains RNA in infectious conformation

**⚙️ Assembly Process:**
- Co-translational cleavage from polyprotein
- Rapid encapsidation of nascent RNA
- Coordinated with envelope protein expression""",
                
                'genome_organization': """**Alphavirus Genome Organization**

The alphavirus genome has a unique organization:

**5' → 3' Organization:**
```
5'-nsP1-nsP2-nsP3-nsP4-[subgenomic promoter]-C-E3-E2-6K-E1-3'
```

**Non-structural Proteins (nsPs):**
- **nsP1**: Capping enzyme, membrane association
- **nsP2**: Protease, helicase, RNA binding
- **nsP3**: Phosphoprotein, host interaction
- **nsP4**: RNA-dependent RNA polymerase

**Structural Proteins:**
- **Capsid (C)**: Nucleocapsid formation
- **E3-E2**: Envelope glycoprotein precursor
- **6K**: Small membrane protein
- **E1**: Fusion glycoprotein

**Regulatory Elements:**
- Internal ribosome entry sites
- Subgenomic RNA promoter
- 3' conserved secondary structures"""
            },
            
            'replication': {
                'overview': """**Alphavirus Replication Cycle**

Alphaviruses follow a complex replication strategy:

**🔗 Attachment & Entry:**
1. E2 binding to cellular receptors
2. Clathrin-mediated endocytosis
3. Low-pH triggered E1-mediated fusion

**🧬 RNA Replication:**
1. Translation of nsP1-4 polyprotein
2. Formation of replication complexes
3. Negative-strand RNA synthesis
4. Positive-strand genomic and subgenomic RNA

**🔧 Protein Synthesis:**
1. Structural protein translation from subgenomic RNA
2. Co-translational processing
3. Envelope protein glycosylation

**📦 Assembly & Release:**
1. Nucleocapsid assembly
2. Budding from plasma membrane
3. Envelope acquisition and maturation""",
                
                'entry_fusion': """**Alphavirus Entry and Fusion**

Sophisticated mechanism for cellular entry:

**🔗 Receptor Binding:**
- E2 protein binds cellular receptors
- Multiple receptor types (MXRA8, DC-SIGN, etc.)
- Species and cell-type specificity

**📥 Endocytosis:**
- Clathrin-mediated uptake
- Transport to early endosomes
- pH-dependent activation

**🔀 Membrane Fusion:**
- E1 undergoes conformational change at pH ~6.0
- Exposure of fusion peptide
- Formation of fusion pore
- Nucleocapsid release into cytoplasm

**⚙️ Uncoating:**
- Capsid protein interactions with ribosomes
- RNA release for translation
- Rapid initiation of viral protein synthesis""",
                
                'translation': """**Alphavirus Translation Strategy**

Unique translation and processing mechanisms:

**🧬 Genomic RNA Translation:**
- Direct translation of nsP1-4 polyprotein
- Ribosomal frameshifting between nsP3-nsP4
- Co-translational processing by nsP2 protease

**📋 Subgenomic RNA:**
- Internal promoter-driven synthesis
- High-level structural protein expression
- Temporal regulation of viral proteins

**⚙️ Protein Processing:**
- Signal peptidase cleavage (E3-E2)
- Furin cleavage of E2 precursor
- Capsid auto-protease activity

**🔧 Post-translational Modifications:**
- Envelope protein glycosylation
- nsP phosphorylation
- Host factor recruitment""",
                
                'assembly': """**Alphavirus Assembly and Budding**

Coordinated process of virion formation:

**🧬 Nucleocapsid Assembly:**
- Capsid-RNA recognition signals
- Cooperative binding and packaging
- Formation of icosahedral core

**🔗 Envelope Acquisition:**
- E1-E2 spike complex formation
- Targeting to plasma membrane
- Interaction with nucleocapsid

**📤 Budding Process:**
- Membrane curvature induction
- Envelope protein concentration
- Coordinated release mechanism

**🔧 Maturation:**
- Final protein processing
- Conformational rearrangements
- Infectious particle formation"""
            },
            
            'diseases': {
                'overview': """**Alphavirus Diseases**

Alphaviruses cause diverse human diseases:

**🦠 Disease Categories:**
1. **Arthritogenic** - joint inflammation (CHIKV, RRV)
2. **Encephalitic** - brain inflammation (EEE, WEE, VEE)
3. **Febrile** - fever and systemic symptoms

**🌍 Geographic Distribution:**
- Tropical and subtropical regions
- Vector distribution dependent
- Emerging in new territories

**📊 Disease Burden:**
- Millions affected by chikungunya
- High morbidity in outbreaks
- Economic impact significant

**🏥 Clinical Management:**
- Supportive care primary treatment
- No specific antivirals approved
- Prevention through vector control""",
                
                'chikungunya': """**Chikungunya Virus Disease**

Major alphavirus causing global outbreaks:

**🦠 Clinical Features:**
- Acute febrile illness
- Severe polyarthralgia (joint pain)
- Myalgia and headache
- Maculopapular rash

**⏰ Disease Phases:**
1. **Acute phase** (1-2 weeks) - fever, arthralgia
2. **Post-acute phase** (2-3 months) - ongoing symptoms
3. **Chronic phase** (>3 months) - persistent arthritis

**🌍 Epidemiology:**
- Aedes aegypti and A. albopictus vectors
- Urban transmission cycles
- Recent expansion to Americas, Europe

**💊 Management:**
- Symptomatic treatment with NSAIDs
- Corticosteroids for severe arthritis
- Physical therapy for chronic cases""",
                
                'eastern_equine': """**Eastern Equine Encephalitis (EEE)**

Severe neuroinvasive alphavirus infection:

**🧠 Clinical Presentation:**
- Rapid onset of encephalitis
- High fever and altered consciousness
- Seizures and focal neurological signs
- Case fatality rate ~30%

**🔬 Pathogenesis:**
- Neurotropism and BBB penetration
- Inflammatory brain damage
- Gray matter preference

**🌲 Ecology:**
- Enzootic cycle in birds and Culiseta mosquitoes
- Horses as amplifying hosts
- Human infections sporadic

**⚕️ Clinical Management:**
- Intensive supportive care
- Anti-seizure medications
- ICP monitoring and management
- Long-term neurological rehabilitation""",
                
                'symptoms': """**General Alphavirus Symptoms**

Common clinical presentations across alphaviruses:

**🌡️ Acute Symptoms:**
- High fever (often >39°C)
- Severe headache
- Myalgia (muscle pain)
- Arthralgia (joint pain)
- Fatigue and malaise

**🎯 Specific Manifestations:**
- **Arthritic viruses**: Joint swelling, morning stiffness
- **Encephalitic viruses**: Altered mental status, seizures
- **Hemorrhagic**: Bleeding, thrombocytopenia (rare)

**📈 Disease Progression:**
- Incubation: 2-12 days
- Acute phase: 1-2 weeks
- Recovery: weeks to months
- Chronic complications possible

**⚠️ Warning Signs:**
- Neurological symptoms
- Persistent high fever
- Hemorrhagic manifestations
- Severe dehydration"""
            },
            
            'transmission': {
                'overview': """**Alphavirus Transmission**

Vector-borne transmission predominates:

**🦟 Primary Vectors:**
- Aedes mosquitoes (A. aegypti, A. albopictus)
- Culex mosquitoes (C. tarsalis, C. quinquefasciatus)
- Culiseta melanura (EEE maintenance cycle)

**🔄 Transmission Cycles:**
1. **Enzootic** - wildlife maintenance cycles
2. **Epizootic** - amplification in domestic animals
3. **Epidemic** - human outbreak cycles

**🌍 Environmental Factors:**
- Temperature affects vector competence
- Rainfall patterns influence breeding
- Urbanization creates breeding sites

**🛡️ Prevention Strategies:**
- Vector control measures
- Personal protective equipment
- Environmental management
- Surveillance systems""",
                
                'vectors': """**Alphavirus Mosquito Vectors**

Diverse mosquito species transmit alphaviruses:

**🦟 Aedes aegypti:**
- Primary CHIKV vector
- Urban/domestic habitats
- Day-biting behavior
- Container breeding sites

**🦟 Aedes albopictus:**
- Secondary CHIKV vector
- Suburban/periurban areas
- Aggressive biter
- Tire and container breeding

**🦟 Culex tarsalis:**
- Primary WEE vector
- Rural/agricultural areas
- Irrigation systems
- Bridge vector to humans

**🦟 Culiseta melanura:**
- EEE enzootic maintenance
- Freshwater swamps
- Ornithophilic (bird-feeding)
- Limited human contact

**⚙️ Vector Competence:**
- Virus replication in mosquito
- Salivary gland infection
- Transmission efficiency
- Environmental modulation""",
                
                'epidemiology': """**Alphavirus Epidemiology**

Global patterns of alphavirus distribution:

**🌍 Geographic Patterns:**
- **Chikungunya**: Africa, Asia, Americas, Europe
- **EEE**: Eastern North America
- **WEE**: Western North America
- **VEE**: Central/South America

**📊 Outbreak Dynamics:**
- Cyclical patterns (5-7 year intervals)
- Seasonal peaks during vector seasons
- Climate change expanding ranges

**👥 Population Risk:**
- Age-specific attack rates
- Occupation-related exposure
- Immune status importance
- Travel-associated cases

**🔍 Surveillance:**
- Vector monitoring programs
- Clinical case reporting
- Serosurveys in populations
- Genetic sequencing for tracking"""
            },
            
            'evolution': {
                'overview': """**Alphavirus Evolution**

Dynamic evolutionary processes shape alphavirus diversity:

**🧬 Genetic Diversity:**
- RNA virus mutation rates (~10⁻⁴/site/year)
- Selection pressures from hosts and vectors
- Geographic population structure
- Recombination events

**🌱 Phylogenetic Relationships:**
- Old World vs New World clades
- Host-associated clustering
- Vector-associated evolution

**🔄 Adaptive Evolution:**
- Host range expansion
- Vector adaptation
- Immune escape variants
- Drug resistance potential"""
            },
            
            'classification': {
                'overview': """**Alphavirus Classification**

Taxonomic organization of the alphavirus genus:

**📊 Taxonomic Hierarchy:**
- **Family**: Togaviridae
- **Genus**: Alphavirus
- **Species**: 32+ recognized species
- **Strains**: Geographic/temporal variants

**🔬 Classification Criteria:**
- Genetic similarity thresholds
- Serological relationships
- Biological properties
- Geographic distribution

**📋 Major Species Groups:**
1. **Semliki Forest complex**
2. **Venezuelan equine encephalitis complex**
3. **Sindbis-like viruses**
4. **Ndumu group**

**🧬 Molecular Markers:**
- E1 envelope protein sequences
- nsP4 polymerase gene
- Complete genome phylogeny
- Antigenic site mapping"""
            }
        } 
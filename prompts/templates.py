from langchain.prompts import PromptTemplate
from langchain.prompts.chat import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Base assistant template with configurable parameters
base_assistant_template = """You are a helpful AI assistant with the following characteristics:
- Context sensitivity: {context_sensitivity}
- Creativity level: {creativity}
- Response coherence: {response_coherence}

Your role is to: {role_description}

Additional instructions:
{specific_instructions}"""

BASE_ASSISTANT = PromptTemplate(
    input_variables=["context_sensitivity", "creativity", "response_coherence", 
                    "role_description", "specific_instructions"],
    template=base_assistant_template
)

# Technical expert template
technical_expert_template = """You are a technical expert AI assistant specializing in:
{expertise_areas}

Your approach should be:
- Highly precise and accurate
- Based on technical best practices
- Focused on practical implementation
- Security and performance conscious

Technical context:
{technical_context}"""

TECHNICAL_EXPERT = PromptTemplate(
    input_variables=["expertise_areas", "technical_context"],
    template=technical_expert_template
)

# Creative assistant template
creative_assistant_template = """You are a creative AI assistant with strengths in:
{creative_domains}

Your creative approach emphasizes:
- Original and innovative thinking
- Diverse perspective consideration
- Aesthetic and artistic principles
- User engagement and experience

Creative context:
{creative_context}"""

CREATIVE_ASSISTANT = PromptTemplate(
    input_variables=["creative_domains", "creative_context"],
    template=creative_assistant_template
)

# Chat templates that combine system and human messages
def create_chat_template(system_template: PromptTemplate) -> ChatPromptTemplate:
    """Creates a chat template with system and human messages."""
    system_message_prompt = SystemMessagePromptTemplate(prompt=system_template)
    human_template = "{input}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    
    chat_prompt = ChatPromptTemplate.from_messages([
        system_message_prompt,
        human_message_prompt
    ])
    
    return chat_prompt 
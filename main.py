import asyncio
from llm_selector import LLMSelector, LLMSelectorConfig

async def main():
    # Configuration des modèles disponibles avec poids
    models = {
        "openai/gpt-4": {
            "description": "OpenAI GPT-4 for complex reasoning and coding",
            "weight": 2.0  # Plus de chances d'être sélectionné
        },
        "anthropic/claude-3": {
            "description": "Anthropic Claude 3 for safe and helpful responses", 
            "weight": 1.5
        },
        "cerebras/llama3.3-70b": {
            "description": "Cerebras Llama 3.3 70B for fast inference",
            "weight": 1.0
        },
        "openai/gpt-3.5-turbo": {
            "description": "OpenAI GPT-3.5 Turbo for quick tasks",
            "weight": 0.8  # Moins de chances
        },
        "anthropic/claude-2": {
            "description": "Anthropic Claude 2 for balanced performance",
            "weight": 1.0
        },
    }
    
    # Configurations des services de sélection avec fallbacks
    # Les clés API peuvent être :
    # - En clair: "sk-your-key"
    # - Variable d'environnement: "env:OPENROUTER_API_KEY"
    # - Clé dans .env: "dotenv:OPENROUTER_KEY"
    selector_configs = [
        LLMSelectorConfig(
            model="openrouter/openai/gpt-oss-20b",
            api_base="https://openrouter.ai/api/v1",
            api_key="env:OPENROUTER_API_KEY"  # Utilise la variable d'environnement
        ),
        LLMSelectorConfig(
            model="anthropic/claude-3-haiku",
            api_key="dotenv:ANTHROPIC_KEY"  # Utilise la clé dans .env
        ),
        LLMSelectorConfig(
            model="openai/gpt-3.5-turbo",
            api_key="sk-your-openai-key"  # Clé en clair
        ),
    ]
    
    selector = LLMSelector(models=models, selector_configs=selector_configs)
    user_input = "I need to write a complex Python script for data analysis."
    selected_model = await selector.select(user_input)
    print(f"Selected model: {selected_model}")

if __name__ == "__main__":
    asyncio.run(main())
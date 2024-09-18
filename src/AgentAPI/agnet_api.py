from lyzr_agent_api import AgentAPI, EnvironmentConfig, FeatureConfig, AgentConfig, ChatRequest
import os
from utils import load_ids, save_ids, compare_configs
from dotenv import load_dotenv

load_dotenv()

ID_FILE = os.getenv('ID_FILE')
X_API_Key = os.getenv('X_API_Key')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
USER_ID = os.getenv('USER_ID')
SESSION_ID = os.getenv('SESSION_ID')


# Initializing Agent API client
api_client = AgentAPI(x_api_key=X_API_Key)


# This function will create the environment based on the given configuration
def create_env(env_name, openai_api_key=OPENAI_API_KEY):
    env_config = EnvironmentConfig(
        name=env_name,
        features=[
            FeatureConfig(
                type="SHORT_TERM_MEMORY",
                config={"max_tries": 3},
                priority=0,
            ),
            FeatureConfig(
                type="TOOL_CALLING",
                config={"max_tries": 3},
                priority=0
            ),
            FeatureConfig(
                type="HUMANIZER",
                config={"max_tries": 3},
                priority=0
            )
        ],
        tools=["perplexity_search","send_email"],
        llm_config={"provider": "openai",
                    "model": "gpt-4o-mini",
                    "config": {
                        "temperature": 0.5,
                        "top_p": 0.9
                    },
                    "env": {
                        "OPENAI_API_KEY": openai_api_key
                    }},
    )

    response = api_client.create_environment_endpoint(json_body=env_config)
    return response['env_id'], env_config


def create_agent(env_id, agent_name, agent_description, agent_sys_prompt):
    agent_config = AgentConfig(
        env_id=env_id,
        system_prompt=agent_sys_prompt,
        name=agent_name,
        agent_description=agent_description
    )

    agent_response = api_client.create_agent_endpoint(json_body=agent_config)
    return agent_response['agent_id'], agent_config


# Check for existing environment and agent based on configuration
def get_or_create_env_and_agent(env_name, agent_name, agent_description, agent_sys_prompt):
    # Load stored IDs and configurations
    stored_data = load_ids(IDfile=ID_FILE)
    
    if stored_data:
        # Check for existing environment with the same config
        for env_id, data in stored_data.items():
            if compare_configs(data['env_config']['name'], env_name) and compare_configs(data['agent_config']['name'], agent_name):
                print("Using stored environment and agent IDs")
                return env_id, data['agent_id']

    # If no matching environment/agent found, create new ones
    print("Creating new environment and agent")
    env_id, env_config = create_env(env_name)
    agent_id, agent_config = create_agent(env_id, agent_name, agent_description, agent_sys_prompt)

    # Save the new configuration and IDs
    save_ids(env_id, env_config, agent_id, agent_config, IDfile=ID_FILE)

    return env_id, agent_id


def creating_chat(agent_id, message):
    json_body = ChatRequest(
        user_id=USER_ID,
        agent_id=agent_id,
        message=message,
        session_id=SESSION_ID
    )

    chat = api_client.chat_with_agent(json_body=json_body)
    return chat['response']

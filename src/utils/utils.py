import os
import json
import streamlit as st
import shutil
import base64
from openai import OpenAI
from st_social_media_links import SocialMediaIcons 



# Utility function to check if two configurations are identical
def compare_configs(stored_config, new_config):
    return stored_config == new_config

# Convert EnvironmentConfig to a dictionary
def env_config_to_dict(env_config):
    return {
        'name': env_config.name,
        'features': [feature.__dict__ for feature in env_config.features],
        'tools': env_config.tools,
        'llm_config': env_config.llm_config
    }

# Convert AgentConfig to a dictionary
def agent_config_to_dict(agent_config):
    return {
        'env_id': agent_config.env_id,
        'system_prompt': agent_config.system_prompt,
        'name': agent_config.name,
        'agent_description': agent_config.agent_description
    }

# Save environment and agent configurations with IDs to the JSON file
def save_ids(env_id, env_config, agent_id, agent_config, IDfile):
    data = load_ids(IDfile) or {}

    # Save the environment and agent config alongside their IDs
    data[env_id] = {
        'env_config': env_config_to_dict(env_config),
        'agent_id': agent_id,
        'agent_config': agent_config_to_dict(agent_config)
    }

    with open(IDfile, 'w') as f:
        json.dump(data, f, indent=4)

# Load the stored configurations and IDs from the JSON file
def load_ids(IDfile):
    if os.path.exists(IDfile):
        with open(IDfile, 'r') as f:
            return json.load(f)
    return None

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  

def gpt_vision_call(openai_api_key, base64_image):
    openai_client = OpenAI(api_key=openai_api_key)
    prompt = "Please analyze the following leaf image and provide a detailed description of any visible problems, such as diseases, pests, nutrient deficiencies, or other signs of stress. Focus on identifying the specific issues affecting the plant's health and explain what might be causing them."

    messages = [
                {"role": "user", "content": [
                    {"type": "text",
                     "text": prompt
                     },
                    {
                      "type": "image_url",
                      "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                      },
                    },
                    ]
                 },
            ]

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages
    )

    description = response.choices[0].message.content

    return description


def remove_existing_files(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            st.error(f"Error while removing existing files: {e}")



def get_files_in_directory(directory):
    # This function help us to get the file path along with filename.
    files_list = []

    if os.path.exists(directory) and os.path.isdir(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path):
                files_list.append(file_path)

    return files_list


def save_uploaded_file(directory, uploaded_file):
    remove_existing_files(directory=directory)
    file_path = os.path.join(directory, uploaded_file.name)
    with open(file_path, "wb") as file:
        file.write(uploaded_file.read())


def get_file_name(directory):
    try:
        files = os.listdir(directory)
        file_names = [file for file in files if os.path.isfile(os.path.join(directory, file))]
        
        return file_names[0]
    
    except FileNotFoundError:
        return f"The directory '{directory}' does not exist."
    
    except Exception as e:
        return f"An error occurred: {e}"


def file_checker(directoryName):
    file = []
    for filename in os.listdir(directoryName):
        file_path = os.path.join(directoryName, filename)
        file.append(file_path)

    return file

def social_media(justify=None):
    # This function will help you to render socila media icons with link on the app
    social_media_links = [
    "https://github.com/LyzrCore/lyzr",
    "https://www.youtube.com/@LyzrAI",
    "https://www.instagram.com/lyzr.ai/",
    "https://www.linkedin.com/company/lyzr-platform/posts/?feedView=all"
                        ]   

    social_media_icons = SocialMediaIcons(social_media_links)
    social_media_icons.render(sidebar=True, justify_content=justify) # will render in the sidebar



def style_app():
    # You can put your CSS styles here
    st.markdown("""
    <style>
    .app-header { visibility: hidden; }
    .css-18e3th9 { padding-top: 0; padding-bottom: 0; }
    .css-1d391kg { padding-top: 1rem; padding-right: 1rem; padding-bottom: 1rem; padding-left: 1rem; }
    [data-testid="stSidebar"][aria-expanded="true"]{
           min-width: 450px;
           max-width: 450px;
           background-color: #E6F4EA;
       }
    </style>
    """, unsafe_allow_html=True)



def page_config(layout = "centered"):
    st.set_page_config(
        page_title="LeafCare AI:",
        layout=layout,  # or "wide" 
        initial_sidebar_state="auto",
        page_icon="./logo/lyzr-logo-cut.png"
    )

def about_app():
    with st.sidebar.expander("ℹ️ - Why this LeafCare AI"):
        st.sidebar.caption("""LeafCare AI is a user-friendly app that helps identify plant health issues by analyzing images of leaves. Users can upload a leaf image, and the app leverages AI to diagnose problems and provide treatment recommendations, making plant care simpler and more efficient.
        """)


def template_end():
    st.sidebar.markdown("### This app is build by using Lyzr's Agent API ")

    st.sidebar.markdown(
        """
        <style>
        .button-container {
            display: flex;
            flex-direction: row;
            justify-content: space-between;
        }
        .button-column {
            flex: 1;
            margin-right: 5px;
        }
        .button-column:last-child {
            margin-right: 0;
        }
        .sidebar-button {
            display: block;
            width: 100%;
            padding: 10px;
            margin: 5px 0;
            text-align: center;
            color: white;
            background-color: #ffffff;
            border: none;
            border-radius: 5px;
            text-decoration: none;
        }
        .sidebar-button:hover {
            background-color: #7458E8;
        }
        </style>

        <div class="button-container">
            <div class="button-column">
                <a class="sidebar-button" href="https://www.lyzr.ai/" target="_blank">Lyzr</a>
                <a class="sidebar-button" href="https://www.lyzr.ai/book-demo/" target="_blank">Book a Demo</a>
                <a class="sidebar-button" href="https://agent.api.lyzr.app/docs#overview" target="_blank">Lyzr Agent API</a>
                <a class="sidebar-button" href="https://discord.gg/nm7zSyEFA2" target="_blank">Discord</a>
                <a class="sidebar-button" href="https://join.slack.com/t/genaiforenterprise/shared_invite/zt-2a7fr38f7-_QDOY1W1WSlSiYNAEncLGw" target="_blank">Slack</a>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )




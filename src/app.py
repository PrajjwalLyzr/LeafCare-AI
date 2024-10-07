from AgentAPI import get_or_create_env_and_agent, creating_chat
from lyzr_agent_api import AgentAPI, ChatRequest
import streamlit as st
import os
from utils import page_config, style_app, template_end, about_app, social_media
from utils import remove_existing_files, save_uploaded_file, get_file_name
from utils import encode_image, gpt_vision_call
from dotenv import load_dotenv

load_dotenv()

Agent_id = os.getenv('Agent_ID')
lyzr_api_key = os.getenv('X_API_Key')
User_id = os.getenv('USER_ID')
Session_id = os.getenv('SESSION_ID')


api_client = AgentAPI(x_api_key=lyzr_api_key)

page_config()
style_app()

ImageData = "ImageData"
os.makedirs(name=ImageData, exist_ok=True)

image = "src/logo/lyzr-logo-cut.png"
st.image(image=image, width=100)

st.header("LeafCare AI")
st.markdown("##### Powered by [Lyzr Agent API](https://agent.api.lyzr.app/docs#overview)")
st.markdown('---')

image_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

if image_file:
    if st.button('Diagnose'):
        remove_existing_files(directory=ImageData)
        save_uploaded_file(directory=ImageData, uploaded_file=image_file)
        file_name = get_file_name(directory=ImageData)
        image_file_path = os.path.join(ImageData, file_name)
        st.image(image=image_file_path, width=100)
        st.markdown('---')
        with st.spinner("🤖 Diagnosis is been processing"):
            base64_image = encode_image(image_file_path)
            leafimageDescription = gpt_vision_call(openai_api_key=os.getenv('OPENAI_API_KEY'),
                                               base64_image=base64_image)
            with st.spinner("Getting the Image description"):
                if leafimageDescription:
                    # environmentName = "Agriculture Environment"
                    # agentName = "Leaf Disease Detector"
                    # agentDescription = "An agent which can detect disease in leafs and provide the treatment accordingly"
                    # agentSystemPrompt = """
                    #                         Task 1: Problem Identification
                    #                         Analyze the leaf image description: {leafImageDescription}. Identify any visible issues such as diseases, nutrient deficiencies, pests, or environmental stress. Provide a detailed diagnosis of the problems affecting the plant’s health.

                    #                         Task 2: Treatment Recommendations and Analysis
                    #                         For the identified problems from Task 1, provide the following:

                    #                         Treatment Recommendations: Specific actions or products to treat or mitigate the issue.
                    #                         Causes: Explanation of the root cause (e.g., pest infestation, nutrient deficiency, fungal infection).
                    #                         Why This Happens: Insights into contributing factors (e.g., environmental conditions, improper care).
                    #                         Task 3: Pesticide/Fertilizer Search and Recommendations
                    #                         Find relevant pesticides, fertilizers, or other products to address the diagnosed issues. Provide links or detailed recommendations for their use, ensuring they are suitable for the problems identified in Task 1.

                    #                         [Important] After completing all tasks, combine the information into a clear and concise output. Ensure it is easy to understand, and include any links with descriptions. Keep the explanation simple and straightforward.
                    #                         """   

                    # env_id, agent_id = get_or_create_env_and_agent(
                    #     env_name=environmentName,
                    #     agent_name=agentName,
                    #     agent_description=agentDescription,
                    #     agent_sys_prompt=agentSystemPrompt
                    # ) 

                    
                    chat_json = ChatRequest(
                        user_id=User_id,
                        agent_id=Agent_id,
                        message=f"This is the leaf image description:{leafimageDescription}. Provide clear and concise treatment recommendations, including potential products like fungicides, bactericides, fertilizers, and insecticides. Ensure the output includes sections for 'Identified Issues' and 'Suggested Treatments,' and list specific products or actions to remedy the plant's health issues. Use perplexity to find the product regarding treatment, if found then give the link.",
                        session_id=Session_id
                    )

                    recommendations = api_client.chat_with_agent(json_body=chat_json)


                    if recommendations:
                        st.write(recommendations['response']) 


else:
    remove_existing_files(directory=ImageData)
    st.warning('Please Upload Leaf Image')


template_end()
st.sidebar.markdown('---')
about_app()
st.sidebar.markdown('---')
social_media(justify="space-evenly")
import streamlit as st
import pandas as pd
from google.cloud import aiplatform
from google.cloud import aiplatform_v1
import vertexai
from vertexai.language_models import TextEmbeddingModel
import moviepy.editor as mpe
from moviepy.editor import VideoFileClip
import os
from vertexai.preview.generative_models import GenerativeModel, Part
import requests
import google.auth
import google.auth.transport.requests
from typing import Optional
from google.cloud import resourcemanager_v3
from google.cloud import discoveryengine

# Define constants
PROJECT_ID = "ai-sb-test" #Replace with your own Project ID
LOCATION = "us-central1"
API_ENDPOINT = f"{LOCATION}-aiplatform.googleapis.com"
SHOTS_FILE_PATH = "./content/shots.csv"
CLIPS_DIR = "./"



# def list_deployed_indexes():
#     """List all deployed indexes in Vertex AI Matching Engine."""
#     client = aiplatform.gapic.IndexEndpointServiceClient(
#         client_options={"api_endpoint": API_ENDPOINT}
#     )
#     parent = f"projects/{PROJECT_ID}/locations/{LOCATION}"
#     index_list = []

#     # List all index endpoints
#     for index_endpoint in client.list_index_endpoints(parent=parent):
#         for deployed_index in index_endpoint.deployed_indexes:
#             index_list.append({
#                 "index_endpoint_name": index_endpoint.name,
#                 "deployed_index_id": deployed_index.id,
#                 "display_name": index_endpoint.display_name
#             })

#     return index_list

def get_embeddings(query, model):
    """Generate embeddings for the query using the specified model."""
    embeddings = model.get_embeddings([query])
    return embeddings[0].values

def load_shots_df(file_path):
    """Load shots_df from a local CSV file."""
    return pd.read_csv(file_path)

def generate_pro(input_prompt):
    model = GenerativeModel("gemini-1.5-pro-002")
    responses = model.generate_content(
    input_prompt,
    generation_config={
        "max_output_tokens": 2048,
        "temperature": 0.2,
        "top_p": 1
    },stream=True,)

    all_response  = []

    for response in responses:
        all_response.append(response.text)

    return(" ".join(all_response))

def combine_column_to_string(df, column_name):

    column_values = df[column_name].tolist()
    combined_string = ', '.join(column_values)
    return combined_string

def combine_two_columns_to_string(df, column_name, column_2_name):
    combined_strings = []
    for clip_name, description in zip(df[column_name], df[column_2_name]):
        combined_string = f"{clip_name}: {description}"
        combined_strings.append(combined_string)
    
    return ", ".join(combined_strings)

def get_project_number(project_id) -> Optional[str]:
    """Given a project id, return the project number"""
    # Create a client
    client = resourcemanager_v3.ProjectsClient()
    # Initialize request argument(s)
    request = resourcemanager_v3.SearchProjectsRequest(query=f"id:{project_id}")
    # Make the request
    page_result = client.search_projects(request=request)
    # Handle the response
    for response in page_result:
        if response.project_id == project_id:
            project = response.name
            return project.replace('projects/', '')


def queryDatastore(query):
    
    project_number = get_project_number(PROJECT_ID)
    
    search_engine_id = "jtc-agent_1730170524209" #Replace with your own Agent Builder ID
    
    search_url = f"https://discoveryengine.googleapis.com/v1alpha/projects/{project_number}/locations/global/collections/default_collection/engines/{search_engine_id}/servingConfigs/default_search:search"
    
    creds, project = google.auth.default()

    auth_req = google.auth.transport.requests.Request()
    creds.refresh(auth_req)

    auth_token = creds.token
    print("Auth Token:", creds.token)

    search_payload = {
        "query": query,
        "pageSize":10,
        "queryExpansionSpec":{"condition":"AUTO"},
        "spellCorrectionSpec": {"mode": "AUTO"},
    }
    
    # Set headers with authorization and content type
    headers = {
        "Authorization": f"Bearer {auth_token}",
        "Content-Type": "application/json"
    }

    # Send the POST request
    search_response = requests.post(search_url, headers=headers, json=search_payload)
    
    search_response_data = search_response.json()
    
    return search_response_data["results"]


vertexai.init(project=PROJECT_ID, location=LOCATION)
text_embedding_model = TextEmbeddingModel.from_pretrained("textembedding-gecko@latest")



# INSTANTIATE TABS
# mainTab1, mainTab2, mainTab3 = st.tabs(["Video Search", "Video List", "Data Source"])
mainTab1, mainTab2 = st.tabs(["Video Search", "Video List"])


# INSTANTIATE TAB 3 FIRST TO DECLARE DATA SOURCE (DEPLOYED INDEX)
# with mainTab3:

    # shots_df = load_shots_df(SHOTS_FILE_PATH)
#     # if shots_df.empty:
#     #     st.write('shots_df', shots_df)

#     # tab1, tab2 = st.tabs(["Select Index", "Query Clips"])

#     # with tab1:
#     st.header("Select a Deployed Index")
#     deployed_indexes = list_deployed_indexes()
#     if deployed_indexes:
#         index_options = [f"{index['display_name']} ({index['deployed_index_id']})" for index in deployed_indexes]
#         selected_option = st.selectbox("Choose an index to query from:", index_options)
#         if selected_option:
#             selected_index = next(index for index in deployed_indexes if f"{index['display_name']} ({index['deployed_index_id']})" == selected_option)
#             st.session_state['selected_index'] = selected_index
#             st.success(f"Selected index: {selected_index['display_name']} with ID {selected_index['deployed_index_id']}")
#     else:
#         st.warning("No deployed indexes found.")
        
        
# INSTANTIATE TAB 1 AFTERWARDS TO PREVENT ERROR
with mainTab1:

    shots_df = load_shots_df(SHOTS_FILE_PATH)
    
    st.title("JTC Video Search App")

#     deployed_indexes = list_deployed_indexes()
#     if deployed_indexes:
#         selected_index = next(index for index in deployed_indexes if f"{index['display_name']} ({index['deployed_index_id']})" == selected_option)
        
#         st.session_state['selected_index'] = selected_index
        
#         selected_index = st.session_state['selected_index']
        
    query = st.text_input("Enter your query:")

    if query:
        try:

            System_Prompts = """ You are an expert in screening video descriptions and understanding the context and contents of the video.
Only answer based on the description of the video provided here: 
"""

            Question_Prompts = """ -- Based on the video description provided below, read through everything in detail before answering the following query as accurately as possible, while keeping your answer concise. If asked to find clips containing certain people, read through every description carefully to identify any mention or appearances of said people. If asked to provide clips relating to a certain subject, do not mention the clips or video names, simply summarise the subject in 5 lines:
            """

#             Clips_Prompts = """ -- If you are providing clip numbers, include the video and clip names in reference to the below table. Present each video name on a new line, and associate all relevant clip numbers.

#             """

#             clipPair = combine_two_columns_to_string(shots_df,"clip_name","description")


            videoDesc = combine_column_to_string(shots_df,'description') + combine_column_to_string(shots_df,'associated_text') + combine_column_to_string(shots_df,'associated_speech') + combine_column_to_string(shots_df,'associated_object')


            combined_prompt = System_Prompts + ' ' + videoDesc + ' ' + Question_Prompts + ' ' + query    # + ' ' + Clips_Prompts + ' ' + clipPair

            gemini_answer = generate_pro(combined_prompt)

            st.write(gemini_answer)
            
            st.text("")
            st.text("")

            try:
                results = queryDatastore(query)
            except:
                results = None
            
            
            if results:
                
                rankClient = discoveryengine.RankServiceClient()
                
                ranking_config = rankClient.ranking_config_path(
                    project=PROJECT_ID,
                    location="global",
                    ranking_config="default_ranking_config",
                )
                
                records = []
                
                for video in results:
                    clipName = video.get("document",{}).get("structData",{}).get("clip_name","Invalid Clip")
                    
                    content = video['document']['structData']["description"] + video['document']['structData']["associated_text"] + video['document']['structData']["associated_object"] + video['document']['structData']["associated_speech"]
                    
                    record = discoveryengine.RankingRecord(
                        id = str(video["id"]),
                        title = clipName,
                        content = content
                    )
                    
                    records.append(record)
                    
                
                request = discoveryengine.RankRequest(
                    ranking_config=ranking_config,
                    model="semantic-ranker-512@latest",
                    top_n=15,
                    query=query,
                    records=records
                )
                
               
                rankResponse = rankClient.rank(request=request)

                rankedVideos = rankResponse.records
            
                scoreVideoList = []
                finalVideos = []
                
                for i in range(rankedVideos.__len__()):
                    video = rankedVideos.pop()
                    
                    videoDict = {}
                    
                    videoDict['id'] = video.id
                    videoDict['title'] = video.title
                    videoDict['score'] = video.score
                    videoDict['content'] = video.content
                    
                    scoreVideoList.append(videoDict)
                     
                
                for record in scoreVideoList:
                    if record['score'] >= 0.33:
                        dict = {}
                        
                        dict['title'] = record['title']
                        dict['score'] = record['score']
                        
                        finalVideos.append(dict)
                
                finalVideos.reverse()
                
                
                        
                for video in finalVideos:
                    
                    videoTitle = video['title']
                    clipName = videoTitle.split('/')[-1]
                    clipScore = int((round(video['score'],3)*100))
                    
                    st.write(f"**Clip Name:** {clipName}")
                    st.write(f'**Confidence Score:** {clipScore}%')
                    video_path = CLIPS_DIR + videoTitle
                    st.video(video_path)
                             
                    
                
            else:
                st.write("No relevant clips found.")

                
            
                    
                


#                 query_embedding = get_embeddings(query, text_embedding_model)

#                 selected_index = st.session_state['selected_index']
#                 deployed_index_id = selected_index['deployed_index_id']
#                 index_endpoint_name = selected_index['index_endpoint_name']

#                 my_index_endpoint = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name)

#                 my_index_endpoint_public_domain = my_index_endpoint.public_endpoint_domain_name


#                 API_ENDPOINT=my_index_endpoint_public_domain
#                 INDEX_ENDPOINT=index_endpoint_name

#                 # Build FindNeighborsRequest
#                 datapoint = aiplatform_v1.IndexDatapoint(
#                     feature_vector=query_embedding
#                 )

#                 find_neighbors_query = aiplatform_v1.FindNeighborsRequest.Query(
#                     datapoint=datapoint,
#                     neighbor_count=3
#                 )

#                 find_neighbors_request = aiplatform_v1.FindNeighborsRequest(
#                     index_endpoint=index_endpoint_name,
#                     deployed_index_id=deployed_index_id,
#                     queries=[find_neighbors_query],
#                     return_full_datapoint=False
#                 )

#                 if not API_ENDPOINT:                
#                     st.write('API_ENDPOINT /n', API_ENDPOINT)
#                 # Initialize MatchServiceClient
#                 match_service_client = aiplatform_v1.MatchServiceClient(
#                     client_options={"api_endpoint": API_ENDPOINT}
#                 )

#                 # Execute the request
#                 response = match_service_client.find_neighbors(find_neighbors_request)


#                 if not response:                                
#                     st.write("Response",response.nearest_neighbors)

#                 # Prepare a DataFrame to store results
#                 results = []

#                 for result in response.nearest_neighbors:
#                     for neighbor in result.neighbors:
#                         if not neighbor :
#                             st.write("neighbor",neighbor)                        
#                         clip_id = int(neighbor.datapoint.datapoint_id)
#                         distance = neighbor.distance
#                         df_match = shots_df.loc[shots_df.index == clip_id]
#                         if not df_match.empty:
#                             match_info = df_match.iloc[0].to_dict()
#                             match_info['distance'] = distance
#                             results.append(match_info)

#                 # Convert results to DataFrame
#                 df_new = pd.DataFrame(results)

#                 if not results:                                                
#                     st.write('df_new', results , df_new)

#                 # Sort by distance
#                 df_sorted = df_new.sort_values(by="distance", ascending=True)
#                 st.write("Top 3 Matching clips:")
#                 st.dataframe(df_sorted[["clip_name", "description", "distance"]])

            # Display each video with label
            # for index, row in df_sorted.iterrows():
            #     st.write(f"**Clip Name:** {row['clip_name']}")
            #     video_path = CLIPS_DIR + row['clip_name']
            #     st.video(video_path)


        except Exception as e:
            st.error(f"Error during query execution: {e}")
            st.error(f"API Endpoint: {API_ENDPOINT}")
            st.error(f"Index Endpoint: {index_endpoint_name}")
            st.error(f"Deployed Index ID: {deployed_index_id}")

        
        
    # else:
    #     st.warning("Please create an index first.")
    
    vertexai.init(project=PROJECT_ID, location="us-central1")

    
# Video List tab shows list of videos that have been converted into embeddings (Currently not dynamic, for demo purposes only)
with mainTab2:
    
    videoList = []
    
    for root, _, files in os.walk("./content/videos"):
        for file in files:
            if file.endswith(('.mp4', '.avi', '.mov', '.mkv')):  # Add more formats as needed
                video_file_path = os.path.join(root, file)
                video_clip = mpe.VideoFileClip(video_file_path)
                videoList.append((file,video_clip))
                
    st.title("Video List")
    
    videoCount = 0
    
    for video_name, video_clip in videoList:
        video_file = open(f"./content/videos/{video_name}", "rb")
        video_bytes = video_file.read()
        
        videoCount += 1
        
        st.write(f"{videoCount}. {video_name}")
        st.video(video_bytes)
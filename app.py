import streamlit as st
import boto3
import json
import time
import os
from io import StringIO
from botocore.exceptions import NoCredentialsError, ClientError
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Bedrock
import requests

# AWS Credentials


# Initialize AWS clients
s3_client = boto3.client('s3', 
                         aws_access_key_id=aws_access_key_id,
                         aws_secret_access_key=aws_secret_access_key,
                         region_name=region_name)

transcribe_client = boto3.client('transcribe',
                                 aws_access_key_id=aws_access_key_id,
                                 aws_secret_access_key=aws_secret_access_key,
                                 region_name=region_name)

bedrock_client = boto3.client('bedrock-runtime',
                              aws_access_key_id=aws_access_key_id,
                              aws_secret_access_key=aws_secret_access_key,
                              region_name=region_name)

st.title("Video to Fill-in-the-Blank Question Generator")
BUCKET_NAME = 'video-to-text-generate--1'

# Function to create S3 bucket if it doesn't exist
def create_bucket(bucket_name):
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        if e.response['Error']['Code'] == '404':
            try:
                s3_client.create_bucket(Bucket=bucket_name)
            except ClientError as e:
                st.error(f"Failed to create bucket: {e}")
        else:
            st.error(f"Error checking bucket: {e}")

create_bucket(BUCKET_NAME)

# Function to upload file to S3
def upload_to_s3(file):
    try:
        s3_client.upload_fileobj(file, BUCKET_NAME, file.name)
        st.success('File uploaded successfully!')
    except ClientError as e:
        st.error(f"Failed to upload file to S3: {e}")

uploaded_file = st.file_uploader("Choose a video file...", type=["mp4", "mov", "avi"])

if uploaded_file is not None:
    progress_bar = st.progress(0)
    status_text = st.empty()

    status_text.text("Uploading file to S3...")
    upload_to_s3(uploaded_file)
    progress_bar.progress(10)

    transcribe_job_name = f"transcribe_{int(time.time())}"
    try:
        status_text.text("Starting transcription job...")
        transcribe_client.start_transcription_job(
            TranscriptionJobName=transcribe_job_name,
            Media={'MediaFileUri': f's3://{BUCKET_NAME}/{uploaded_file.name}'},
            MediaFormat=uploaded_file.name.split('.')[-1],
            LanguageCode='en-US'
        )
        progress_bar.progress(20)

        status_text.text("Transcribing audio...")
        start_time = time.time()
        while True:
            status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            if time.time() - start_time > 360:  # Stop after 6 minutes
                raise TimeoutError("Transcription job took too long")
            time.sleep(5)
            progress_bar.progress(min(20 + int((time.time() - start_time) * 0.5), 70))

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            status_text.text("Fetching transcription...")
            transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            response = requests.get(transcription_url)
            transcript_text = json.loads(response.text)['results']['transcripts'][0]['transcript']
            progress_bar.progress(80)

            st.subheader("Transcribed Text")
            st.text_area("", transcript_text, height=300)

            status_text.text("Generating Fill-in-the-Blank questions...")
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            texts = text_splitter.split_text(transcript_text)

            prompt_template = PromptTemplate(
                input_variables=["text"],
                template="""Generate fill-in-the-blank questions from the following text. 
                Each question should be on a new line and should have exactly one blank represented by an underscore (_____).
                Each question should in a order with the number
                Also provide the correct answer for each blank after the question, separated by a colon (:).
                Text: {text}
                Fill-in-the-Blank Questions with Answers:"""
            )

            llm = Bedrock(
                model_id="anthropic.claude-v2", 
                client=bedrock_client,
                model_kwargs={"max_tokens_to_sample": 500}
            )
            chain = LLMChain(llm=llm, prompt=prompt_template)

            st.subheader("Generated Fill-in-the-Blank Questions with Answers")
            all_questions = []
            for i, text in enumerate(texts):
                try:
                    if isinstance(text, str):
                        result = chain.run(text)
                        all_questions.append(result)
                    else:
                        st.write(f"Skipping non-string input: {type(text)}")
                except Exception as e:
                    st.error(f"Error processing text: {e}")
                progress_bar.progress(min(80 + int((i + 1) / len(texts) * 20), 100))
                status_text.text(f"Generating questions... {i+1}/{len(texts)}")

            st.text_area("", "\n\n".join(all_questions), height=500)
            status_text.text("Process completed!")
            progress_bar.progress(100)
        else:
            st.error("Transcription job failed.")
    except TimeoutError:
        st.error("Transcription job timed out. Please try again with a shorter video.")
    except ClientError as e:
        st.error(f"Error in transcription process: {e}")
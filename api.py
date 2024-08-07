from flask import Flask, request, jsonify
import boto3
import json
import time
import os
from botocore.exceptions import NoCredentialsError, ClientError
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Bedrock
import requests

app = Flask(__name__)

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
                return jsonify({"error": f"Failed to create bucket: {e}"}), 500
        else:
            return jsonify({"error": f"Error checking bucket: {e}"}), 500

create_bucket(BUCKET_NAME)

@app.route('/')
def home():
    return jsonify({
        "message": "Welcome to the Video to Fill-in-the-Blank Question Generator API",
        "endpoints": {
            "/process-video": "POST - Upload and process a video file to generate questions"
        }
    })

@app.route('/process-video', methods=['POST'])
def process_video():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    try:
        # Upload file to S3
        filename = f"video_{int(time.time())}.{file.filename.split('.')[-1]}"
        s3_client.upload_fileobj(file, BUCKET_NAME, filename)
        
        # Start transcription job
        transcribe_job_name = f"transcribe_{int(time.time())}"
        transcribe_client.start_transcription_job(
            TranscriptionJobName=transcribe_job_name,
            Media={'MediaFileUri': f's3://{BUCKET_NAME}/{filename}'},
            MediaFormat=filename.split('.')[-1],
            LanguageCode='en-US'
        )

        # Wait for transcription job to complete
        max_wait_time = 600  # 10 minutes
        start_time = time.time()
        while True:
            if time.time() - start_time > max_wait_time:
                return jsonify({"error": "Transcription job timed out"}), 504
            
            status = transcribe_client.get_transcription_job(TranscriptionJobName=transcribe_job_name)
            if status['TranscriptionJob']['TranscriptionJobStatus'] in ['COMPLETED', 'FAILED']:
                break
            time.sleep(10)

        if status['TranscriptionJob']['TranscriptionJobStatus'] == 'COMPLETED':
            transcription_url = status['TranscriptionJob']['Transcript']['TranscriptFileUri']
            response = requests.get(transcription_url)
            transcript_text = json.loads(response.text)['results']['transcripts'][0]['transcript']

            questions = generate_numbered_questions(transcript_text)

            return jsonify({
                "transcript": transcript_text,
                "questions": questions,
                "question_count": len(questions)
            }), 200
        else:
            return jsonify({"error": "Transcription job failed"}), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def generate_numbered_questions(text):
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    texts = text_splitter.split_text(text)

    prompt_template = PromptTemplate(
        input_variables=["text", "start_number"],
        template="""Generate as many fill-in-the-blank questions as possible from the following text, aiming for at least 10 questions. 
        Start numbering from {start_number}.
        Each question should be numbered and have exactly one blank represented by an underscore (_____).
        Provide the correct answer for each blank after the question, separated by a colon (:).
        Make sure to cover different aspects of the text and vary the difficulty of the questions.
        Text: {text}
        Numbered Fill-in-the-Blank Questions with Answers:"""
    )

    llm = Bedrock(
        model_id="anthropic.claude-v2", 
        client=bedrock_client,
        model_kwargs={"max_tokens_to_sample": 2000}  # Increased token limit
    )
    chain = LLMChain(llm=llm, prompt=prompt_template)

    all_questions = []
    start_number = 1
    for chunk in texts:
        if isinstance(chunk, str):
            result = chain.run(text=chunk, start_number=start_number)
            questions = result.strip().split('\n')
            all_questions.extend(questions)
            start_number += len(questions)

    return all_questions

if __name__ == '__main__':
    app.run(debug=True)